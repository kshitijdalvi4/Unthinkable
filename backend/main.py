import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import json
import re
import time

from google.oauth2 import id_token
from google.auth.transport import requests

from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
import chromadb
import numpy as np

# V2 Imports
from agents.graph import build_discovery_graph, build_application_graph
from agents.state import AgentState, ResumeStateData

# V3 Imports
from automation.browser_agent import auto_fill_job

# MongoDB
from db import get_candidate, upsert_candidate, update_additional_info, save_application, get_applications, get_candidate_by_email

# Initialize V2 Graphs
discovery_graph = build_discovery_graph()
application_graph = build_application_graph()

# FastAPI
app = FastAPI(title="Smart Resume Screener API")

# CORS - Allow all for deployment ease
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Custom Gemini Embeddings Class
class GeminiEmbeddings(Embeddings):
    """Custom embeddings class using Gemini API"""
    
    def __init__(self, client):
        self.client = client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            try:
                # User requested gemini-embedding-001
                response = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config={'output_dimensionality': 768}
                )
                embeddings.append(response.embeddings[0].values)
            except Exception as e:
                print(f"Error embedding document: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            # User requested gemini-embedding-001
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config={'output_dimensionality': 768}
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 768

# Initialize embeddings
embeddings = GeminiEmbeddings(gemini_client)

# ChromaDB setup
CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Pydantic Models
class JobDescription(BaseModel):
    title: str
    description: str
    required_skills: List[str]
    experience_years: Optional[int] = 0

class ResumeData(BaseModel):
    candidate_id: str
    name: str
    email: Optional[str]
    skills: List[str]
    experience_years: float
    education: List[str]
    raw_text: str

class MatchResult(BaseModel):
    candidate_id: str
    candidate_name: str
    match_score: float
    justification: str
    matched_skills: List[str]
    missing_skills: List[str]
    experience_match: bool

class ChatQuery(BaseModel):
    candidate_id: str
    question: str

class GoogleToken(BaseModel):
    token: str

# Helper Functions
import time
from typing import Optional

# Add retry decorator
def retry_on_503(max_retries=3, delay=2):
    """Decorator to retry on 503 errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if '503' in error_str and attempt < max_retries - 1:
                        print(f"503 error, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

def clean_json_response(text: str) -> str:
    """Clean Gemini response to extract JSON"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Find JSON object - more aggressive matching
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str
    
    # If no match, try to fix the text directly
    # Remove leading text before {
    if '{' in text:
        text = text[text.index('{'):]
    # Remove trailing text after }
    if '}' in text:
        text = text[:text.rindex('}') + 1]
    
    return text.strip()

@retry_on_503(max_retries=3, delay=2)
def call_gemini_sync(prompt: str, model: str = None) -> str:
    """Call Gemini with retry logic (Synchronous)"""
    if model is None:
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    print(f"[LLM] Calling {model}...")
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text

async def call_gemini(prompt: str, model: str = None) -> str:
    """Async wrapper for call_gemini_sync"""
    return await asyncio.to_thread(call_gemini_sync, prompt, model)

async def extract_resume_data(file_path: str, candidate_id: str, force_extract: bool = False) -> dict:
    """Extract structured data from resume using Gemini or load from MongoDB cache."""
    # --- MongoDB cache lookup ---
    if not force_extract:
        cached = await get_candidate(candidate_id)
        if cached:
            print(f"[CACHE] MongoDB hit for {candidate_id}. additional_info: {cached.get('additional_info')}")
            return cached

    # --- Full Gemini extraction ---
    try:
        loader = PyPDFLoader(file_path)
        documents = await asyncio.to_thread(loader.load)
        
        full_text = "\n".join([doc.page_content for doc in documents])
        
        extraction_prompt = """Extract candidate information from this resume as JSON.

Resume:
{resume_text}

Return this JSON (only JSON, no extra text):
{{"name":"Full Name","email":"email@example.com","skills":["Python","Java"],"experience_years":3,"education":["Degree","University"]}}

Rules:
- name: Get from top of resume
- email: Extract email address
- skills: All technical skills as array
- experience_years: Total years as number
- education: Degrees/schools as array
"""
        
        result_text = await call_gemini(extraction_prompt.format(resume_text=full_text[:6000]))
        result = clean_json_response(result_text)
        
        try:
            data = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            data = {}
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', result_text)
            if name_match:
                data['name'] = name_match.group(1)
            email_match = re.search(r'"email"\s*:\s*"([^"]*)"', result_text)
            if email_match:
                data['email'] = email_match.group(1)
            skills_match = re.search(r'"skills"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
            if skills_match:
                data['skills'] = [s.strip(' "\'') for s in skills_match.group(1).split(',') if s.strip()]
            else:
                data['skills'] = []
            exp_match = re.search(r'"experience_years"\s*:\s*(\d+\.?\d*)', result_text)
            data['experience_years'] = float(exp_match.group(1)) if exp_match else 0.0
            edu_match = re.search(r'"education"\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
            if edu_match:
                data['education'] = [s.strip(' "\'') for s in edu_match.group(1).split(',') if s.strip()]
            else:
                data['education'] = []
        
        # Fallback name extraction: Look for most likely name in first 10 lines
        if not data.get('name') or data.get('name').lower() in ['unknown', 'candidate', '', 'full name']:
            print("[EXTRACT] LLM missed name. Attempting fallback extraction...")
            lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            for line in lines[:10]:
                # Heuristic: First non-empty short line (2-4 words) that doesn't look like contact info
                words = line.split()
                if 1 <= len(words) <= 5 and not any(c in line for c in ['@', 'http', ':', '/', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    data['name'] = line
                    print(f"[EXTRACT] Fallback found name: {line}")
                    break
        
        data['raw_text'] = full_text
        data['pdf_path'] = file_path
        
        # Initialize Knowledge Base with extracted data
        data['knowledge_base'] = {
            "name": data.get("name", ""),
            "email": data.get("email", ""),
            "skills": data.get("skills", []),
            "experience_years": data.get("experience_years", 0),
            "phone": "",
            "expected_salary": "",
            "current_ctc": "",
            "notice_period": "",
        }
        data['additional_info'] = {}

        # --- Deduplication logic ---
        email = data.get('email')
        if email:
            existing = await get_candidate_by_email(email)
            if existing:
                print(f"[DB] Deduplication hit for email: {email}. Merging with existing ID: {existing['_id']}")
                # Merge: Keep existing answers, but prioritize new extracted data for core fields if "Unknown"
                merged_data = {**existing, **data}
                # Ensure we use the freshly extracted info if the old one was "Unknown"
                if existing.get("name") == "Unknown" and data.get("name"):
                    merged_data["name"] = data["name"]
                
                await upsert_candidate(existing['_id'], merged_data)
                merged_data['_id'] = existing['_id']
                return merged_data

        # Save to MongoDB
        await upsert_candidate(candidate_id, data)
        data['_id'] = candidate_id
        print(f"[CACHE] Saved to MongoDB for {candidate_id}")
        return data
        
    except Exception as e:
        print(f"Error extracting resume data: {e}")
        return {
            "name": "Unknown",
            "email": "",
            "skills": [],
            "experience_years": 0.0,
            "education": [],
            "raw_text": full_text if 'full_text' in locals() else "",
            "additional_info": {}
        }

def compute_match_score(resume_data: dict, job_desc: JobDescription) -> MatchResult:
    """Use Gemini to compute semantic match between resume and job"""
    
    # Extract actual name from resume text if still "Unknown"
    candidate_name = resume_data.get('name', 'Unknown')
    if candidate_name.lower() in ['unknown', 'candidate', '']:
        # Try to extract from raw text
        first_lines = resume_data.get('raw_text', '').split('\n')[:10]
        for line in first_lines:
            line = line.strip()
            # Look for name pattern (2-4 words, capitalized, no special chars)
            if line and 2 <= len(line.split()) <= 4 and line[0].isupper() and not any(c in line for c in ['@', 'http', ':', '/']):
                candidate_name = line
                resume_data['name'] = line
                break
    
    matching_prompt = f"""
You are an expert HR recruiter. Compare this candidate's resume with the job description.

JOB REQUIREMENTS:
Title: {job_desc.title}
Description: {job_desc.description}
Required Skills: {', '.join(job_desc.required_skills)}
Required Experience: {job_desc.experience_years} years

CANDIDATE PROFILE:
Name: {candidate_name}
Skills: {', '.join(resume_data.get('skills', []))}
Experience: {resume_data.get('experience_years', 0)} years
Education: {', '.join(resume_data.get('education', []))}

Resume excerpt:
{resume_data.get('raw_text', '')[:2500]}

Provide a detailed analysis in JSON format (return ONLY the JSON, no markdown):

{{
  "match_score": 7.5,
  "justification": "Brief 2-3 sentence summary without using asterisks or markdown. Use plain text only.",
  "matched_skills": ["skills candidate has that match requirements"],
  "missing_skills": ["required skills candidate lacks"],
  "experience_match": true
}}

IMPORTANT: 
- Write justification in plain text without any markdown formatting
- Do not use asterisks (*) or other markdown symbols
- Use natural sentences with proper punctuation

Scoring guide:
- 9-10: Perfect fit, exceeds requirements
- 7-8: Strong fit, meets most requirements
- 5-6: Moderate fit, some gaps
- 3-4: Weak fit, significant gaps
- 0-2: Poor fit, major misalignment

Return ONLY the JSON object, no other text or formatting.
"""
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=matching_prompt,
        )
        
        result = clean_json_response(response.text)
        match_data = json.loads(result)
        
        return MatchResult(
            candidate_id=resume_data.get('candidate_id', ''),
            candidate_name=resume_data.get('name', 'Unknown'),
            match_score=float(match_data.get('match_score', 0)),
            justification=match_data.get('justification', ''),
            matched_skills=match_data.get('matched_skills', []),
            missing_skills=match_data.get('missing_skills', []),
            experience_match=bool(match_data.get('experience_match', False))
        )
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in match: {e}")
        print(f"Response text: {response.text[:500]}")
        return MatchResult(
            candidate_id=resume_data.get('candidate_id', ''),
            candidate_name=resume_data.get('name', 'Unknown'),
            match_score=0.0,
            justification=f"Error parsing match response",
            matched_skills=[],
            missing_skills=[],
            experience_match=False
        )
    except Exception as e:
        print(f"Error computing match: {e}")
        return MatchResult(
            candidate_id=resume_data.get('candidate_id', ''),
            candidate_name=resume_data.get('name', 'Unknown'),
            match_score=0.0,
            justification=f"Error analyzing candidate: {str(e)}",
            matched_skills=[],
            missing_skills=[],
            experience_match=False
        )

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart Resume Screener API", 
        "status": "running",
        "model": "Gemini 2.5 Flash",
        "embeddings": "Gemini text-embedding-004"
    }

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...), candidate_id: Optional[str] = Form(None)):
    """Upload and process a single resume. If candidate_id is provided, it overwrites the existing record."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Save uploaded file
        if not candidate_id:
            candidate_id = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        file_path = f"./uploads/{candidate_id}.pdf"
        os.makedirs("./uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract data (async — force re-extraction for new file)
        processed_data = await extract_resume_data(file_path, candidate_id, force_extract=True)
        
        # In case deduplication used a different ID
        effective_id = processed_data.get('_id', candidate_id)
        processed_data['candidate_id'] = effective_id
        
        # Store in ChromaDB
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)
        
        # Add metadata
        for doc in docs:
            doc.metadata['candidate_id'] = effective_id
            doc.metadata['candidate_name'] = processed_data.get('name', 'Unknown')
        
        # Store in vector DB (non-fatal if it fails - resume data is still returned)
        try:
            # Delete old chunks for this candidate to ensure "most recent resume" is used for RAG
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name="resumes"
            )
            # Use delete with filter
            try:
                vectorstore.delete(where={"candidate_id": effective_id})
                print(f"[RAG] Deleted old chunks for {effective_id}")
            except Exception as e:
                print(f"[RAG] Skip deletion (collection may be empty): {e}")

            # Store new chunks
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_name="resumes"
            )
            print(f"[RAG] Ingested {len(docs)} new chunks for {effective_id}")
        except Exception as chroma_err:
            print(f"[WARN] ChromaDB storage failed (non-fatal): {chroma_err}")
        
        return {
            "candidate_id": effective_id,
            "message": "Resume uploaded successfully",
            "data": processed_data
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/match-job/")
async def match_job(job_desc: JobDescription):
    """Match all uploaded resumes against a job description using MongoDB"""
    
    try:
        from db import get_db
        cursor = get_db()["candidates"].find({}, {"_id": 1, "name": 1, "skills": 1, "experience_years": 1, "education": 1, "raw_text": 1})
        results = []
        async for doc in cursor:
            candidate_id = doc["_id"]
            file_path = f"./uploads/{candidate_id}.pdf"
            resume_data = dict(doc)
            resume_data["candidate_id"] = candidate_id
            match_result = compute_match_score(resume_data, job_desc)
            results.append(match_result)

        results.sort(key=lambda x: x.match_score, reverse=True)
        return {
            "total_candidates": len(results),
            "job_title": job_desc.title,
            "matches": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

@app.post("/chat-resume/")
async def chat_resume(query: ChatQuery):
    """Ask questions about a specific resume using RAG"""
    
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="resumes"
        )
        
        # Get relevant documents for this candidate
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"candidate_id": query.candidate_id}
            }
        )
        
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query.question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if not context.strip():
            return {
                "candidate_id": query.candidate_id,
                "question": query.question,
                "answer": "No resume data found for this candidate."
            }
        
        prompt = f"""Use the following pieces of context from the candidate's resume to answer the question.
If you don't know the answer based on the context, just say you don't know. Don't make up information.

IMPORTANT: Provide your answer in plain text format without using markdown symbols like asterisks (*), bullets, or special formatting. Write in natural, flowing sentences.

Context from resume:
{context}

Question: {query.question}

Answer (plain text only, no markdown):"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        return {
            "candidate_id": query.candidate_id,
            "question": query.question,
            "answer": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/auth/google/")
async def google_auth(data: GoogleToken):
    """Verify Google ID token and return/create candidate profile."""
    try:
        # Verify the token
        # In a real production app, CLIENT_ID should be in .env
        CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        idinfo = id_token.verify_oauth2_token(data.token, requests.Request(), CLIENT_ID)
        
        email = idinfo.get('email')
        name = idinfo.get('name', 'Google User')
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in Google token")
            
        # Check if user exists
        candidate = await get_candidate_by_email(email)
        
        if candidate:
            print(f"[AUTH] Returning user found: {email}")
            return {
                "candidate_id": candidate["_id"],
                "candidate_name": candidate.get("name", name),
                "profile": candidate.get("knowledge_base", {}),
                "has_resume": True
            }
        else:
            print(f"[AUTH] New user via Google: {email}")
            # Create a stub profile
            candidate_id = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            stub_profile = {
                "name": name,
                "email": email,
                "skills": [],
                "experience_years": 0.0,
                "phone": "",
                "expected_salary": "",
                "current_ctc": "",
                "notice_period": ""
            }
            # Upsert into MongoDB
            from db import upsert_candidate
            await upsert_candidate(candidate_id, {"email": email, "name": name, "knowledge_base": stub_profile})
            
            return {
                "candidate_id": candidate_id,
                "candidate_name": name,
                "profile": stub_profile,
                "has_resume": False
            }
            
    except ValueError as e:
        # Invalid token
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/candidates/")
async def get_all_candidates():
    """Get list of all uploaded candidates from MongoDB"""
    try:
        from db import get_db
        cursor = get_db()["candidates"].find({}, {"_id": 1, "name": 1})
        candidates = []
        async for doc in cursor:
            candidates.append({
                "candidate_id": doc["_id"],
                "candidate_name": doc.get("name", "Unknown")
            })
        return {"candidates": candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {str(e)}")

@app.delete("/candidate/{candidate_id}")
async def delete_candidate(candidate_id: str):
    """Delete a candidate's resume"""
    try:
        file_path = f"./uploads/{candidate_id}.pdf"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        os.remove(file_path)
        
        return {
            "success": True,
            "message": f"Candidate {candidate_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "Gemini 2.5 Flash",
        "embeddings": "Gemini text-embedding-004",
        "chroma_path": CHROMA_PATH
    }

# --- V2 MODELS ---

class V2SuggestRolesRequest(BaseModel):
    candidate_id: str

class V2DiscoverJobsRequest(BaseModel):
    candidate_id: str
    user_selected_roles: Optional[List[str]] = None
    location: Optional[str] = "Worldwide"
    cities: Optional[List[str]] = None
    work_type: Optional[str] = "Any"

class V2ApplyJobRequest(BaseModel):
    candidate_id: str
    selected_job_url: str
    job_title: Optional[str] = None

class V2ApproveApplicationRequest(BaseModel):
    thread_id: str
    candidate_id: str
    answers: Optional[dict] = None  # user-edited answers keyed by question_identifier

# --- V2 ENDPOINTS ---

@app.post("/api/v2/suggest-roles/")
async def v2_suggest_roles(request: V2SuggestRolesRequest):
    """Step 2a: Suggest roles based on the resume only (no crawling yet)"""
    # Use MongoDB as source of truth — file still needed for ChromaDB RAG
    resume_data = await extract_resume_data(f"./uploads/{request.candidate_id}.pdf", request.candidate_id)
    if not resume_data or resume_data.get('name') == 'Unknown' and not resume_data.get('email'):
        raise HTTPException(status_code=404, detail="Candidate not found")
    try:
        resume_state = ResumeStateData(**{
            k: v for k, v in resume_data.items()
            if k in ResumeStateData.model_fields
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State error: {str(e)}")

    from agents.agents import suggest_roles_node
    state = AgentState(
        candidate_id=request.candidate_id,
        resume_data=resume_state
    )
    result = suggest_roles_node(state)
    return {
        "suggested_roles": [r.model_dump() for r in result.get("suggested_roles", [])]
    }

@app.post("/api/v2/discover-jobs/")
async def v2_discover_jobs(request: V2DiscoverJobsRequest):
    """Step 3: Crawl jobs for the user-confirmed roles"""
    resume_data = await extract_resume_data(f"./uploads/{request.candidate_id}.pdf", request.candidate_id)
    if not resume_data:
        raise HTTPException(status_code=404, detail="Candidate not found")
    try:
        resume_state = ResumeStateData(**{
            k: v for k, v in resume_data.items()
            if k in ResumeStateData.model_fields
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State error: {str(e)}")

    state = AgentState(
        candidate_id=request.candidate_id,
        resume_data=resume_state,
        user_selected_roles=request.user_selected_roles or [],
        location=request.location or "Worldwide",
        cities=request.cities or [],
        work_type=request.work_type or "Any"
    )

    # If user provided roles, skip suggestion and go straight to crawling
    if request.user_selected_roles:
        from agents.agents import crawl_jobs_node
        result = crawl_jobs_node(state)
        return {
            "suggested_roles": [],
            "crawled_jobs": [j.model_dump() for j in result.get("crawled_jobs", [])]
        }

    result = discovery_graph.invoke(state)
    return {
        "suggested_roles": [r.model_dump() for r in result.get("suggested_roles", [])],
        "crawled_jobs": [j.model_dump() for j in result.get("crawled_jobs", [])]
    }

@app.post("/api/v2/apply-job/")
async def v2_apply_job(request: V2ApplyJobRequest):
    # Pass the ID — extract_resume_data will handle deduplication and return the canonical doc
    resume_data = await extract_resume_data(f"./uploads/{request.candidate_id}.pdf", request.candidate_id)
    if not resume_data:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # The ID might have changed due to deduplication
    canonical_id = resume_data.get('_id', request.candidate_id)
    
    resume_state = ResumeStateData(**{k: v for k, v in resume_data.items() if k in ResumeStateData.model_fields})
    
    thread_id = f"thread_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = AgentState(
        candidate_id=canonical_id,
        resume_data=resume_state,
        selected_job_url=request.selected_job_url
    )
    
    # Run heavy LangGraph invocation in a separate thread to keep event loop responsive
    result = await asyncio.to_thread(application_graph.invoke, state, config)
    
    return {
        "candidate_id": canonical_id, # Return canonical ID to frontend
        "thread_id": thread_id,
        "status": result.get("application_status", "running"),
        "requires_human_approval": result.get("requires_human_approval", False),
        "form_questions": [q.model_dump() for q in result.get("form_questions", [])]
    }

@app.post("/api/v2/approve-application/")
async def v2_approve_application(request: V2ApproveApplicationRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # ── Persist answers to MongoDB ──────────────────────────────────────────
    if request.answers:
        print(f"[MONGO] Saving answers for {request.candidate_id}: {request.answers}")
        await update_additional_info(request.candidate_id, request.answers)
        print(f"[MONGO] Answers saved.")

    # ── Log the application ────────────────────────────────────────────────
    await save_application(
        candidate_id=request.candidate_id,
        job_url=request.answers.get("job_url", "") if request.answers else "",
        job_title=request.answers.get("job_title", "") if request.answers else "",
        answers=request.answers or {},
        status="logged"
    )

    # ── Resume the LangGraph ───────────────────────────────────────────────
    # Update state to human_approved=True before resuming invocation
    application_graph.update_state(config, {"human_approved": True})
    
    # Run heavy LangGraph invocation in a separate thread
    result = await asyncio.to_thread(application_graph.invoke, None, config)
    
    return {
        "application_status": result.get("application_status")
    }


# ─── V3: Phase 3 Browser Automation ──────────────────────────────────────────

# Simple in-memory status tracker
automation_statuses = {}

class V3AutoSubmitRequest(BaseModel):
    candidate_id: str
    job_url: str
    job_title: Optional[str] = None
    credentials: Optional[dict] = None

async def run_automation_task(candidate_id: str, job_url: str, job_title: str, credentials: Optional[dict]):
    import sys
    import os
    import traceback
    
    # Ensure current directory is in path for subprocesses
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
        
    log_file = "automation_error.log"
    
    automation_statuses[candidate_id] = {"status": "running", "message": "Browser agent started..."}
    print(f"[P3] Starting automation task for {candidate_id} on {job_url}")
    
    try:
        from db import get_candidate, save_application
        from automation.browser_agent import auto_fill_job
        
        candidate_data = await get_candidate(candidate_id)
        if not candidate_data:
            msg = f"Candidate {candidate_id} not found in DB"
            print(f"[P3] Error: {msg}")
            automation_statuses[candidate_id] = {"status": "error", "message": msg}
            return

        print(f"[P3] Dispatching to auto_fill_job for {job_url}...")
        result = await auto_fill_job(
            job_url=job_url,
            candidate_data=candidate_data,
            credentials=credentials,
            headless=False,
        )

        print(f"[P3] Automation completed with status: {result.get('status')}")
        
        # Log automation attempt
        await save_application(
            candidate_id=candidate_id,
            job_url=job_url,
            job_title=job_title or "",
            answers=candidate_data.get("additional_info", {}),
            status=result.get("status", "unknown"),
        )
        
        automation_statuses[candidate_id] = {
            "status": "completed",
            "message": result.get("message"),
            "res": result
        }
    except Exception as e:
        print(f"[P3] CRITICAL ERROR in run_automation_task: {e}")
        with open(log_file, "a") as f:
            f.write(f"\n--- {datetime.now()} ---\n")
            f.write(f"Candidate: {candidate_id}, Job: {job_url}\n")
            f.write(traceback.format_exc())
            f.write("-" * 30 + "\n")
        
        automation_statuses[candidate_id] = {"status": "error", "message": f"Critical error: {str(e)}"}

@app.get("/api/v3/automation-status/{candidate_id}")
async def get_automation_status(candidate_id: str):
    return automation_statuses.get(candidate_id, {"status": "idle"})

@app.post("/api/v3/auto-submit/")
async def v3_auto_submit(request: V3AutoSubmitRequest, background_tasks: BackgroundTasks):
    """
    Phase 3: Launches Playwright browser in background.
    """
    # Quick check if candidate exists
    from db import get_candidate, get_candidate_by_email
    candidate_data = await get_candidate(request.candidate_id)
    
    # Recovery logic
    if not candidate_data and request.credentials and request.credentials.get("email"):
        candidate_data = await get_candidate_by_email(request.credentials.get("email"))
    
    if not candidate_data:
        raise HTTPException(status_code=404, detail="Candidate not found in MongoDB. Please re-upload.")

    background_tasks.add_task(
        run_automation_task, 
        candidate_data["_id"], 
        request.job_url, 
        request.job_title or "", 
        request.credentials
    )

    return {
        "status": "started",
        "message": "Playwright agent launched in background. Check the browser window!",
        "candidate_id": candidate_data["_id"]
    }


@app.post("/api/v3/update-profile")
async def v3_update_profile(request: dict):
    """
    Update candidate's professional profile knowledge base.
    Expects: {candidate_id, profile: {...}}
    """
    candidate_id = request.get("candidate_id")
    profile = request.get("profile")
    
    if not candidate_id or not profile:
        raise HTTPException(status_code=400, detail="Missing candidate_id or profile data")
    
    from db import update_knowledge_base
    await update_knowledge_base(candidate_id, profile)
    return {"status": "success", "message": "Profile updated in Knowledge Base"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
