from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import json
import re

from google import genai
from google.genai import types
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
import chromadb
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Smart Resume Screener API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "gemini_api_key")

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
                response = self.client.models.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                embeddings.append(response.embedding)
            except Exception as e:
                print(f"Error embedding document: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = self.client.models.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            return response.embedding
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

# Helper Functions
def clean_json_response(text: str) -> str:
    """Clean Gemini response to extract JSON"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text.strip()

def extract_resume_data(file_path: str) -> dict:
    """Extract structured data from resume using Gemini"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        full_text = "\n".join([doc.page_content for doc in documents])
        
        extraction_prompt = """
Extract the following information from this resume and return ONLY a valid JSON object:

{
  "name": "Full name of candidate",
  "email": "Email address or empty string",
  "skills": ["List of technical and soft skills"],
  "experience_years": 0,
  "education": ["List of degrees/certifications"]
}

Resume text:
{resume_text}

Instructions:
- Return ONLY the JSON object, nothing else
- If information is missing, use empty string "" or empty list []
- For experience_years, estimate total years as a number (integer or float)
- Extract all identifiable skills (programming languages, tools, frameworks, soft skills)
"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=extraction_prompt.format(resume_text=full_text[:6000]),
        )
        
        result = clean_json_response(response.text)
        data = json.loads(result)
        data['raw_text'] = full_text
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response.text[:500]}")
        return {
            "name": "Unknown",
            "email": "",
            "skills": [],
            "experience_years": 0,
            "education": [],
            "raw_text": full_text if 'full_text' in locals() else ""
        }
    except Exception as e:
        print(f"Error extracting resume data: {e}")
        return {
            "name": "Unknown",
            "email": "",
            "skills": [],
            "experience_years": 0,
            "education": [],
            "raw_text": ""
        }

def compute_match_score(resume_data: dict, job_desc: JobDescription) -> MatchResult:
    """Use Gemini to compute semantic match between resume and job"""
    
    matching_prompt = f"""
You are an expert HR recruiter. Compare this candidate's resume with the job description.

JOB REQUIREMENTS:
Title: {job_desc.title}
Description: {job_desc.description}
Required Skills: {', '.join(job_desc.required_skills)}
Required Experience: {job_desc.experience_years} years

CANDIDATE PROFILE:
Name: {resume_data.get('name', 'Unknown')}
Skills: {', '.join(resume_data.get('skills', []))}
Experience: {resume_data.get('experience_years', 0)} years
Education: {', '.join(resume_data.get('education', []))}

Resume excerpt:
{resume_data.get('raw_text', '')[:2500]}

Provide a detailed analysis in JSON format (return ONLY the JSON):

{{
  "match_score": 7.5,
  "justification": "Brief 2-3 sentence summary of overall fit",
  "matched_skills": ["skills candidate has that match requirements"],
  "missing_skills": ["required skills candidate lacks"],
  "experience_match": true
}}

Scoring guide:
- 9-10: Perfect fit, exceeds requirements
- 7-8: Strong fit, meets most requirements
- 5-6: Moderate fit, some gaps
- 3-4: Weak fit, significant gaps
- 0-2: Poor fit, major misalignment

Return ONLY the JSON object, no other text.
"""
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
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
        "model": "Gemini 2.0 Flash",
        "embeddings": "Gemini text-embedding-004"
    }

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and process a single resume"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Save uploaded file
        candidate_id = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        file_path = f"./uploads/{candidate_id}.pdf"
        os.makedirs("./uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract data
        resume_data = extract_resume_data(file_path)
        resume_data['candidate_id'] = candidate_id
        
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
            doc.metadata['candidate_id'] = candidate_id
            doc.metadata['candidate_name'] = resume_data.get('name', 'Unknown')
        
        # Store in vector DB
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="resumes"
        )
        
        return {
            "candidate_id": candidate_id,
            "message": "Resume uploaded successfully",
            "data": resume_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/match-job/")
async def match_job(job_desc: JobDescription):
    """Match all uploaded resumes against a job description"""
    
    try:
        # Get all uploaded resume files
        if not os.path.exists("./uploads"):
            return {
                "total_candidates": 0,
                "job_title": job_desc.title,
                "matches": []
            }
        
        resume_files = [f for f in os.listdir("./uploads") if f.endswith('.pdf')]
        
        if not resume_files:
            return {
                "total_candidates": 0,
                "job_title": job_desc.title,
                "matches": []
            }
        
        results = []
        for resume_file in resume_files:
            candidate_id = resume_file.replace('.pdf', '')
            file_path = f"./uploads/{resume_file}"
            
            if os.path.exists(file_path):
                resume_data = extract_resume_data(file_path)
                resume_data['candidate_id'] = candidate_id
                match_result = compute_match_score(resume_data, job_desc)
                results.append(match_result)
        
        # Sort by match score
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

Context from resume:
{context}

Question: {query.question}

Answer:"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
        )
        
        return {
            "candidate_id": query.candidate_id,
            "question": query.question,
            "answer": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/candidates/")
async def get_all_candidates():
    """Get list of all uploaded candidates"""
    
    try:
        if not os.path.exists("./uploads"):
            return {"candidates": []}
        
        resume_files = [f for f in os.listdir("./uploads") if f.endswith('.pdf')]
        
        candidates = []
        for resume_file in resume_files:
            candidate_id = resume_file.replace('.pdf', '')
            file_path = f"./uploads/{resume_file}"
            
            # Try to get candidate name from ChromaDB first
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=embeddings,
                    collection_name="resumes"
                )
                
                collection = vectorstore._collection
                all_docs = collection.get(
                    where={"candidate_id": candidate_id},
                    limit=1
                )
                
                candidate_name = "Unknown"
                if all_docs['metadatas']:
                    candidate_name = all_docs['metadatas'][0].get('candidate_name', 'Unknown')
                
                candidates.append({
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name
                })
            except:
                # Fallback: extract from file
                resume_data = extract_resume_data(file_path)
                candidates.append({
                    "candidate_id": candidate_id,
                    "candidate_name": resume_data.get('name', 'Unknown')
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
        "model": "Gemini 2.0 Flash",
        "embeddings": "Gemini text-embedding-004",
        "chroma_path": CHROMA_PATH
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)