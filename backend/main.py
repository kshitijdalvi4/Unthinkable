from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb

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

# OpenAI API Key
OPENAI_API_KEY = "open_ai_key"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# ChromaDB setup
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)

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
def extract_resume_data(file_path: str) -> dict:
    """Extract structured data from resume using LLM"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    full_text = "\n".join([doc.page_content for doc in documents])
    
    extraction_prompt = """
    Extract the following information from this resume in JSON format:
    - name (candidate's full name)
    - email (email address)
    - skills (list of technical and soft skills)
    - experience_years (total years of experience as a number)
    - education (list of degrees/certifications)
    
    Resume:
    {resume_text}
    
    Return ONLY valid JSON with these exact keys. If information is not found, use empty string or empty list.
    """
    
    result = llm.predict(extraction_prompt.format(resume_text=full_text[:4000]))
    
    try:
        data = json.loads(result)
        data['raw_text'] = full_text
        return data
    except:
        return {
            "name": "Unknown",
            "email": "",
            "skills": [],
            "experience_years": 0,
            "education": [],
            "raw_text": full_text
        }

def compute_match_score(resume_data: dict, job_desc: JobDescription) -> MatchResult:
    """Use LLM to compute semantic match between resume and job"""
    
    matching_prompt = f"""
    You are an expert HR recruiter. Compare this candidate's resume with the job description.
    
    Job Title: {job_desc.title}
    Job Description: {job_desc.description}
    Required Skills: {', '.join(job_desc.required_skills)}
    Required Experience: {job_desc.experience_years} years
    
    Candidate Name: {resume_data.get('name', 'Unknown')}
    Candidate Skills: {', '.join(resume_data.get('skills', []))}
    Candidate Experience: {resume_data.get('experience_years', 0)} years
    
    Resume Summary:
    {resume_data.get('raw_text', '')[:2000]}
    
    Provide:
    1. Match score (0-10) based on skills, experience, and overall fit
    2. Brief justification (2-3 sentences)
    3. List of matched skills (skills the candidate has that match job requirements)
    4. List of missing skills (required skills the candidate lacks)
    5. Whether experience requirement is met (true/false)
    
    Return as JSON with keys: match_score, justification, matched_skills, missing_skills, experience_match
    """
    
    result = llm.predict(matching_prompt)
    
    try:
        match_data = json.loads(result)
        return MatchResult(
            candidate_id=resume_data.get('candidate_id', ''),
            candidate_name=resume_data.get('name', 'Unknown'),
            match_score=float(match_data.get('match_score', 0)),
            justification=match_data.get('justification', ''),
            matched_skills=match_data.get('matched_skills', []),
            missing_skills=match_data.get('missing_skills', []),
            experience_match=match_data.get('experience_match', False)
        )
    except Exception as e:
        return MatchResult(
            candidate_id=resume_data.get('candidate_id', ''),
            candidate_name=resume_data.get('name', 'Unknown'),
            match_score=0.0,
            justification=f"Error computing match: {str(e)}",
            matched_skills=[],
            missing_skills=[],
            experience_match=False
        )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Smart Resume Screener API", "status": "running"}

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and process a single resume"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Save uploaded file
    candidate_id = f"candidate_{datetime.now().timestamp()}"
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
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

@app.post("/match-job/")
async def match_job(job_desc: JobDescription):
    """Match all uploaded resumes against a job description"""
    
    # Load all resumes from ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="resumes"
    )
    
    # Get all unique candidate IDs
    collection = vectorstore._collection
    all_docs = collection.get()
    
    candidate_ids = set()
    for metadata in all_docs['metadatas']:
        if 'candidate_id' in metadata:
            candidate_ids.add(metadata['candidate_id'])
    
    results = []
    for candidate_id in candidate_ids:
        file_path = f"./uploads/{candidate_id}.pdf"
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

@app.post("/chat-resume/")
async def chat_resume(query: ChatQuery):
    """Ask questions about a specific resume using RAG"""
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="resumes"
    )
    
    # Create retriever filtered by candidate_id
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"candidate_id": query.candidate_id}
        }
    )
    
    qa_prompt = PromptTemplate(
        template="""Use the following pieces of context from the candidate's resume to answer the question.
        If you don't know the answer, just say you don't know. Don't make up information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    
    result = qa_chain({"query": query.question})
    
    return {
        "candidate_id": query.candidate_id,
        "question": query.question,
        "answer": result['result']
    }

@app.get("/candidates/")
async def get_all_candidates():
    """Get list of all uploaded candidates"""
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="resumes"
    )
    
    collection = vectorstore._collection
    all_docs = collection.get()
    
    candidates = {}
    for metadata in all_docs['metadatas']:
        if 'candidate_id' in metadata:
            cid = metadata['candidate_id']
            if cid not in candidates:
                candidates[cid] = {
                    "candidate_id": cid,
                    "candidate_name": metadata.get('candidate_name', 'Unknown')
                }
    
    return {"candidates": list(candidates.values())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)