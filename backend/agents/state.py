from typing import List, Dict, Optional, Any, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Define models that could be used inside the state
class ResumeStateData(BaseModel):
    name: str = ""
    email: str = ""
    skills: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    education: List[str] = Field(default_factory=list)
    raw_text: str = ""
    additional_info: Dict[str, str] = Field(default_factory=dict)

class JobRoleMatch(BaseModel):
    role_name: str
    match_reasoning: str

class CrawledJob(BaseModel):
    title: str
    company: str
    url: str
    source: str
    description: str

class FormQuestion(BaseModel):
    question_identifier: str
    question_text: str
    proposed_answer: str
    is_unknown: bool = False

# This is the overall LangGraph state
class AgentState(TypedDict, total=False):
    candidate_id: str
    resume_data: ResumeStateData
    
    # Role Suggestion Phase
    suggested_roles: List[JobRoleMatch]
    user_selected_roles: List[str]
    
    # Location & Preferences
    location: str          # e.g. "India", "US", "Worldwide"
    cities: List[str]      # e.g. ["Mumbai", "Bangalore"]
    work_type: str         # "Remote", "Hybrid", "In-Office", "Any"
    
    # Job Crawling Phase
    crawled_jobs: List[CrawledJob]
    selected_job_url: str
    
    # Form Filling Phase
    form_questions: List[FormQuestion]
    requires_human_approval: bool
    human_approved: bool
    application_status: str # "started", "waiting_approval", "submitted", "failed"
