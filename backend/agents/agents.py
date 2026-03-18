import os
import time
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
import json

from agents.state import AgentState, JobRoleMatch, ResumeStateData, CrawledJob, FormQuestion

# Initialize LLM - reads model from .env (LLM_MODEL), defaults to gemini-1.5-flash
# gemini-1.5-flash has much higher free-tier quota: 1500 req/day vs 200 for gemini-2.0-flash
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    print(f"[LLM] Initializing {model}...")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
        request_timeout=30 # Add timeout to prevent forever-hang
    )

class RoleSuggestionsOutput(BaseModel):
    roles: List[JobRoleMatch] = Field(description="A list of 3-5 suggested job roles based on the resume")

def suggest_roles_node(state: AgentState) -> dict:
    """
    Analyzes the resume data and suggests 3-5 job roles the candidate is a fit for.
    """
    resume_data = state.get("resume_data", ResumeStateData())
    
    # If no raw_text or skills, we can't suggest roles properly, return empty or generic
    if not resume_data.raw_text and not resume_data.skills:
        return {"suggested_roles": []}

    llm = get_llm()
    structured_llm = llm.with_structured_output(RoleSuggestionsOutput)

    prompt = PromptTemplate.from_template(
        """Analyze the following candidate's resume and suggest 3 to 5 job roles that match their skills and experience.
        
        Candidate Name: {name}
        Skills: {skills}
        Experience (Years): {experience_years}
        Education: {education}
        
        Resume Text:
        {raw_text}
        
        For each suggested role, provide a brief reasoning (1-2 sentences) explaining why.
        """
    )

    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "name": resume_data.name,
            "skills": ", ".join(resume_data.skills),
            "experience_years": resume_data.experience_years,
            "education": ", ".join(resume_data.education),
            "raw_text": resume_data.raw_text[:4000] # truncate logic if too long, flash can handle 1M though
        })
        return {"suggested_roles": result.roles}
    except Exception as e:
        print(f"Error in suggest_roles_node: {e}")
        return {"suggested_roles": []}

def crawl_jobs_node(state: AgentState) -> dict:
    """
    Crawls job boards for the selected roles using DuckDuckGo Search.
    Incorporates location, cities and work_type preferences into search.
    """
    roles_to_search = state.get("user_selected_roles", [])
    if not roles_to_search:
        suggested = state.get("suggested_roles", [])
        roles_to_search = [r.role_name for r in suggested[:2]]
        
    if not roles_to_search:
        return {"crawled_jobs": []}

    # Build location context for search query
    location = state.get("location", "")
    cities = state.get("cities", [])
    work_type = state.get("work_type", "Any")
    
    location_str = ""
    if cities:
        location_str = " OR ".join(cities)
    elif location and location != "Worldwide":
        location_str = location

    work_str = ""
    if work_type and work_type != "Any":
        work_str = work_type

    all_jobs = []
    
    import threading
    crawl_done = threading.Event()

    def do_crawl():
        try:
            with DDGS() as ddgs:
                for role in roles_to_search:
                    parts = [f"'{role}' job opening"]
                    if location_str:
                        parts.append(location_str)
                    if work_str:
                        parts.append(work_str)
                    parts.append("site:linkedin.com/jobs OR site:jooble.org OR site:indeed.com")
                    query = " ".join(parts)
                    try:
                        results = ddgs.text(query, max_results=4)
                        for res in results:
                            url = res.get("href", "")
                            if "linkedin" in url: source = "LinkedIn"
                            elif "indeed" in url: source = "Indeed"
                            elif "jooble" in url: source = "Jooble"
                            elif "glassdoor" in url: source = "Glassdoor"
                            else: source = "Web"
                            all_jobs.append(CrawledJob(
                                title=role,
                                company="Unknown",
                                url=url,
                                source=source,
                                description=res.get("body", "")[:250]
                            ))
                    except Exception as e:
                        print(f"Error searching for {role} via DDG: {e}")
        except Exception as e:
            print(f"Failed to initialize DDGS: {e}")
        finally:
            crawl_done.set()

    crawl_thread = threading.Thread(target=do_crawl, daemon=True)
    crawl_thread.start()
    finished = crawl_done.wait(timeout=25)  # 25-second hard timeout

    if not finished:
        print("[WARN] DDGS crawl timed out after 25s — returning partial/fallback results")

    # Fallback if nothing was found
    if not all_jobs:
        for role in roles_to_search[:2]:
            all_jobs.append(CrawledJob(
                title=role,
                company="Search Unavailable",
                url=f"https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '+')}",
                source="LinkedIn",
                description="Direct link to LinkedIn job search for this role."
            ))
            all_jobs.append(CrawledJob(
                title=role,
                company="Search Unavailable",
                url=f"https://in.indeed.com/jobs?q={role.replace(' ', '+')}",
                source="Indeed",
                description="Direct link to Indeed job search for this role."
            ))

    return {"crawled_jobs": all_jobs[:12]}

class FormQuestionsList(BaseModel):
    questions: List[FormQuestion] = Field(description="The list of parsed form questions mapped with proposed answers from the candidate profile.")

def fill_form_node(state: AgentState) -> dict:
    """
    Generates a profile-based application form using the candidate's resume data.
    Flags novel questions for human approval.
    NOTE: This is a profile-based simulation. Real form-scraping via Playwright is a Phase 3 feature.
    """
    selected_job_url = state.get("selected_job_url")
    if not selected_job_url:
        return {"form_questions": [], "requires_human_approval": False}

    resume_data = state.get("resume_data", ResumeStateData())
    
    mock_form_text = f"""
    Form for {selected_job_url}:
    1. Full Name
    2. Email Address
    3. How many years of experience do you have?
    4. Provide a link to your GitHub profile.
    5. What is your expected salary?
    6. Why do you want to work at this company specifically?
    """

    llm = get_llm()
    structured_llm = llm.with_structured_output(FormQuestionsList)

    prompt = PromptTemplate.from_template(
        """You are an AI assistant helping a candidate fill out a job application form.
        
        Candidate Info:
        Candidate Info:
        Name: {name}
        Email: {email}
        Skills: {skills}
        Experience (Years): {experience_years}
        Education: {education}
        Additional Known Info (from previous applications):
        {additional_info}
        
        Form Text / Questions:
        {form_text}
        
        Task: 
        For each question found in the form text, generate a 'FormQuestion' object.
        - Set 'question_identifier' to a short code like 'q1' or the name of the field.
        - Set 'question_text' to the actual question.
        - Set 'proposed_answer' to the candidate's data if known. Check 'Additional Known Info' closely for things like Expected Salary or GitHub.
        - VERY IMPORTANT: If the question requires subjective input (like "Why do you want to work here?") or information NOT found in the candidate info or additional info, set 'is_unknown' to True and provide your best guess or leave blank in 'proposed_answer'.
        - If the answer is already provided in the Additional Known Info, or is easily extracted (like Name), set 'is_unknown' to False.
        """
    )
    
    chain = prompt | structured_llm
    
    try:
        print(f"[AGENT] Invoking LLM for form filling on job: {selected_job_url}...")
        start_time = time.time()
        result = chain.invoke({
            "name": resume_data.name,
            "email": resume_data.email,
            "skills": ", ".join(resume_data.skills),
            "experience_years": resume_data.experience_years,
            "education": ", ".join(resume_data.education),
            "additional_info": json.dumps(resume_data.additional_info),
            "form_text": mock_form_text
        })
        print(f"[AGENT] LLM response received in {time.time() - start_time:.2f}s")
        
        questions = result.questions
        
        # Fallback: if Gemini returned nothing, use sensible static defaults
        if not questions:
            raise ValueError("Empty questions from LLM")
            
        requires_approval = any(q.is_unknown for q in questions)
        return {
            "form_questions": questions,
            "requires_human_approval": requires_approval,
            "application_status": "waiting_approval" if requires_approval else "ready_to_submit"
        }
    except Exception as e:
        print(f"Error in fill_form_node (using fallback): {e}")
        # Safe fallback questions always present. Checks prior info first.
        def get_fallback(identifier, default=""):
            return resume_data.additional_info.get(identifier, default)
            
        def is_unknown(identifier):
            return identifier not in resume_data.additional_info

        fallback = [
            FormQuestion(question_identifier="name", question_text="Full Name", proposed_answer=resume_data.name, is_unknown=False),
            FormQuestion(question_identifier="email", question_text="Email Address", proposed_answer=resume_data.email, is_unknown=False),
            FormQuestion(question_identifier="exp", question_text="Years of experience", proposed_answer=str(resume_data.experience_years), is_unknown=False),
            FormQuestion(question_identifier="skills", question_text="Key skills", proposed_answer=", ".join(resume_data.skills[:5]), is_unknown=False),
            FormQuestion(question_identifier="salary", question_text="Expected salary / CTC", proposed_answer=get_fallback("salary"), is_unknown=is_unknown("salary")),
            FormQuestion(question_identifier="why", question_text="Why do you want to work here?", proposed_answer=get_fallback("why"), is_unknown=is_unknown("why")),
            FormQuestion(question_identifier="github", question_text="GitHub / Portfolio link", proposed_answer=get_fallback("github"), is_unknown=is_unknown("github")),
        ]
        
        requires_approval = any(q.is_unknown for q in fallback)
        
        return {
            "form_questions": fallback,
            "requires_human_approval": requires_approval,
            "application_status": "waiting_approval" if requires_approval else "ready_to_submit"
        }

def submit_application_node(state: AgentState) -> dict:
    """
    Simulates submitting the final application after all questions have been reviewed.
    """
    if state.get("application_status") == "waiting_approval" and not state.get("human_approved", False):
        return {"application_status": "failed_requires_approval"}
        
    return {"application_status": "submitted"}


