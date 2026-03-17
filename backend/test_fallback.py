import os
import json
from dotenv import load_dotenv
load_dotenv()

from agents.state import ResumeStateData, AgentState
from agents.agents import fill_form_node

# Mock resume data with answers
resume_state = ResumeStateData(
    name="Test User",
    email="test@example.com",
    skills=["Python", "FastAPI"],
    experience_years=3.0,
    education=["BSc CS"],
    raw_text="Test resume content",
    additional_info={
        "salary": "120,000",
        "why": "I like the tech stack.",
        "github": "https://github.com/kshitijdalvi4"
    }
)

# Mock agent state
state = AgentState(
    candidate_id="test_candidate",
    resume_data=resume_state,
    selected_job_url="https://example.com/job"
)

# Run fill form node
result = fill_form_node(state)

print(json.dumps([q.model_dump() for q in result.get("form_questions", [])], indent=2))
print("Requires approval:", result.get("requires_human_approval"))
print("Application status:", result.get("application_status"))
