import asyncio
import os
import sys

# Ensure backend imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from automation.browser_agent import auto_fill_job

async def test_generic_submit():
    candidate_data = {
        "name": "Kshitij Dalvi",
        "email": "kshitijdalvi22@gmail.com",
        "experience_years": 3,
        "skills": ["Python", "FastAPI", "React"],
        "additional_info": {
            "phone": "9876543210",
            "github": "https://github.com/kshitijdalvi",
            "salary": "120000",
            "why": "I like this job."
        }
    }

    print("Testing generic handler on an example form (Wait for browser popup)...")
    res = await auto_fill_job(
        job_url="https://getbootstrap.com/docs/5.3/forms/overview/", 
        candidate_data=candidate_data, 
        headless=True
    )
    print("Result:")
    print(res)

if __name__ == "__main__":
    asyncio.run(test_generic_submit())
