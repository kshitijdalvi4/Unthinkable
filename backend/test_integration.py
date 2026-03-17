import asyncio
import os
import sys
from automation.browser_agent import auto_fill_job

# Add current path
sys.path.append(os.getcwd())

async def test_integration():
    candidate_data = {
        "_id": "65f2a1b2c3d4e5f67890abcd",
        "name": "Kshitij Dalvi",
        "email": "kshitijdalvi22@gmail.com",
        "knowledge_base": {
            "name": "Kshitij Dalvi",
            "email": "kshitijdalvi22@gmail.com",
            "phone": "+91 9876543210",
            "experience_years": 2,
            "skills": ["Python", "React", "TypeScript", "Node.js"],
            "expected_salary": "15 LPA",
            "location": "Bangalore, India",
            "notice_period": "Immediate"
        },
        "additional_info": {
            "github": "https://github.com/kshitijdalvi",
            "linkedin": "https://linkedin.com/in/kshitijdalvi",
            "website": "https://kshitijdalvi.com",
            "why": "I am passionate about building AI-driven products."
        }
    }

    job_url = "https://www.linkedin.com/jobs/view/generative-ai-engineer-internship-in-bangalore-at-aiqwip-4373874596/"
    credentials = {
        "email": "kshitijdalvi22@gmail.com",
        "password": "Dr@arninzola4"
    }

    print("[INTEGRATION TEST] Starting auto_fill_job...")
    result = await auto_fill_job(
        job_url=job_url,
        candidate_data=candidate_data,
        credentials=credentials,
        headless=True # Use headless for CI/Agent verification
    )
    print("\nRESULT:", result)

if __name__ == "__main__":
    asyncio.run(test_integration())
