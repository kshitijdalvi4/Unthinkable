import asyncio
import sys
import os

from automation.browser_agent import auto_fill_job

async def main():
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
    
    try:
        # Run the generic one just to make sure Playwright launches at all
        res = await auto_fill_job(
            job_url="https://getbootstrap.com/docs/5.3/forms/overview/",
            candidate_data=candidate_data,
            headless=False
        )
        print("FINISHED GENERIC:", res)
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
