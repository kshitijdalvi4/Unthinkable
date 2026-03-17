import asyncio
import os
import sys
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

# Add backend to path for imports
sys.path.append(os.getcwd())

from automation.sites.linkedin import handle_linkedin

async def test_linkedin_form():
    # Mock candidate data
    candidate_data = {
        "_id": "test_candidate_123",
        "name": "Kshitij Dalvi",
        "email": "kshitijdalvi22@gmail.com",
        "knowledge_base": {
            "name": "Kshitij Dalvi",
            "email": "kshitijdalvi22@gmail.com",
            "phone": "+91 9876543210",
            "experience_years": 2,
            "skills": "Python, React, TypeScript, Node.js",
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

    # Use the user's provided LinkedIn job
    job_url = "https://www.linkedin.com/jobs/view/generative-ai-engineer-internship-in-bangalore-at-aiqwip-4373874596/"
    
    # Credentials for login
    credentials = {
        "email": "kshitijdalvi22@gmail.com",
        "password": "Dr@arninzola4"
    }

    async with async_playwright() as p:
        print("[TEST] Launching browser...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"[TEST] Testing LinkedIn Job: {job_url}")
        result = await handle_linkedin(page, job_url, candidate_data, credentials)

        print("\n" + "="*50)
        print("TEST RESULT:")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Filled Fields: {result.get('filled_fields')}")
        print("="*50 + "\n")

        print("[TEST] Browser will stay open for 60 seconds for inspection...")
        await asyncio.sleep(60)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_linkedin_form())
