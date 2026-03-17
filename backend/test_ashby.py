import asyncio
import os
import sys
from playwright.async_api import async_playwright

# Add backend to path for imports
sys.path.append(os.getcwd())

from automation.sites.generic import handle_generic

async def test_ashby():
    # Mock candidate data from MongoDB format
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

    job_url = "https://jobs.ashbyhq.com/orb/20199cf9-8f79-4e26-bc9d-dd98b3008d59"

    async with async_playwright() as p:
        print("[TEST] Launching browser...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"[TEST] Testing Ashby Job: {job_url}")
        result = await handle_generic(page, job_url, candidate_data)

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
    asyncio.run(test_ashby())
