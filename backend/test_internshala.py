import asyncio
import os
import sys
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

# Add backend to path for imports
sys.path.append(os.getcwd())

from automation.sites.generic import handle_generic

async def test_internshala():
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
            "why": "I am passionate about building AI-driven products and helping companies scale with Generative AI."
        }
    }

    # URL retrieved by subagent
    job_url = "https://internshala.com/internship/detail/generative-ai-engineer-internship-in-bangalore-at-aiqwip1771236839/"
    
    async with async_playwright() as p:
        print("[TEST] Launching browser...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"[TEST] Testing Internshala Job: {job_url}")
        # Note: Internshala might require login for the actual "Apply" step, 
        # but we want to see if the generic filler detects the fields on the apply page.
        # We navigate to the job page and click "Apply now" if present.
        
        await page.goto(job_url, wait_until="networkidle")
        
        # Internshala "Apply now" button
        apply_btn = page.locator("#apply_now, button:has-text('Apply now')").first
        if await apply_btn.count() > 0:
            print("[TEST] Clicking 'Apply now' to reach the form...")
            await apply_btn.click()
            await asyncio.sleep(2)
        
        # If redirected to login, this test won't fill, but we can verify field detection
        # on a mock page or just use handle_generic directly
        result = await handle_generic(page, page.url, candidate_data)

        print("\n" + "="*50)
        print("TEST RESULT:")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Filled Fields: {result.get('filled_fields')}")
        print("="*50 + "\n")

        print("[TEST] Browser will stay open for 30 seconds for inspection...")
        await asyncio.sleep(30)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_internshala())
