import asyncio
import os
import sys
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load API Keys and environment
load_dotenv()

# Add parent directory to path to allow imports from automation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automation.sites.linkedin import handle_linkedin

async def test_linkedin_button():
    # 1. Target URL from User
    job_url = "https://www.linkedin.com/jobs/view/generative-ai-engineer-internship-in-bangalore-at-aiqwip-4373874596/"
    
    print(f"\n[TEST] Target URL: {job_url}")
    print(f"[TEST] Using model: {os.getenv('LLM_MODEL', 'gemini-2.5-flash')}")

    candidate_data = {
        "name": "Kshitij Dalvi",
        "email": "kshitijdalvi22@gmail.com",
        "knowledge_base": {
            "name": "Kshitij Dalvi",
            "email": "kshitijdalvi22@gmail.com",
            "phone": "9029895438",
            "location": "Mumbai, Maharashtra, India",
            "experience_years": "2",
            "expected_salary": "1200000"
        },
        "additional_info": {
            "github": "https://github.com/kshitijdalvi",
            "linkedin": "https://linkedin.com/in/kshitijdalvi",
            "why": "I am passionate about Generative AI."
        }
    }

    async with async_playwright() as pw:
        print("[TEST] Launching browser...")
        browser = await pw.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            print("[TEST] Navigating to LinkedIn job page...")
            # Note: LinkedIn often requires login to see the Apply button reliably.
            # If nothing happens, you might need to login manually in the opened window first.
            await page.goto(job_url, wait_until="domcontentloaded")
            await asyncio.sleep(5)
            
            print("[TEST] Running robust button detection and click logic...")
            result = await handle_linkedin(page, job_url, candidate_data, credentials=None)
            
            print("\n" + "="*50)
            print(f"RESULT STATUS: {result.get('status')}")
            print(f"RESULT MESSAGE: {result.get('message')}")
            print(f"FILLED FIELDS: {result.get('filled_fields')}")
            print("="*50 + "\n")

            if result.get("status") == "needs_manual":
                print("[!] The agent couldn't find the button automatically.")
                print("[!] Check if you are logged in or if the page layout has changed.")
            else:
                print("[SUCCESS] The agent attempted the click sequence.")
            
            print("\n[TEST] Browser will stay open for 60 seconds for inspection...")
            await asyncio.sleep(60)

        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_linkedin_button())
