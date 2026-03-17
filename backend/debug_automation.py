import asyncio
import traceback
from automation.browser_agent import auto_fill_job
import db

async def test():
    print("Fetching candidate...")
    candidate_id = "candidate_20260314_164333_976875"
    candidate_data = await db.get_candidate(candidate_id)
    if not candidate_data:
        print("Candidate not found!")
        return

    job_url = "https://www.linkedin.com/jobs/view/4338284054/"
    creds = {
        "email": "kshitijdalvi22@gmail.com",
        "password": "Dr@arninzola4"
    }

    try:
        print("Starting auto-fill...")
        result = await auto_fill_job(
            job_url=job_url,
            candidate_data=candidate_data,
            credentials=creds,
            headless=False # Headed for debugging visibility
        )
        print("\n--- Result ---")
        import json
        # Don't print the huge screenshot
        res_copy = result.copy()
        if "screenshot_b64" in res_copy: res_copy["screenshot_b64"] = "[BASE64 DATA]"
        print(json.dumps(res_copy, indent=2))
        
    except Exception:
        print("\n--- CRITICAL ERROR ---")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
