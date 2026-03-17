"""
Indeed-specific form filler.
"""
from playwright.async_api import Page
from typing import Optional
import asyncio


INDEED_LOGIN_URL = "https://secure.indeed.com/auth"


async def _login(page: Page, credentials: dict) -> bool:
    try:
        await page.goto(INDEED_LOGIN_URL, wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # Some Indeed flows start with just email, then move to password. Sometimes it's a captcha.
        if await page.locator("input[type='email']").is_visible(timeout=3000):
            await page.fill("input[type='email']", credentials.get("email", ""))
            try:
                await page.click("button:has-text('Continue'), button[type='submit']", timeout=2000)
                await asyncio.sleep(2)
            except Exception:
                pass
        
        # Wait to see if password field appears or if it hit a challenge
        if await page.locator("input[type='password']").is_visible(timeout=5000):
            await page.fill("input[type='password']", credentials.get("password", ""))
            await page.click("button[type='submit']")
            await asyncio.sleep(3)
        else:
            print("[Indeed] Password field not found. Might be a bot check or magic link.")

        # Determine if we are still on an auth or trap page
        current_url = page.url.lower()
        if "auth" in current_url or "challenge" in current_url or "captcha" in current_url:
            print("[Indeed] Security challenge or 2FA detected!")
            print("[Indeed] >> WAITING 60 SECONDS FOR YOU TO SOLVE IT MANUALLY IN THE BROWSER <<")
            try:
                # Wait for the URL to change away from auth/captcha domains
                async def check_auth_cleared():
                    while "auth" in page.url.lower() or "challenge" in page.url.lower():
                        await asyncio.sleep(1)
                    return True
                
                await asyncio.wait_for(check_auth_cleared(), timeout=60.0)
                print("[Indeed] Manual login resolution successful")
                return True
            except Exception:
                print("[Indeed] Manual login timed out")
                return False

        print("[Indeed] Login attempted successfully")
        return True

    except Exception as e:
        print(f"[Indeed] Login failed: {e}")
        return False


async def handle_indeed(
    page: Page,
    job_url: str,
    candidate_data: dict,
    credentials: Optional[dict],
) -> dict:
    filled_fields = []

    if credentials:
        await _login(page, credentials)

    print(f"[Indeed] Navigating to {job_url}")
    await page.goto(job_url, wait_until="domcontentloaded")
    await asyncio.sleep(2)

    # Try Apply Now button
    try:
        apply_btn = page.locator("button:has-text('Apply now'), a:has-text('Apply now')").first
        await apply_btn.click(timeout=5000)
        await asyncio.sleep(2)
        print("[Indeed] Apply Now clicked")
    except Exception:
        pass

    # Fill form fields
    field_map = {
        "name": candidate_data.get("name", ""),
        "email": candidate_data.get("email", ""),
        "salary": candidate_data.get("additional_info", {}).get("salary", ""),
    }

    for label, value in field_map.items():
        if not value:
            continue
        try:
            inp = page.locator(
                f"input[name*='{label}' i], input[placeholder*='{label}' i], "
                f"input[id*='{label}' i]"
            ).first
            if await inp.is_visible(timeout=1500):
                await inp.fill(value)
                filled_fields.append(label)
        except Exception:
            pass

    return {
        "status": "partial" if filled_fields else "needs_manual",
        "message": f"Filled {len(filled_fields)} field(s) on Indeed. Please review and submit.",
        "filled_fields": filled_fields,
    }
