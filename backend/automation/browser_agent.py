"""
Phase 3 Browser Automation Agent
Detects job site, dispatches to platform-specific handler, fills forms.
"""
import asyncio
from typing import Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# Site detection
SITE_PATTERNS = {
    "linkedin":   ["linkedin.com"],
    "indeed":     ["indeed.com", "indeed.co.in"],
    "jooble":     ["jooble.org"],
    "glassdoor":  ["glassdoor.com"],
    "naukri":     ["naukri.com"],
}

def detect_site(url: str) -> str:
    url_lower = url.lower()
    for site, patterns in SITE_PATTERNS.items():
        if any(p in url_lower for p in patterns):
            return site
    return "generic"


async def auto_fill_job(
    job_url: str,
    candidate_data: dict,
    credentials: Optional[dict] = None,
    headless: bool = False,
) -> dict:
    """
    Entry point for Phase 3 browser automation.
    Returns {status, message, filled_fields, screenshot_b64}
    """
    site = detect_site(job_url)
    print(f"[P3] Auto-fill triggered for site={site} url={job_url}")

    async with async_playwright() as pw:
        launch_args = ["--start-maximized"] if not headless else []
        print(f"[P3] Launching browser (headless={headless})...")
        browser: Browser = await pw.chromium.launch(
            headless=headless,
            slow_mo=500 if not headless else 0,
            args=launch_args
        )
        context: BrowserContext = await browser.new_context(
            viewport=None if not headless else {"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page: Page = await context.new_page()

        try:
            # Dispatch to the right handler
            if site == "linkedin":
                from automation.sites.linkedin import handle_linkedin
                result = await handle_linkedin(page, job_url, candidate_data, credentials)
            elif site == "indeed":
                from automation.sites.indeed import handle_indeed
                result = await handle_indeed(page, job_url, candidate_data, credentials)
            else:
                from automation.sites.generic import handle_generic
                result = await handle_generic(page, job_url, candidate_data)

            # Capture screenshot for audit trail
            screenshot = await page.screenshot(full_page=False)
            import base64
            result["screenshot_b64"] = base64.b64encode(screenshot).decode()
            return result

        except Exception as e:
            print(f"[P3] Error during automation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "filled_fields": [],
                "screenshot_b64": None,
            }
        finally:
            if not headless:
                print("[P3] Browser will remain open for 5 minutes for manual review/submission.")
                await asyncio.sleep(300) 
            await browser.close()
