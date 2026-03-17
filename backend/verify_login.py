import asyncio
from automation.sites.linkedin import _login
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print("[TEST] Running login field detection test with NEW selectors...")
        # Credentials don't matter for field detection, they just need to be provided
        res = await _login(page, {"email": "test@example.com", "password": "password"})
        print(f"[TEST] Login result (expected to try entering but fail at feed check): {res}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test())
