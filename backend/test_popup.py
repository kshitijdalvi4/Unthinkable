import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print("Opened page. Navigating to google.com...")
        await page.goto("https://www.google.com")
        print("Success! If you see a browser window, tell me.")
        await asyncio.sleep(10)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
