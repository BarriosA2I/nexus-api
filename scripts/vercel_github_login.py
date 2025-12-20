"""
Login to Vercel with GitHub
"""
from playwright.sync_api import sync_playwright
import time

def login_vercel():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find the Vercel login page
        page = None
        for pg in context.pages:
            if "vercel" in pg.url.lower():
                page = pg
                break

        if not page:
            page = context.pages[-1]  # Use the last page

        print(f"Current URL: {page.url}")

        # Click Continue with GitHub
        print("Clicking 'Continue with GitHub'...")
        try:
            page.click("button:has-text('Continue with GitHub')", timeout=5000)
            time.sleep(5)
            print(f"New URL: {page.url}")
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/vercel_after_github.png", full_page=True)
            print("Screenshot saved")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    login_vercel()
