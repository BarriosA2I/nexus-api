"""
Click on barrios-landing project in Vercel
"""
from playwright.sync_api import sync_playwright
import time

def click_project():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find Vercel page
        page = None
        for pg in context.pages:
            if "vercel" in pg.url.lower():
                page = pg
                break

        if page:
            print(f"Current URL: {page.url}")

            # Click on barrios-landing project
            print("Clicking on barrios-landing...")
            try:
                page.click("a:has-text('barrios-landing')", timeout=5000)
                time.sleep(3)
                print(f"New URL: {page.url}")
                page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/vercel_barrios_landing.png", full_page=True)
                print("Screenshot saved")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    click_project()
