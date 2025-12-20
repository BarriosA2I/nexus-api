"""
Click on nexus-api service to see details
"""
from playwright.sync_api import sync_playwright
import time

def click_nexus():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0]

        print(f"Current URL: {page.url}")

        # Click on nexus-api link
        print("Clicking on nexus-api...")
        try:
            page.click("a:has-text('nexus-api')", timeout=5000)
            time.sleep(3)
            print(f"New URL: {page.url}")

            # Take screenshot
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/render_nexus_detail.png", full_page=True)
            print("Screenshot saved: render_nexus_detail.png")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    click_nexus()
