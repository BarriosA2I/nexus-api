"""
Test the live chat on barrios-landing.vercel.app
"""
from playwright.sync_api import sync_playwright
import time

def test_chat():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        print("Navigating to barrios-landing.vercel.app...")
        page.goto("https://barrios-landing.vercel.app", timeout=30000)
        time.sleep(5)

        print(f"Current URL: {page.url}")

        # Take screenshot
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/barrios_landing_live.png", full_page=True)
        print("Screenshot saved: barrios_landing_live.png")

        # Check for Nexus chat button
        print("\nLooking for Nexus chat elements...")
        try:
            # Look for the chat launcher button
            nexus_button = page.query_selector("[data-nexus-launcher], .nexus-launcher, #nexus-fab, button[aria-label*='chat']")
            if nexus_button:
                print("Found Nexus launcher button!")
            else:
                print("Nexus launcher not found in DOM")

            # Check console for API base URL
            print("\nDone - check the screenshot")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_chat()
