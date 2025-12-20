"""
Navigate to barrios-landing and test the Nexus chat
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
        time.sleep(3)

        print(f"Current URL: {page.url}")

        # Look for and click the Nexus FAB button
        print("\nLooking for Nexus chat button...")

        # List all buttons on page
        buttons = page.query_selector_all("button")
        print(f"Found {len(buttons)} buttons")

        # Look for any fixed positioned element that might be the FAB
        fab = page.query_selector("#nexus-fab")
        if fab:
            print("Found #nexus-fab!")
            fab.click()
            time.sleep(2)
        else:
            print("nexus-fab not found, trying alternatives...")
            # Check page HTML for nexus elements
            html = page.content()
            if "nexus" in html.lower():
                print("Nexus elements exist in HTML")
            else:
                print("No nexus elements found in HTML!")

        # Take screenshot
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/nexus_test_result.png", full_page=True)
        print("\nScreenshot saved: nexus_test_result.png")

        # Check console logs
        print("\nGetting console logs...")

if __name__ == "__main__":
    test_chat()
