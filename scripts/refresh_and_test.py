"""
Refresh page and test chat again
"""
from playwright.sync_api import sync_playwright
import time

def test_chat():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find barrios-landing page
        page = None
        for pg in context.pages:
            if "barrios-landing.vercel.app" in pg.url:
                page = pg
                break

        if not page:
            print("barrios-landing page not found!")
            return

        print("Refreshing page...")
        page.reload(timeout=30000)
        time.sleep(5)

        print("Opening chat panel...")
        launcher = page.query_selector(".nexus-launcher")
        if launcher:
            launcher.click()
            time.sleep(2)

        print("Typing message...")
        page.keyboard.type("What is Barrios A2I?")
        time.sleep(1)

        print("Pressing Enter...")
        page.keyboard.press("Enter")

        print("Waiting for response (15 seconds)...")
        time.sleep(15)

        # Take screenshot
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/chat_final_test.png")
        print("Screenshot saved: chat_final_test.png")

if __name__ == "__main__":
    test_chat()
