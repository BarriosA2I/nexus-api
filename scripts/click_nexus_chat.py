"""
Click the Nexus chat button and test
"""
from playwright.sync_api import sync_playwright
import time

def click_chat():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find the barrios-landing page
        page = None
        for pg in context.pages:
            if "barrios-landing" in pg.url.lower():
                page = pg
                break

        if not page:
            print("barrios-landing page not found!")
            return

        print(f"Current URL: {page.url}")

        # Click on Nexus chat button (bottom right FAB)
        print("Looking for Nexus chat button...")
        try:
            # Try different selectors
            selectors = [
                "#nexus-fab",
                ".nexus-launcher",
                "[data-nexus-launcher]",
                "button[aria-label*='Nexus']",
                "button[aria-label*='chat']",
                ".fixed.bottom-4.right-4",
            ]

            for selector in selectors:
                btn = page.query_selector(selector)
                if btn:
                    print(f"Found button with selector: {selector}")
                    btn.click()
                    time.sleep(2)
                    break

            # Take screenshot after click
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/nexus_chat_open.png", full_page=True)
            print("Screenshot saved: nexus_chat_open.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/nexus_chat_error.png")

if __name__ == "__main__":
    click_chat()
