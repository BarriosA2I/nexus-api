"""
Send a test message using Enter key
"""
from playwright.sync_api import sync_playwright
import time

def send_message():
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

        print(f"URL: {page.url}")

        # Type in the input field
        print("Typing message...")
        page.keyboard.type("Hello, what services does Barrios A2I offer?")
        time.sleep(1)

        print("Pressing Enter...")
        page.keyboard.press("Enter")

        print("Waiting for response...")
        time.sleep(8)

        # Take screenshot of response
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/chat_response.png")
        print("Screenshot saved: chat_response.png")

if __name__ == "__main__":
    send_message()
