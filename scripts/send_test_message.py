"""
Send a test message in the Nexus chat
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

        # Find the input field
        input_field = page.query_selector("input[placeholder*='Ask Nexus'], textarea[placeholder*='Ask Nexus'], .nexus-panel input, .nexus-panel textarea")
        if input_field:
            print("Found input field!")
            input_field.fill("Hello, what services does Barrios A2I offer?")
            time.sleep(1)

            # Find and click the send button
            send_btn = page.query_selector("button[type='submit'], .nexus-panel button[aria-label*='send'], .nexus-panel-input-container button")
            if send_btn:
                print("Found send button, clicking...")
                send_btn.click()
                time.sleep(5)  # Wait for response
                print("Message sent! Waiting for response...")
            else:
                print("Send button not found, trying Enter key...")
                input_field.press("Enter")
                time.sleep(5)
        else:
            print("Input field not found!")

        # Take screenshot of response
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/chat_response.png")
        print("Screenshot saved: chat_response.png")

if __name__ == "__main__":
    send_message()
