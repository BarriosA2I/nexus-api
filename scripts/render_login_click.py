"""
Click Sign In button on Render/GitHub login
"""
from playwright.sync_api import sync_playwright
import time

def click_signin():
    print("\n[1] Connecting to Chrome...")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        print(f"    Current URL: {page.url}")

        # Click sign in button
        print("\n[2] Clicking Sign In button...")
        try:
            page.click("input[type='submit'][value='Sign in'], button:has-text('Sign in')", timeout=5000)
            print("    [OK] Clicked Sign In")

            # Wait for navigation
            time.sleep(5)
            print(f"    New URL: {page.url}")

            # Take screenshot
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/render_after_login.png", full_page=True)
            print("    [OK] Screenshot saved: render_after_login.png")

        except Exception as e:
            print(f"    [XX] Error: {e}")
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/render_error.png", full_page=True)

if __name__ == "__main__":
    click_signin()
