"""
Check Render Dashboard Logs via Playwright
"""
from playwright.sync_api import sync_playwright
import time

def check_render_logs():
    print("\n" + "="*70)
    print("  Checking Render Dashboard Logs")
    print("="*70)

    with sync_playwright() as p:
        try:
            # Connect to Chrome debug port
            print("\n[1] Connecting to Chrome on port 9222...")
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            print("    [OK] Connected to Chrome")

            # Get default context and page
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()
            print(f"    Current URL: {page.url}")

        except Exception as e:
            print(f"    [XX] Failed to connect: {e}")
            return

        # Navigate to Render dashboard for nexus-api
        print("\n[2] Navigating to Render nexus-api service...")
        try:
            page.goto("https://dashboard.render.com", timeout=30000)
            time.sleep(3)
            print(f"    Current URL: {page.url}")

            # Take screenshot
            screenshot_path = "C:/Users/gary/nexus_assistant_unified/scripts/render_dashboard.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"    [OK] Screenshot saved: {screenshot_path}")

            # Get page content
            content = page.content()
            print(f"    Page content length: {len(content)} chars")

            # Look for error indicators
            if "nexus-api" in content.lower():
                print("    [OK] Found nexus-api on page")

            # Check if logged in
            if "sign in" in content.lower() or "log in" in content.lower():
                print("    [!!] Not logged in - please log in to Render in the browser")

        except Exception as e:
            print(f"    [XX] Navigation error: {e}")

        print("\n[3] Done - check the screenshot")
        print("    Screenshot: C:/Users/gary/nexus_assistant_unified/scripts/render_dashboard.png")

if __name__ == "__main__":
    check_render_logs()
