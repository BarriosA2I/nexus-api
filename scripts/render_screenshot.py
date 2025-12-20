"""
Take screenshot of current Render page
"""
from playwright.sync_api import sync_playwright

def screenshot():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0]

        print(f"Current URL: {page.url}")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/render_current.png", full_page=True)
        print("Screenshot saved: render_current.png")

        # Get page title
        print(f"Page title: {page.title()}")

if __name__ == "__main__":
    screenshot()
