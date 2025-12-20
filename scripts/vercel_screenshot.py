"""
Screenshot Vercel dashboard
"""
from playwright.sync_api import sync_playwright

def screenshot():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find Vercel page
        page = None
        for pg in context.pages:
            if "vercel" in pg.url.lower():
                page = pg
                break

        if page:
            print(f"URL: {page.url}")
            page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/vercel_projects.png", full_page=True)
            print("Screenshot saved: vercel_projects.png")

if __name__ == "__main__":
    screenshot()
