"""
Check Render deployment status
"""
from playwright.sync_api import sync_playwright
import time

def check_deploy():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Find or navigate to Render
        page = None
        for pg in context.pages:
            if "render" in pg.url.lower() and "nexus-api" in pg.url.lower():
                page = pg
                break

        if not page:
            page = context.new_page()
            page.goto("https://dashboard.render.com/web/srv-d52t6hjuibrs73a66frg", timeout=30000)
            time.sleep(3)

        print(f"URL: {page.url}")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/render_deploy_status.png", full_page=True)
        print("Screenshot saved")

if __name__ == "__main__":
    check_deploy()
