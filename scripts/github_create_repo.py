"""
Create barrios-landing GitHub repo via browser
"""
from playwright.sync_api import sync_playwright
import time

def create_repo():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        print("Navigating to GitHub new repo page...")
        page.goto("https://github.com/new", timeout=30000)
        time.sleep(3)

        print(f"Current URL: {page.url}")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/github_new_repo.png", full_page=True)
        print("Screenshot saved: github_new_repo.png")

if __name__ == "__main__":
    create_repo()
