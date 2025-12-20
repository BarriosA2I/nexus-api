"""
Check Vercel dashboard to find barrios-landing source repo
"""
from playwright.sync_api import sync_playwright
import time

def check_vercel():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        print("Navigating to Vercel dashboard...")
        page.goto("https://vercel.com/dashboard", timeout=30000)
        time.sleep(3)

        print(f"Current URL: {page.url}")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/vercel_dashboard.png", full_page=True)
        print("Screenshot saved: vercel_dashboard.png")

if __name__ == "__main__":
    check_vercel()
