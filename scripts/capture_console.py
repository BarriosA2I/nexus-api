"""
Capture browser console logs during chat
"""
from playwright.sync_api import sync_playwright
import time

def capture_console():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        # Create new page with console listener
        page = context.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_logs.append(f"[PAGE_ERROR] {err}"))

        print("Navigating to barrios-landing...")
        page.goto("https://barrios-landing.vercel.app", timeout=30000)
        time.sleep(3)

        print("Opening chat panel...")
        launcher = page.query_selector(".nexus-launcher")
        if launcher:
            launcher.click()
            time.sleep(2)

        print("Typing and sending message...")
        page.keyboard.type("Hi")
        page.keyboard.press("Enter")

        print("Waiting for response (20 seconds)...")
        time.sleep(20)

        print("\n=== Console Logs ===")
        for log in console_logs:
            print(log)

        print("\n=== Screenshot ===")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/console_test.png")

if __name__ == "__main__":
    capture_console()
