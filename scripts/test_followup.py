"""
Test follow-up message in Nexus chat
"""
from playwright.sync_api import sync_playwright
import time

def test_followup():
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

        print("Sending first message...")
        page.keyboard.type("Hi")
        page.keyboard.press("Enter")

        print("Waiting for response (15 seconds)...")
        time.sleep(15)

        # Check if input is enabled
        input_el = page.query_selector("#nexus-input")
        is_disabled = input_el.get_attribute("disabled") if input_el else "not found"
        print(f"Input disabled: {is_disabled}")

        # Check status text
        status_text = page.query_selector(".nexus-status-text")
        status = status_text.text_content() if status_text else "not found"
        print(f"Status: {status}")

        # Try sending follow-up
        print("Attempting follow-up message...")
        page.keyboard.type("What services do you offer?")
        page.keyboard.press("Enter")

        print("Waiting for follow-up response (15 seconds)...")
        time.sleep(15)

        print("\n=== Console Logs ===")
        for log in console_logs[-20:]:  # Last 20 logs
            print(log)

        print("\n=== Screenshot ===")
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/followup_test.png")
        print("Saved to followup_test.png")

if __name__ == "__main__":
    test_followup()
