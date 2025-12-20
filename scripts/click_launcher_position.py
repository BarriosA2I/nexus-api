"""
Click on the Nexus launcher by position (bottom right)
"""
from playwright.sync_api import sync_playwright
import time

def click_launcher():
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
            page = context.new_page()
            page.goto("https://barrios-landing.vercel.app", timeout=30000)
            time.sleep(3)

        print(f"URL: {page.url}")

        # Get viewport size
        viewport = page.viewport_size
        print(f"Viewport: {viewport}")

        # Look for the launcher element
        launcher = page.query_selector(".nexus-launcher, #nexus-launcher")
        if launcher:
            print("Found launcher element!")
            box = launcher.bounding_box()
            if box:
                print(f"Launcher box: {box}")
                launcher.click()
                time.sleep(2)
                print("Clicked launcher!")
            else:
                print("Launcher has no bounding box (might be hidden)")
        else:
            print("Launcher element not found")
            # List all elements with nexus in class
            elements = page.query_selector_all("[class*='nexus']")
            print(f"Found {len(elements)} elements with 'nexus' in class")

        # Take screenshot
        page.screenshot(path="C:/Users/gary/nexus_assistant_unified/scripts/launcher_test.png")
        print("Screenshot saved: launcher_test.png")

if __name__ == "__main__":
    click_launcher()
