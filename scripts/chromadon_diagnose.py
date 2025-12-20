"""
Chromadon Browser Diagnostic - Test localhost URLs
Connects to Chrome debug port and tests APEX stack endpoints
"""
from playwright.sync_api import sync_playwright
import json

URLS_TO_TEST = [
    ("Nexus Root", "http://localhost:8000/"),
    ("Nexus Health", "http://localhost:8000/api/nexus/health"),
    ("Nexus Swagger", "http://localhost:8000/docs"),
    ("RAGNAROK Health", "http://localhost:8001/health"),
    ("RAGNAROK Swagger", "http://localhost:8001/docs"),
]


def diagnose():
    print("\n" + "="*70)
    print("  CHROMADON Browser Diagnostic - APEX Stack")
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
            print(f"    [OK] Current page: {page.url}")

        except Exception as e:
            print(f"    [XX] Failed to connect: {e}")
            print("\n    Make sure Chrome is running with --remote-debugging-port=9222")
            return False

        print("\n[2] Testing localhost endpoints...\n")

        results = []
        for name, url in URLS_TO_TEST:
            print(f"    Testing: {name}")
            print(f"    URL: {url}")

            try:
                # Navigate to URL
                response = page.goto(url, timeout=15000, wait_until="networkidle")

                if response:
                    status = response.status

                    # Get page content
                    content = page.content()
                    body_text = page.inner_text("body") if page.query_selector("body") else ""

                    # Check for errors
                    if status == 200:
                        # Try to parse JSON for API endpoints
                        if "/health" in url or url.endswith("/"):
                            try:
                                json_data = json.loads(body_text.strip())
                                print(f"    Status: {status} OK")
                                print(f"    Response: {json.dumps(json_data, indent=2)[:200]}...")
                                results.append((name, True, "OK", json_data))
                            except:
                                print(f"    Status: {status} OK")
                                print(f"    Content length: {len(content)} bytes")
                                results.append((name, True, "OK", None))
                        else:
                            print(f"    Status: {status} OK")
                            print(f"    Content length: {len(content)} bytes")
                            results.append((name, True, "OK", None))
                    else:
                        print(f"    Status: {status} ERROR")
                        print(f"    Body: {body_text[:200]}")
                        results.append((name, False, f"HTTP {status}", body_text[:200]))
                else:
                    print(f"    Status: No response")
                    results.append((name, False, "No response", None))

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"    [XX] Error: {error_msg}")
                results.append((name, False, "Error", error_msg))

            print()

        # Take screenshot of last page
        try:
            screenshot_path = "C:/Users/gary/nexus_assistant_unified/scripts/localhost_screenshot.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"[3] Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"[3] Screenshot failed: {e}")

        # Summary
        print("\n" + "="*70)
        print("  DIAGNOSTIC SUMMARY")
        print("="*70)

        passed = sum(1 for _, ok, _, _ in results if ok)
        total = len(results)

        for name, ok, status, data in results:
            icon = "[OK]" if ok else "[XX]"
            print(f"  {icon} {name}: {status}")

        print(f"\n  Total: {passed}/{total} passed")
        print("="*70 + "\n")

        return passed == total


if __name__ == "__main__":
    success = diagnose()
    exit(0 if success else 1)
