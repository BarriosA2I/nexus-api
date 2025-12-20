"""
Quick localhost test using Playwright
Tests Nexus and RAGNAROK endpoints
"""
import asyncio
from playwright.async_api import async_playwright

ENDPOINTS = [
    ("Nexus Root", "http://localhost:8000/"),
    ("Nexus Health", "http://localhost:8000/api/nexus/health"),
    ("Nexus Docs", "http://localhost:8000/docs"),
    ("RAGNAROK Health", "http://localhost:8001/health"),
    ("RAGNAROK Docs", "http://localhost:8001/docs"),
]


async def test_endpoints():
    print("\n" + "="*60)
    print("  APEX Stack - Localhost Test")
    print("="*60 + "\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        results = []

        for name, url in ENDPOINTS:
            try:
                response = await page.goto(url, timeout=10000)
                status = response.status if response else "No response"

                if response and response.status == 200:
                    content = await page.content()
                    size = len(content)
                    print(f"  [OK] {name}")
                    print(f"       {url}")
                    print(f"       Status: {status}, Size: {size} bytes\n")
                    results.append((name, True, status))
                else:
                    print(f"  [XX] {name}")
                    print(f"       {url}")
                    print(f"       Status: {status}\n")
                    results.append((name, False, status))

            except Exception as e:
                print(f"  [XX] {name}")
                print(f"       {url}")
                print(f"       Error: {str(e)[:50]}\n")
                results.append((name, False, str(e)[:50]))

        await browser.close()

        # Summary
        passed = sum(1 for _, ok, _ in results if ok)
        total = len(results)

        print("="*60)
        print(f"  Results: {passed}/{total} endpoints passed")
        print("="*60 + "\n")

        return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_endpoints())
    exit(0 if success else 1)
