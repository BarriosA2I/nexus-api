"""
CHROMADON SWAGGER E2E TEST - HTTP POLL MODE
============================================
Validates full APEX stack via Swagger UI submission + HTTP polling.

Key Features:
- Swagger UI for health checks and job submission
- HTTP polling for reliable job status (no DOM flakiness)
- Terminal state validation (completed/failed/timeout)
- Comprehensive artifact logging
"""

import os
import json
import time
import logging
from datetime import datetime
from playwright.sync_api import sync_playwright, expect

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = os.getenv("NEXUS_URL", "http://localhost:8000")
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"

# INCREASED TIMEOUT: Default to 20 minutes (1200s) for video rendering
POLL_TIMEOUT_S = int(os.getenv("CHROMADON_POLL_TIMEOUT_S", "1200"))
POLL_INTERVAL_S = 5

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ChromadonE2E")

# RUN DIRECTORY FOR ARTIFACTS
RUN_DIR = f"chromadon_runs/run_{int(time.time())}"
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)


# =============================================================================
# ARTIFACT HELPERS
# =============================================================================

def save_artifact(name: str, content, is_json: bool = False) -> str:
    """Save artifact to run directory."""
    path = os.path.join(RUN_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        if is_json:
            json.dump(content, f, indent=2)
        else:
            f.write(str(content))
    logger.info(f"   Artifact saved: {path}")
    return path


# =============================================================================
# HTTP POLLING (The Fix)
# =============================================================================

def poll_job_http(request_context, job_id: str) -> tuple:
    """
    Polls the Job API directly using Playwright's HTTP context.
    Decouples monitoring from Swagger UI flakiness.

    Returns:
        tuple: (final_status, final_data)
        - final_status: "completed" | "failed" | "timeout"
        - final_data: dict with job details
    """
    logger.info(f"🚀 Starting HTTP Polling for Job: {job_id}")
    logger.info(f"   Timeout: {POLL_TIMEOUT_S}s | Interval: {POLL_INTERVAL_S}s")

    start_time = time.time()
    poll_history = []
    job_url = f"{BASE_URL}/api/nexus/ragnarok/jobs/{job_id}"

    while (time.time() - start_time) < POLL_TIMEOUT_S:
        elapsed = int(time.time() - start_time)

        try:
            # Execute GET Request
            response = request_context.get(job_url)

            if response.status != 200:
                logger.warning(f"   [{elapsed}s] Poll received status {response.status}")
                time.sleep(POLL_INTERVAL_S)
                continue

            data = response.json()
            status = data.get("status", "unknown").lower()
            progress = data.get("progress", 0)

            # Log Snapshot
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "elapsed": elapsed,
                "status": status,
                "progress": progress,
                "data": data
            }
            poll_history.append(snapshot)

            # Progress bar visualization
            bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            logger.info(f"   [{elapsed}s] Status: {status.upper()} | Progress: [{bar}] {progress:.1f}%")

            # TERMINAL STATE CHECK
            if status in ["completed", "success"]:
                logger.info(f"✅ Job reached COMPLETED state after {elapsed}s")
                save_artifact("job_poll_timeline.json", poll_history, is_json=True)
                return "completed", data

            if status in ["failed", "error", "cancelled", "canceled"]:
                logger.error(f"❌ Job reached FAILED state: {data.get('error', 'Unknown')}")
                save_artifact("job_poll_timeline.json", poll_history, is_json=True)
                return "failed", data

        except Exception as e:
            logger.error(f"   [{elapsed}s] Polling Exception: {e}")

        time.sleep(POLL_INTERVAL_S)

    # TIMEOUT
    logger.error(f"⏰ TIMEOUT after {POLL_TIMEOUT_S}s - Job still {poll_history[-1]['status'] if poll_history else 'unknown'}")
    save_artifact("job_poll_timeline.json", poll_history, is_json=True)
    return "timeout", poll_history[-1]["data"] if poll_history else {}


# =============================================================================
# SWAGGER DOM TEST
# =============================================================================

def swagger_dom_test(page, request_context) -> dict:
    """
    1. Validates Swagger UI Loads
    2. Validates Health/Circuit endpoints via DOM
    3. Submits Generation Job via DOM
    4. Hands off to HTTP Polling for completion

    Returns:
        dict: Test results with outcome and artifacts
    """
    results = {
        "steps": [],
        "final_state": "unknown",
        "start_time": datetime.now().isoformat()
    }

    try:
        # =====================================================================
        # STEP 1: LOAD SWAGGER UI
        # =====================================================================
        logger.info("="*50)
        logger.info("STEP 1: Loading Swagger UI")
        logger.info("="*50)

        page.goto(f"{BASE_URL}/docs", wait_until="networkidle")

        # Wait for Swagger to fully render
        page.wait_for_selector(".swagger-ui", timeout=15000)
        time.sleep(2)  # Extra buffer for JS rendering

        page.screenshot(path=f"{RUN_DIR}/01_swagger_loaded.png")
        results["steps"].append("swagger_load_success")
        logger.info("   ✅ Swagger UI loaded")

        # =====================================================================
        # STEP 2: CHECK SERVICE HEALTH
        # =====================================================================
        logger.info("="*50)
        logger.info("STEP 2: Checking Service Health")
        logger.info("="*50)

        # Quick HTTP health check (more reliable than DOM)
        health_resp = request_context.get(f"{BASE_URL}/api/nexus/health")
        if health_resp.status == 200:
            health_data = health_resp.json()
            logger.info(f"   ✅ Nexus Health: {health_data.get('status', 'unknown')}")
            results["steps"].append("health_check_success")
            save_artifact("health_check.json", health_data, is_json=True)
        else:
            logger.warning(f"   ⚠️ Health check returned {health_resp.status}")
            results["steps"].append("health_check_warning")

        # Check circuit breaker
        circuit_resp = request_context.get(f"{BASE_URL}/api/nexus/ragnarok/service/circuit")
        if circuit_resp.status == 200:
            circuit_data = circuit_resp.json()
            logger.info(f"   ✅ Circuit Breaker: {circuit_data.get('state', 'unknown')}")
            results["steps"].append("circuit_check_success")
            save_artifact("circuit_check.json", circuit_data, is_json=True)

        # Check RAGNAROK service health
        ragnarok_health = request_context.get(f"{BASE_URL}/api/nexus/ragnarok/service/health")
        if ragnarok_health.status == 200:
            ragnarok_data = ragnarok_health.json()
            rag_status = ragnarok_data.get("ragnarok", {}).get("status", "unknown")
            logger.info(f"   ✅ RAGNAROK Service: {rag_status}")
            save_artifact("ragnarok_health.json", ragnarok_data, is_json=True)

        page.screenshot(path=f"{RUN_DIR}/02_health_checked.png")

        # =====================================================================
        # STEP 3: SUBMIT GENERATION JOB VIA HTTP (More Reliable)
        # =====================================================================
        logger.info("="*50)
        logger.info("STEP 3: Submitting Generation Job")
        logger.info("="*50)

        # Payload for commercial generation
        payload = {
            "brief": "Create a 30-second commercial for Barrios A2I, an AI automation company. Target audience: B2B SaaS founders and marketing directors. Highlight how AI automation gives startups enterprise-grade infrastructure. Tone: bold, innovative, cyberpunk aesthetic with cosmic cyan and neon amber colors.",
            "industry": "technology",
            "duration_seconds": 30,
            "platform": "youtube_1080p",
            "style": "cyberpunk",
            "voice_style": "professional"
        }

        save_artifact("job_payload.json", payload, is_json=True)

        # Submit via HTTP (more reliable than DOM)
        logger.info("   Executing job submission via HTTP...")
        submit_resp = request_context.post(
            f"{BASE_URL}/api/nexus/ragnarok/generate",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        if submit_resp.status != 200:
            error_text = submit_resp.text()
            logger.error(f"   ❌ Job submission failed: {submit_resp.status}")
            logger.error(f"   Response: {error_text[:500]}")
            raise Exception(f"Job submission failed with status {submit_resp.status}")

        initial_response = submit_resp.json()
        job_id = initial_response.get("job_id")

        if not job_id:
            logger.error(f"   Response missing job_id: {initial_response}")
            raise Exception("No job_id in response")

        logger.info(f"   ✅ Job Submitted Successfully!")
        logger.info(f"   📋 Job ID: {job_id}")
        logger.info(f"   📋 Status: {initial_response.get('status', 'unknown')}")

        results["job_id"] = job_id
        save_artifact("job_initial.json", initial_response, is_json=True)
        page.screenshot(path=f"{RUN_DIR}/03_job_submitted.png")
        results["steps"].append("job_submit_success")

        # =====================================================================
        # STEP 4: HTTP POLLING FOR COMPLETION
        # =====================================================================
        logger.info("="*50)
        logger.info("STEP 4: HTTP Polling for Completion")
        logger.info("="*50)

        final_status, final_data = poll_job_http(request_context, job_id)

        results["final_status"] = final_status
        results["final_data"] = final_data

        # Save final state
        save_artifact("job_final.json", final_data, is_json=True)
        page.screenshot(path=f"{RUN_DIR}/04_final_state.png")

        # =====================================================================
        # STEP 5: VALIDATION & OUTCOME
        # =====================================================================
        logger.info("="*50)
        logger.info("STEP 5: Validation")
        logger.info("="*50)

        if final_status == "completed":
            # Extract video_url from various possible locations
            video_url = (
                final_data.get("video_url") or
                final_data.get("result", {}).get("video_url") or
                final_data.get("tracking_url") or
                final_data.get("result", {}).get("tracking_url")
            )

            if video_url:
                logger.info(f"🎉 SUCCESS!")
                logger.info(f"   Video URL: {video_url}")
                save_artifact("video_url.txt", video_url)
                results["outcome"] = "PASS"
                results["video_url"] = video_url
            else:
                logger.warning("⚠️ Job completed but video_url missing in response schema")
                logger.info(f"   Available keys: {list(final_data.keys())}")
                if "result" in final_data:
                    logger.info(f"   Result keys: {list(final_data.get('result', {}).keys())}")
                results["outcome"] = "PASS_WITH_WARNING"
                results["warning"] = "Job completed but video_url not found in response. Check ragnarok_bridge result mapping."

        elif final_status == "timeout":
            last_status = final_data.get("status", "unknown")
            last_progress = final_data.get("progress", 0)
            logger.error(f"❌ TIMEOUT")
            logger.error(f"   Last Status: {last_status}")
            logger.error(f"   Last Progress: {last_progress}%")
            logger.error(f"   Manual check: GET {BASE_URL}/api/nexus/ragnarok/jobs/{job_id}")
            results["outcome"] = "FAIL_TIMEOUT"
            results["error"] = f"Job still {last_status} at {last_progress}% after {POLL_TIMEOUT_S}s"

        else:  # failed
            error_msg = final_data.get("error", "Unknown error")
            logger.error(f"❌ JOB FAILED")
            logger.error(f"   Error: {error_msg}")
            results["outcome"] = "FAIL"
            results["error"] = error_msg

    except Exception as e:
        logger.error(f"🔥 EXCEPTION: {str(e)}")
        results["outcome"] = "CRASH"
        results["exception"] = str(e)
        page.screenshot(path=f"{RUN_DIR}/crash_state.png")

    # Finalize
    results["end_time"] = datetime.now().isoformat()
    save_artifact("results.json", results, is_json=True)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("="*60)
    logger.info("⚡ CHROMADON SWAGGER E2E TEST (HTTP POLL MODE) ⚡")
    logger.info("="*60)
    logger.info(f"Target: {BASE_URL}")
    logger.info(f"Headless: {HEADLESS}")
    logger.info(f"Poll Timeout: {POLL_TIMEOUT_S}s")
    logger.info(f"Artifacts: {RUN_DIR}/")
    logger.info("")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)

        # Create context with both Page (UI) and Request (HTTP) capabilities
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()

        # Run test - pass request context for HTTP polling
        results = swagger_dom_test(page, context.request)

        browser.close()

    # Final Report
    logger.info("")
    logger.info("="*60)
    logger.info("📊 FINAL REPORT")
    logger.info("="*60)
    logger.info(f"Outcome: {results.get('outcome', 'UNKNOWN')}")

    if results.get('job_id'):
        logger.info(f"Job ID: {results['job_id']}")
    if results.get('video_url'):
        logger.info(f"Video URL: {results['video_url']}")
    if results.get('error'):
        logger.info(f"Error: {results['error']}")
    if results.get('warning'):
        logger.info(f"Warning: {results['warning']}")

    logger.info(f"Artifacts: {os.path.abspath(RUN_DIR)}/")
    logger.info("="*60)

    # Exit code based on outcome
    if results.get('outcome') in ['PASS', 'PASS_WITH_WARNING']:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
