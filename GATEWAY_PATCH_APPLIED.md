# Nexus Gateway Proxy Patch Applied

**Date:** December 19, 2025

## Changes Made

### 1. Error Propagation Fixed (`ragnarok_service.py`)

Added `RagnarokAPIError` exception class that preserves HTTP status codes from RAGNAROK:

```python
class RagnarokAPIError(HTTPException):
    """Preserves RAGNAROK's HTTP status code (429, 400, etc.)"""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
```

Now when RAGNAROK returns:
- 429 Rate Limit → Nexus returns 429
- 400 Bad Request → Nexus returns 400
- 503 Service Unavailable → Nexus returns 503

### 2. Standardized Response Shape

All generation endpoints now return BOTH IDs:

```json
{
    "job_id": "job_abc123...",
    "workflow_id": "565c03d4-74b3-4752-a228-1bf94070a52b",
    "status": "queued",
    "tracking_url": "/api/nexus/ragnarok/jobs/job_abc123...",
    "trace_id": "nxs_1234567890_abc123"
}
```

### 3. Unified Polling Logic

**SINGLE POLLING ENDPOINT:**
```
GET /api/nexus/ragnarok/jobs/{job_id}
```

This endpoint:
- Returns Nexus job status
- Fetches RAGNAROK workflow status via background polling
- Syncs status between Nexus and RAGNAROK
- Returns video_url, cost, quality_score when complete

**Frontend NEVER needs to:**
- Know about port 8001
- Poll RAGNAROK directly
- Parse workflow_id vs job_id differently

## Verification Commands

### After Restart, Verify 429 Passthrough:

```bash
# This should return HTTP 429 (not 500)
curl -s -w "\nHTTP_STATUS: %{http_code}\n" \
  -X POST "http://localhost:8000/api/nexus/ragnarok/direct/submit?business_name=Test"
```

Expected output:
```
{"detail":"Daily limit (3 workflows) exceeded"}
HTTP_STATUS: 429
```

### Test Enriched Generation:

```bash
curl -s -X POST "http://localhost:8000/api/nexus/ragnarok/generate/enriched" \
  -H "Content-Type: application/json" \
  -d '{"company_name":"Barrios A2I","industry":"AI Automation"}'
```

Expected response:
```json
{
    "job_id": "job_abc123...",
    "workflow_id": "uuid...",
    "status": "queued",
    "tracking_url": "/api/nexus/ragnarok/jobs/job_abc123...",
    "enrichment": {...},
    "trace_id": "nxs_..."
}
```

### Poll Job Status:

```bash
curl -s "http://localhost:8000/api/nexus/ragnarok/jobs/{job_id}"
```

Expected response when complete:
```json
{
    "job_id": "job_abc123...",
    "workflow_id": "uuid...",
    "status": "completed",
    "video_url": "s3://commercials/1080p/...",
    "cost_usd": 2.64,
    "quality_score": 0.88,
    "enrichment": {...}
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND                             │
│  (Only knows about /api/nexus/ragnarok/*)               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   NEXUS :8000                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Gateway Proxy Layer                   │  │
│  │  - Creates Nexus job_id for every request         │  │
│  │  - Propagates errors with correct status codes    │  │
│  │  - Background polling updates job status          │  │
│  │  - Unified polling via /jobs/{job_id}             │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Trinity :8002  │     │ RAGNAROK :8001  │
│  (Intelligence) │     │ (Video Gen)     │
└─────────────────┘     └─────────────────┘
```

## Files Modified

1. `backend/app/services/ragnarok_service.py`
   - Added `RagnarokAPIError` exception class
   - Updated error handling to preserve status codes

2. `backend/app/routers/ragnarok.py`
   - Updated imports to include `RagnarokAPIError`
   - Updated `/generate/enriched` to create Nexus jobs
   - Updated `/direct/submit` to create Nexus jobs
   - Updated `/jobs/{job_id}` to fetch RAGNAROK status

---

*Gateway Proxy pattern ensures frontend simplicity - the JS only ever calls Nexus.*
