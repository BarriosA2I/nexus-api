# Nexus Assistant Unified

Production-ready FastAPI backend for Nexus Assistant with RAG and RAGNAROK integration.

## Features

- **TF-IDF RAG**: Zero-cost local retrieval (no API costs)
- **RAGNAROK Bridge**: Commercial video generation pipeline integration
- **Circuit Breakers**: Netflix Hystrix pattern for resilience
- **SSE Streaming**: Real-time response streaming
- **Async Job Queue**: Background job processing

## Quick Start

### 1. Install Dependencies

```bash
cd nexus_assistant_unified/backend
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# OR directly
python -m app.main
```

### 4. Verify

```bash
# Health check
curl http://localhost:8000/api/nexus/health

# Chat test
curl -N -H "Content-Type: application/json" \
  -d '{"message":"What is the v2.1 architecture?"}' \
  http://localhost:8000/api/nexus/chat
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/nexus/health` | GET | System health + component status |
| `/api/nexus/chat` | POST | SSE streaming chat (RAG or RAGNAROK) |
| `/api/nexus/ragnarok/generate` | POST | Queue commercial generation job |
| `/api/nexus/ragnarok/jobs/{id}` | GET | Get job status + results |
| `/api/nexus/ragnarok/jobs` | GET | List recent jobs |
| `/docs` | GET | Swagger UI |

## SSE Event Format

```
data: {"type":"status","step":"semantic_search","message":"Searching...","trace_id":"nxs_xxx"}
data: {"type":"chunk","content":"The "}
data: {"type":"chunk","content":"Nexus "}
data: {"type":"complete","trace_id":"nxs_xxx","confidence":0.95,"sources":[...]}
```

## File Structure

```
nexus_assistant_unified/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── config.py         # Configuration
│   │   ├── schemas.py        # Pydantic models
│   │   ├── routers/
│   │   │   ├── nexus.py      # Chat + health
│   │   │   └── ragnarok.py   # Job management
│   │   └── services/
│   │       ├── rag_local.py      # TF-IDF RAG
│   │       ├── job_store.py      # Async jobs
│   │       ├── ragnarok_bridge.py
│   │       └── circuit_breaker.py
│   ├── requirements.txt
│   └── .env
├── knowledge/
│   ├── nexus-v2.1-synthesis-report.txt
│   └── nexus-sync-engine-v2.1-enhanced-spec.txt
└── README.md
```

## Frontend Integration

Point your frontend to:
- Development: `http://localhost:8000/api/nexus`
- Production: Your deployed URL

## RAGNAROK Integration

Set environment variables to connect real RAGNAROK:

```bash
RAGNAROK_V6_PATH=C:\Users\gary\python-commercial-video-agents\ragnarok_v6_legendary
```
