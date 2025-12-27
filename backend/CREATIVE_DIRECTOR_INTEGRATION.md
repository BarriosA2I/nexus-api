# 🎬 NEXUS UNIFIED + CREATIVE DIRECTOR INTEGRATION HANDOFF

## 📅 Date: December 27, 2024
## 🎯 Priority: HIGH
## 🎯 Objective: Wire Creative Director 6-agent pipeline into NEXUS Unified

---

## 📋 CONTEXT

We've built a complete Creative Director integration into NEXUS Unified that:
- Detects video creation intent from user messages
- Seamlessly hands off to Creative Director intake conversation
- Runs the 6-agent pipeline (Research → Ideation → Script → Review)
- Connects to RAGNAROK for video generation
- Returns user to NEXUS after video delivery

The integration uses:
- **Intent Detection**: Pattern + keyword matching for video requests
- **Session Management**: Tracks CD workflow state within NEXUS sessions
- **Pipeline Bridge**: Orchestrates the 6 legendary agents
- **Unified Router**: Intelligent routing between NEXUS and CD

---

## 📁 FILE INVENTORY

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `app/creative_director/__init__.py` | 60 | Module exports |
| `app/creative_director/intent_detector.py` | 200 | Detects video creation intent |
| `app/creative_director/session_manager.py` | 550 | Manages CD sessions + intake |
| `app/creative_director/pipeline_bridge.py` | 550 | Bridges to 6-agent pipeline |
| `app/creative_director/unified_router.py` | 400 | Intelligent routing layer |
| `app/creative_director/creative_director_wiring.py` | 1100 | External system adapters |
| `app/routers/creative_director.py` | 350 | FastAPI endpoints |

---

## 🏗️ ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NEXUS UNIFIED v6.0                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Message ──► UnifiedRouter ──┬──► NEXUS Brain (standard)       │
│                                   │                                  │
│                                   └──► Creative Director ──┐         │
│                                                            │         │
│  ┌─────────────────────────────────────────────────────────┘         │
│  │                                                                   │
│  ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    CREATIVE DIRECTOR                            │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │                                                                 │ │
│  │  IntentDetector ──► SessionManager ──► IntakeConversation       │ │
│  │                                              │                  │ │
│  │                                              ▼                  │ │
│  │                                        BriefData                │ │
│  │                                              │                  │ │
│  │                                              ▼                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │              PIPELINE BRIDGE (6 Agents)                  │   │ │
│  │  ├──────────────────────────────────────────────────────────┤   │ │
│  │  │  🔍 Research ──► 💡 Ideation ──► ✍️ Script ──► ✅ Review  │   │ │
│  │  └─────────────────────────────────────│────────────────────┘   │ │
│  │                                        │                        │ │
│  │                                        ▼                        │ │
│  │                                   RAGNAROK v7.0                 │ │
│  │                                        │                        │ │
│  │                                        ▼                        │ │
│  │                                   Video URL                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 INSTALLATION STEPS

### Step 1: Ensure All Files Are In Place
```
C:\Users\gary\nexus_unified\
├── app/
│   ├── creative_director/
│   │   ├── __init__.py
│   │   ├── intent_detector.py
│   │   ├── session_manager.py
│   │   ├── pipeline_bridge.py
│   │   ├── unified_router.py
│   │   └── creative_director_wiring.py
│   ├── routers/
│   │   └── creative_director.py
│   └── main.py (updated with CD router)
```

### Step 2: Install Additional Dependencies
```bash
pip install aiohttp httpx qdrant-client aio-pika
```

### Step 3: Environment Variables
```bash
# Creative Director specific
export TRINITY_MCP_ENDPOINT="http://localhost:8080"
export QDRANT_URL="http://localhost:6333"
export RAGNAROK_ENDPOINT="http://localhost:9000"
export RAGNAROK_API_KEY="your-key"
```

### Step 4: Run NEXUS with Creative Director
```bash
python -m app.main brain
# OR
python -m app.main all
```

---

## 🔌 API ENDPOINTS

### Creative Director Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/creative-director/session` | Create new CD session |
| `GET` | `/api/creative-director/session/{id}` | Get session details |
| `DELETE` | `/api/creative-director/session/{id}` | Close session |
| `POST` | `/api/creative-director/intake` | Process intake message |
| `POST` | `/api/creative-director/brief` | Submit complete brief |
| `POST` | `/api/creative-director/pipeline/start` | Start 6-agent pipeline |
| `GET` | `/api/creative-director/pipeline/status/{id}` | Get pipeline status |
| `GET` | `/api/creative-director/pipeline/stream/{id}` | SSE event stream |
| `GET` | `/api/creative-director/concepts/{id}` | Get generated concepts |
| `POST` | `/api/creative-director/concepts/select` | Select concept |
| `GET` | `/api/creative-director/video/{id}` | Get video status |
| `POST` | `/api/creative-director/intent/detect` | Detect video intent |
| `GET` | `/api/creative-director/health` | Health check |

---

## 🎯 USAGE EXAMPLES

### Example 1: Create Session + Intake Conversation
```python
import httpx

async with httpx.AsyncClient() as client:
    # Create session
    resp = await client.post("http://localhost:8000/api/creative-director/session", json={
        "user_id": "user_123"
    })
    session = resp.json()
    session_id = session["session_id"]
    
    # Process intake messages
    messages = [
        "FitTech Pro",
        "Fitness technology",
        "Health-conscious millennials",
        "TikTok",
        "30 seconds",
        "Transform your fitness with AI",
        "AI personalization, 24/7 coaching",
        "Bold and energetic",
        "Premium",
        "Peloton, Nike Training Club",
        "yes"
    ]
    
    for msg in messages:
        resp = await client.post("http://localhost:8000/api/creative-director/intake", json={
            "session_id": session_id,
            "message": msg
        })
        result = resp.json()
        print(f"Phase: {result['phase']}, Complete: {result['is_complete']}")
        
        if result["is_complete"]:
            break
```

### Example 2: Direct Brief + Pipeline
```python
async with httpx.AsyncClient() as client:
    # Create session
    resp = await client.post("http://localhost:8000/api/creative-director/session", json={
        "user_id": "user_456"
    })
    session_id = resp.json()["session_id"]
    
    # Submit brief directly
    await client.post("http://localhost:8000/api/creative-director/brief", json={
        "session_id": session_id,
        "business_name": "CloudSync Solutions",
        "industry": "Enterprise SaaS",
        "target_audience": "IT Directors and CTOs",
        "target_platform": "linkedin",
        "video_duration": 60,
        "key_message": "Simplify your cloud infrastructure",
        "unique_selling_points": ["Multi-cloud support", "40% cost reduction"],
        "brand_tone": "professional",
        "budget_tier": "enterprise",
        "competitors": ["Datadog", "New Relic"]
    })
    
    # Start pipeline
    resp = await client.post("http://localhost:8000/api/creative-director/pipeline/start", json={
        "session_id": session_id,
        "auto_select_concept": True
    })
    result = resp.json()
    print(f"Video URL: {result.get('video_url')}")
```

### Example 3: Intent Detection
```python
async with httpx.AsyncClient() as client:
    resp = await client.post("http://localhost:8000/api/creative-director/intent/detect", json={
        "message": "I want to create a video ad for my fitness app"
    })
    intent = resp.json()
    
    if intent["should_handoff"]:
        # Route to Creative Director
        print("Detected video intent, handing off to Creative Director")
    else:
        # Continue with standard NEXUS
        print("Standard NEXUS query")
```

---

## 🧪 TESTING

### Run Intent Detector Test
```python
from app.creative_director.intent_detector import CreativeIntentDetector

detector = CreativeIntentDetector()

test_messages = [
    "I want to create a video ad",           # Should handoff
    "Make me a 30 second commercial",        # Should handoff
    "How much does video creation cost?",    # Video inquiry, no handoff
    "What's the weather today?",             # General, no handoff
]

for msg in test_messages:
    result = detector.detect(msg)
    print(f"{msg}: {result.intent.value} ({result.confidence:.2f}), handoff={result.should_handoff}")
```

### Run Pipeline Test
```bash
python -m app.creative_director.pipeline_bridge
```

---

## 🔄 WIRING REAL EXTERNAL SYSTEMS

Currently using **mock agents** for testing. To wire real systems:

### 1. Trinity MCP Connection
Edit `pipeline_bridge.py`:
```python
from creative_director_wiring import TrinityMCPAdapter, TrinityConfig

self.trinity_client = TrinityMCPAdapter(TrinityConfig(
    mcp_endpoint="http://localhost:8080"
))
```

### 2. Qdrant RAG Connection
```python
from creative_director_wiring import QdrantRAGAdapter, RAGConfig

self.rag_client = QdrantRAGAdapter(RAGConfig(
    qdrant_url="http://localhost:6333",
    collection_name="winning_commercials"
))
```

### 3. RAGNAROK Connection
```python
from creative_director_wiring import RAGNAROKAPIAdapter, RAGNAROKConfig

self.ragnarok_client = RAGNAROKAPIAdapter(RAGNAROKConfig(
    api_endpoint="http://localhost:9000",
    api_key=os.getenv("RAGNAROK_API_KEY")
))
```

---

## ✅ VERIFICATION CHECKLIST

After installation, verify:

- [ ] `/api/creative-director/health` returns healthy
- [ ] Session creation works
- [ ] Intake conversation processes messages
- [ ] Intent detection identifies video requests
- [ ] Brief submission works
- [ ] Pipeline starts (with mock agents)
- [ ] Events stream via SSE
- [ ] Video URL returned on completion

---

## 📊 METRICS

The integration exposes Prometheus metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `nexus_cd_pipeline_runs_total` | Counter | Total pipeline runs by status |
| `nexus_cd_pipeline_duration_seconds` | Histogram | Duration by phase |
| `nexus_cd_active_productions` | Gauge | Currently active productions |

Access at: `http://localhost:8000/metrics`

---

## 🐛 TROUBLESHOOTING

### Session Not Found
- Ensure session was created first
- Check session_id format (should be `cd-{nexus_session_id}`)

### Pipeline Stuck
- Check mock vs real agents
- Verify external system connectivity
- Check logs for circuit breaker trips

### Intent Not Detected
- Lower confidence threshold in `CreativeIntentDetector`
- Add domain-specific keywords to `VIDEO_KEYWORDS`

---

## 📞 NEXT STEPS

1. **Test End-to-End** with mock agents
2. **Wire Real Agents** when Trinity/RAGNAROK deployed
3. **Add Webhooks** for async pipeline completion
4. **Configure Alerts** for production monitoring
5. **Load Test** with concurrent sessions

---

**🎬 Creative Director Integration v6.0**
**Barrios A2I Cognitive Systems Division**
