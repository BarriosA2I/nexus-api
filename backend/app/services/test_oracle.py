"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      NEXUS ORACLE TEST SCRIPT                                ║
║              Verify Research Pipeline End-to-End                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script tests:
1. Perplexity API connection (web research)
2. Haiku API connection (structured extraction)
3. Qdrant connection (vector storage) - optional
4. Full pipeline execution

Usage:
    python test_oracle.py

Required env vars:
    PERPLEXITY_API_KEY - Get from https://perplexity.ai/settings/api
    ANTHROPIC_API_KEY  - Already have this
    QDRANT_URL         - Optional (will skip storage test if not set)
    QDRANT_API_KEY     - Optional
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# COLOR OUTPUT
# ═══════════════════════════════════════════════════════════════════

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_success(msg: str):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")

def log_error(msg: str):
    print(f"{Colors.RED}[FAIL] {msg}{Colors.END}")

def log_info(msg: str):
    print(f"{Colors.CYAN}[INFO] {msg}{Colors.END}")

def log_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}\n")


# ═══════════════════════════════════════════════════════════════════
# TEST 1: PERPLEXITY API
# ═══════════════════════════════════════════════════════════════════

async def test_perplexity():
    """Test Perplexity API connection."""
    log_header("TEST 1: PERPLEXITY API")
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        log_error("PERPLEXITY_API_KEY not set")
        log_info("Get your key at: https://perplexity.ai/settings/api")
        return False
    
    log_info(f"API key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "user", "content": "What are the top 3 challenges facing dental practices in 2025? Be specific with statistics."}
                    ],
                    "return_citations": True,
                    "search_recency_filter": "month",
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                citations = data.get("citations", [])
                tokens = data.get("usage", {}).get("total_tokens", 0)
                
                log_success("Perplexity API working!")
                log_info(f"Tokens used: {tokens}")
                log_info(f"Citations: {len(citations)}")
                print(f"\n{Colors.CYAN}Sample response (first 500 chars):{Colors.END}")
                print(content[:500] + "..." if len(content) > 500 else content)
                
                return {"content": content, "citations": citations, "tokens": tokens}
            else:
                log_error(f"Perplexity API error: {response.status_code}")
                log_error(response.text)
                return False
                
    except Exception as e:
        log_error(f"Perplexity test failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# TEST 2: HAIKU API (Structured Extraction)
# ═══════════════════════════════════════════════════════════════════

async def test_haiku(raw_research: Optional[str] = None):
    """Test Haiku API for structured extraction."""
    log_header("TEST 2: HAIKU API (Structured Extraction)")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log_error("ANTHROPIC_API_KEY not set")
        return False
    
    log_info(f"API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Use real research or sample
    if not raw_research:
        raw_research = """
        Dental practices face major challenges in 2025:
        1. Staffing shortages - 70% report difficulty hiring hygienists, costing $150/hour in lost revenue
        2. No-shows cost practices $200/appointment on average, with 23% no-show rates
        3. Insurance claim denials waste 12 hours/week of staff time at $25/hour = $300/week
        4. Patient acquisition costs $300-500 per new patient through traditional marketing
        5. Manual appointment reminders take 8 hours/week
        
        Automation opportunities:
        - Automated reminders reduce no-shows by 40%, saving $15,000/year for average practice
        - AI-powered insurance verification saves 10 hours/week
        - Online scheduling increases bookings by 25%
        """
    
    try:
        import httpx
        
        extraction_prompt = f"""Extract structured business intelligence from this dental industry research.
Return a JSON object with these exact keys:
- pain_points: array of {{issue, cost_impact}}
- automation_opportunities: array of {{opportunity, roi, difficulty}}
- statistics: array of {{metric, value, source}}

Research:
{raw_research}

Return ONLY valid JSON, no explanation."""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "user", "content": extraction_prompt}
                    ],
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"]
                tokens_in = data.get("usage", {}).get("input_tokens", 0)
                tokens_out = data.get("usage", {}).get("output_tokens", 0)
                
                log_success("Haiku API working!")
                log_info(f"Tokens: {tokens_in} in / {tokens_out} out")
                
                # Try to parse JSON
                try:
                    # Handle markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    structured = json.loads(content.strip())
                    log_success("Structured extraction successful!")
                    print(f"\n{Colors.CYAN}Extracted structure:{Colors.END}")
                    print(json.dumps(structured, indent=2)[:1000])
                    return structured
                except json.JSONDecodeError as e:
                    log_warning(f"JSON parsing failed: {e}")
                    print(content[:500])
                    return {"raw": content}
            else:
                log_error(f"Haiku API error: {response.status_code}")
                log_error(response.text)
                return False
                
    except Exception as e:
        log_error(f"Haiku test failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# TEST 3: QDRANT CONNECTION
# ═══════════════════════════════════════════════════════════════════

async def test_qdrant():
    """Test Qdrant vector store connection."""
    log_header("TEST 3: QDRANT VECTOR STORE")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        log_warning("QDRANT_URL not set - skipping vector store test")
        log_info("Set up Qdrant Cloud at: https://cloud.qdrant.io")
        return None
    
    log_info(f"Qdrant URL: {qdrant_url[:30]}...")
    
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection
        collections = await client.get_collections()
        log_success("Qdrant connection successful!")
        log_info(f"Existing collections: {[c.name for c in collections.collections]}")
        
        # Create test collection if needed
        COLLECTION_NAME = "nexus_knowledge"
        exists = any(c.name == COLLECTION_NAME for c in collections.collections)
        
        if not exists:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            log_success(f"Created collection: {COLLECTION_NAME}")
        else:
            log_info(f"Collection {COLLECTION_NAME} already exists")
        
        # Get collection info
        info = await client.get_collection(COLLECTION_NAME)
        log_info(f"Points count: {info.points_count}")
        
        return True
        
    except Exception as e:
        log_error(f"Qdrant test failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# TEST 4: EMBEDDING SERVICE
# ═══════════════════════════════════════════════════════════════════

async def test_embeddings():
    """Test embedding generation."""
    log_header("TEST 4: EMBEDDING SERVICE")
    
    try:
        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            test_texts = [
                "Dental practice scheduling automation",
                "Law firm document management",
            ]
            
            embeddings = model.encode(test_texts).tolist()
            
            log_success("Sentence Transformers working!")
            log_info(f"Embedding dimension: {len(embeddings[0])}")
            log_info(f"Sample values: {embeddings[0][:5]}")
            return True
            
        except ImportError:
            log_warning("sentence-transformers not installed")
            log_info("Install with: pip install sentence-transformers")
            return None
            
    except Exception as e:
        log_error(f"Embedding test failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# TEST 5: FULL PIPELINE (if Oracle available)
# ═══════════════════════════════════════════════════════════════════

async def test_full_pipeline():
    """Test full Oracle pipeline if available."""
    log_header("TEST 5: FULL ORACLE PIPELINE")
    
    # Check all required keys
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    
    if not perplexity_key or not anthropic_key:
        log_warning("Missing API keys - skipping full pipeline test")
        return None
    
    if not qdrant_url:
        log_warning("QDRANT_URL not set - pipeline will run but won't store results")
    
    try:
        # Import Oracle
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from nexus_research_oracle import NexusResearchOracle, ResearchTrigger, ResearchPriority
        
        log_info("Initializing Oracle...")
        oracle = NexusResearchOracle(
            perplexity_api_key=perplexity_key,
            anthropic_api_key=anthropic_key,
            qdrant_url=qdrant_url or "http://localhost:6333",
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # Queue a test research task
        log_info("Queuing research task for 'dental_practices'...")
        task_id = await oracle.queue_research(
            industry="dental_practices",
            trigger=ResearchTrigger.MANUAL,
            priority=ResearchPriority.HIGH,
        )
        log_success(f"Task queued: {task_id}")
        
        # Process the task
        log_info("Processing research task (this may take 30-60 seconds)...")
        result = await oracle.process_next_task()
        
        if result:
            log_success("Pipeline completed!")
            log_info(f"Quality Score: {result['quality_score']}")
            log_info(f"Duration: {result['duration_ms']}ms")
            log_info(f"Chunks Created: {len(result['chunks'])}")
            
            print(f"\n{Colors.CYAN}Pipeline Messages:{Colors.END}")
            for msg in result["messages"]:
                print(f"  {msg}")
            
            return result
        else:
            log_error("Pipeline returned no result")
            return False
            
    except ImportError as e:
        log_warning(f"Oracle import failed: {e}")
        log_info("Make sure nexus_research_oracle.py is in the same directory")
        return None
    except Exception as e:
        log_error(f"Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

async def main():
    print(f"""
{Colors.BOLD}{Colors.CYAN}
================================================================================
                      NEXUS RESEARCH ORACLE TEST SUITE
              Verifying Data Gathering for Nexus Brain
================================================================================
{Colors.END}
""")

    results = {}
    
    # Test 1: Perplexity
    perplexity_result = await test_perplexity()
    results["perplexity"] = bool(perplexity_result)
    
    # Test 2: Haiku (use Perplexity result if available)
    raw_research = perplexity_result.get("content") if isinstance(perplexity_result, dict) else None
    haiku_result = await test_haiku(raw_research)
    results["haiku"] = bool(haiku_result)
    
    # Test 3: Qdrant
    qdrant_result = await test_qdrant()
    results["qdrant"] = qdrant_result if qdrant_result is not None else "skipped"
    
    # Test 4: Embeddings
    embedding_result = await test_embeddings()
    results["embeddings"] = embedding_result if embedding_result is not None else "skipped"
    
    # Test 5: Full Pipeline (optional)
    if results["perplexity"] and results["haiku"]:
        pipeline_result = await test_full_pipeline()
        results["pipeline"] = bool(pipeline_result) if pipeline_result is not None else "skipped"
    else:
        log_warning("Skipping full pipeline test - Perplexity or Haiku failed")
        results["pipeline"] = "skipped"
    
    # Summary
    log_header("TEST SUMMARY")
    
    for test, result in results.items():
        if result == True:
            log_success(f"{test.upper()}: PASS")
        elif result == False:
            log_error(f"{test.upper()}: FAIL")
        else:
            log_warning(f"{test.upper()}: SKIPPED")
    
    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")

    if not results["perplexity"]:
        print("  -> Get Perplexity API key: https://perplexity.ai/settings/api")
    if results["qdrant"] == "skipped":
        print("  -> Set up Qdrant Cloud: https://cloud.qdrant.io")
    if results["embeddings"] == "skipped":
        print("  -> Install: pip install sentence-transformers")

    all_critical_pass = results["perplexity"] and results["haiku"]

    if all_critical_pass:
        print(f"\n{Colors.GREEN}{Colors.BOLD}SUCCESS! ORACLE READY FOR DATA GATHERING!{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}WARNING: Fix failing tests before proceeding{Colors.END}")
    
    return all_critical_pass


if __name__ == "__main__":
    asyncio.run(main())
