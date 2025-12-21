"""
Run Nexus Research Oracle to gather industry intelligence.
"""
import asyncio
import os
import sys
import json
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('.env')

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.research_oracle import (
    NexusResearchOracle,
    ResearchTrigger,
    ResearchPriority,
)

async def run_research(industry: str = "dental_practices"):
    """Run research for a specific industry."""
    print(f"\n{'='*60}")
    print(f"NEXUS RESEARCH ORACLE - Gathering Intelligence")
    print(f"Industry: {industry}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Get credentials
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not perplexity_key:
        print("[ERROR] PERPLEXITY_API_KEY not set")
        return None
    if not anthropic_key:
        print("[ERROR] ANTHROPIC_API_KEY not set")
        return None

    print(f"[OK] Perplexity API: {perplexity_key[:12]}...")
    print(f"[OK] Anthropic API: {anthropic_key[:12]}...")
    print(f"[OK] Qdrant URL: {qdrant_url[:40]}..." if qdrant_url else "[WARN] Qdrant not configured")

    # Initialize Oracle
    print("\n[INFO] Initializing Research Oracle...")
    oracle = NexusResearchOracle(
        perplexity_api_key=perplexity_key,
        anthropic_api_key=anthropic_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )

    # Queue research task
    print(f"\n[INFO] Queuing research for '{industry}'...")
    task_id = await oracle.queue_research(
        industry=industry,
        trigger=ResearchTrigger.MANUAL,
        priority=ResearchPriority.HIGH,
    )
    print(f"[OK] Task queued: {task_id}")

    # Process the task
    print("\n[INFO] Processing research (this may take 30-60 seconds)...")
    print("[INFO] -> Perplexity gathering web research...")
    print("[INFO] -> Haiku extracting structured data...")
    print("[INFO] -> Generating embeddings...")
    print("[INFO] -> Storing in Qdrant...\n")

    result = await oracle.process_next_task()

    if result:
        print(f"\n{'='*60}")
        print("RESEARCH COMPLETE!")
        print(f"{'='*60}")
        print(f"Quality Score: {result.get('quality_score', 'N/A')}")
        print(f"Duration: {result.get('duration_ms', 0)}ms")
        print(f"Chunks Created: {len(result.get('chunks', []))}")

        print("\nPipeline Messages:")
        for msg in result.get("messages", []):
            print(f"  {msg}")

        # Show sample chunks
        chunks = result.get("chunks", [])
        if chunks:
            print(f"\nSample Chunks (first 3):")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i+1} ---")
                content = chunk.get("content", "")[:200]
                print(f"Content: {content}...")
                print(f"Type: {chunk.get('chunk_type', 'N/A')}")

        return result
    else:
        print("[ERROR] Research returned no result")
        return None


async def main():
    # Default to dental_practices, or use command line arg
    industry = sys.argv[1] if len(sys.argv) > 1 else "dental_practices"

    result = await run_research(industry)

    if result:
        print(f"\n[SUCCESS] Research data gathered and stored!")
        print(f"[INFO] Query Qdrant 'nexus_knowledge' collection to retrieve")
    else:
        print(f"\n[FAILED] Research did not complete successfully")


if __name__ == "__main__":
    asyncio.run(main())
