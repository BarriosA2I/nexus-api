"""
BOOTSTRAP COMPANY KNOWLEDGE - Barrios A2I Core
===============================================
Seeds critical company information into Qdrant so Nexus ALWAYS knows:
- Pricing (Marketing Overlord, Neural Ad Forge, Cinesite, Total Command)
- Services and offerings
- Positioning and philosophy
- Deployment options

RUN THIS ONCE on deployment to ensure company omniscience.

Usage:
    python bootstrap_company_knowledge.py

Environment:
    QDRANT_URL (required)
    QDRANT_API_KEY (required)
    OPENAI_API_KEY (required for embeddings)
"""

import asyncio
import hashlib
import os
import sys
from typing import List, Dict, Any

import httpx

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# COMPANY CORE KNOWLEDGE (SOURCE OF TRUTH)
# =============================================================================

COMPANY_CORE: List[Dict[str, Any]] = [
    # Company Overview
    {
        "content": "Barrios A2I is an AI automation consultancy founded by Gary Barrios. Tagline: 'This is not automation. This is operational intelligence.' Core concept: 'Your Business. With a Nervous System.' We install autonomous intelligence that senses, decides, and acts across your operations without human bottlenecks.",
        "industry": "barrios_a2i",
        "type": "company_overview",
        "source_title": "Barrios A2I Website",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Barrios A2I philosophy: 'Autonomy compounds. Manual effort decays.' We believe businesses should have nervous systems that sense market changes, decide on optimal responses, and act without human bottlenecks. Our systems run 24/7 with zero human interaction required.",
        "industry": "barrios_a2i",
        "type": "philosophy",
        "source_title": "About Page",
        "priority": "high",
        "quality_score": 0.95
    },

    # PRICING - CRITICAL
    {
        "content": "Marketing Overlord: $199 per month subscription. AI-powered marketing automation system with multi-channel orchestration (email, social, ads), 24/7 autonomous optimization, performance analytics dashboard, and campaign A/B testing. Live in 48 hours after signup.",
        "industry": "barrios_a2i",
        "type": "pricing",
        "source_title": "Pricing Page",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Neural Ad Forge: $500 per video, one-time fee. AI commercial generation delivering professional videos in 24-48 hours. Includes script writing, AI voiceover, visual generation, 3 revision rounds, and full ownership rights. Cost to produce: $2.60 per video using RAGNAROK v7.0 pipeline.",
        "industry": "barrios_a2i",
        "type": "pricing",
        "source_title": "Pricing Page",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Cinesite Autopilot: $1,500 one-time payment. Self-optimizing landing pages that A/B test themselves in real-time. Includes conversion-optimized templates, automatic variant testing, performance analytics, and continuous optimization. Perfect for agencies and businesses wanting maximum conversion rates.",
        "industry": "barrios_a2i",
        "type": "pricing",
        "source_title": "Pricing Page",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Total Command: Custom enterprise pricing ranging from $50,000 to $300,000. Full autonomous AI system installation including custom agent development, integration with existing tech stack, 90-day implementation support, performance guarantees, and dedicated success manager. Timeline: 4-12 weeks depending on scope.",
        "industry": "barrios_a2i",
        "type": "pricing",
        "source_title": "Pricing Page",
        "priority": "critical",
        "quality_score": 1.0
    },

    # Business Model
    {
        "content": "Barrios A2I deployment paths: Option A - Free build in exchange for 30% equity stake (for promising startup ideas with high growth potential). Option B - Flat fee payment for 100% client ownership of all delivered systems and IP.",
        "industry": "barrios_a2i",
        "type": "business_model",
        "source_title": "About Page",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Target market for Barrios A2I: Marketing directors at B2B SaaS companies, digital agency owners seeking automation, e-commerce managers needing 24/7 operations, and enterprises ready for AI transformation. High-ticket positioning: $50K-$300K for custom AI systems.",
        "industry": "barrios_a2i",
        "type": "target_market",
        "source_title": "Business Strategy",
        "priority": "high",
        "quality_score": 0.95
    },

    # Services Description
    {
        "content": "Barrios A2I RAG Research Agents: Autonomous systems that scrape competitor websites, analyze market data, monitor industry trends, and provide actionable intelligence reports. Run 24/7 without human intervention, continuously updating your knowledge base.",
        "industry": "barrios_a2i",
        "type": "service",
        "source_title": "Services Page",
        "priority": "high",
        "quality_score": 0.95
    },
    {
        "content": "Barrios A2I Marketing Overlord System: Automated marketing campaigns with AI-driven content generation, multi-channel distribution (email, social media, paid ads), lead generation, and performance optimization. Replaces entire marketing teams with autonomous intelligence.",
        "industry": "barrios_a2i",
        "type": "service",
        "source_title": "Services Page",
        "priority": "high",
        "quality_score": 0.95
    },
    {
        "content": "Barrios A2I Legendary Websites: Not basic chatbots - intelligent website assistants with generative UI, React components, cyberpunk/HUD aesthetics. The Nexus assistant you're talking to right now is an example of this technology.",
        "industry": "barrios_a2i",
        "type": "service",
        "source_title": "Services Page",
        "priority": "high",
        "quality_score": 0.95
    },

    # Technical Capabilities
    {
        "content": "RAGNAROK v7.0 APEX: Barrios A2I's 9-agent video generation system. Produces professional commercials in 243 seconds at $2.60 cost per video with 97.5% success rate. Agents: Script, Director, Visual, Audio, VFX, QA, Render, Export, Delivery.",
        "industry": "barrios_a2i",
        "type": "technology",
        "source_title": "Technical Documentation",
        "priority": "high",
        "quality_score": 0.95
    },
    {
        "content": "Trinity Orchestrator: Barrios A2I's 3-agent market intelligence system. Continuously researches competitors, analyzes market trends, and feeds insights to other systems. Data-driven commercial generation pipeline: Trinity (intel) -> RAGNAROK (video).",
        "industry": "barrios_a2i",
        "type": "technology",
        "source_title": "Technical Documentation",
        "priority": "high",
        "quality_score": 0.90
    },

    # Contact and Next Steps
    {
        "content": "To get started with Barrios A2I: Book a 15-minute discovery call at cal.com/barriosa2i/discovery. We'll discuss your needs, recommend the right solution, and provide a custom proposal within 48 hours. No obligation, no pressure.",
        "industry": "barrios_a2i",
        "type": "cta",
        "source_title": "Contact Page",
        "priority": "critical",
        "quality_score": 1.0
    },
    {
        "content": "Barrios A2I contact: Website at barriosa2i.com. Email inquiries welcome. Calendar booking for discovery calls available 24/7. Response time: Within 24 hours for all inquiries.",
        "industry": "barrios_a2i",
        "type": "contact",
        "source_title": "Contact Page",
        "priority": "high",
        "quality_score": 0.95
    },
]


# =============================================================================
# EMBEDDING & QDRANT FUNCTIONS
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


async def get_embedding(text: str, api_key: str) -> List[float]:
    """Get embedding from OpenAI API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENAI_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


async def seed_qdrant(
    qdrant_url: str,
    qdrant_api_key: str,
    openai_api_key: str,
    collection_name: str = "nexus_knowledge",
    recreate_collection: bool = False
) -> int:
    """
    Seed Qdrant with company knowledge.

    Returns number of chunks seeded.
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )

    print(f"\n{'='*60}")
    print("BARRIOS A2I COMPANY KNOWLEDGE BOOTSTRAP")
    print(f"{'='*60}\n")

    # Connect to Qdrant
    print(f"[1/5] Connecting to Qdrant...")
    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Check/create collection
    print(f"[2/5] Checking collection '{collection_name}'...")
    collections = await client.get_collections()
    collection_exists = any(c.name == collection_name for c in collections.collections)

    if recreate_collection and collection_exists:
        print(f"       Recreating collection (recreate_collection=True)...")
        await client.delete_collection(collection_name)
        collection_exists = False

    if not collection_exists:
        print(f"       Creating collection with {EMBEDDING_DIMENSIONS} dimensions...")
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"       Collection exists, will upsert points...")

    # Delete existing barrios_a2i chunks (to avoid duplicates)
    print(f"[3/5] Removing existing barrios_a2i chunks...")
    try:
        await client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="industry",
                        match=MatchValue(value="barrios_a2i")
                    )
                ]
            )
        )
        print(f"       Cleared existing company knowledge")
    except Exception as e:
        print(f"       No existing chunks to clear ({e})")

    # Generate embeddings and upsert
    print(f"[4/5] Generating embeddings and upserting {len(COMPANY_CORE)} chunks...")
    points = []

    for i, chunk in enumerate(COMPANY_CORE):
        content = chunk["content"]

        # Generate deterministic ID from content
        point_id = hashlib.md5(content.encode()).hexdigest()

        # Get embedding
        print(f"       [{i+1}/{len(COMPANY_CORE)}] Embedding: {chunk['type']} - {content[:50]}...")
        embedding = await get_embedding(content, openai_api_key)

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "content": content,
                "industry": chunk["industry"],
                "type": chunk["type"],
                "source_title": chunk["source_title"],
                "priority": chunk["priority"],
                "quality_score": chunk["quality_score"],
                "citations": [chunk["source_title"]]
            }
        ))

    # Batch upsert
    await client.upsert(collection_name=collection_name, points=points)

    # Verify
    print(f"[5/5] Verifying...")
    count_result = await client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="industry",
                    match=MatchValue(value="barrios_a2i")
                )
            ]
        )
    )

    await client.close()

    print(f"\n{'='*60}")
    print(f"SUCCESS: Seeded {count_result.count} company knowledge chunks")
    print(f"{'='*60}\n")

    print("Company knowledge now includes:")
    print("  - Marketing Overlord: $199/mo")
    print("  - Neural Ad Forge: $500/video")
    print("  - Cinesite Autopilot: $1,500")
    print("  - Total Command: $50K-$300K")
    print("  - Philosophy, services, contact info")
    print("\nNexus will ALWAYS know Barrios A2I offerings!\n")

    return count_result.count


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run bootstrap."""
    # Load config
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Validate
    missing = []
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("\nSet them in .env or environment before running.")
        sys.exit(1)

    # Run
    try:
        count = await seed_qdrant(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            openai_api_key=openai_api_key,
            collection_name=os.getenv("NEXUS_KNOWLEDGE_COLLECTION", "nexus_knowledge"),
            recreate_collection=False
        )
        print(f"Bootstrap complete: {count} chunks seeded")
    except Exception as e:
        print(f"ERROR: Bootstrap failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
