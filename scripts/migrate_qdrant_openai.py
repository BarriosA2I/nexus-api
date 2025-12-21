#!/usr/bin/env python3
"""
QDRANT MIGRATION SCRIPT - OpenAI Embeddings
============================================
Migrates nexus_knowledge collection from sentence-transformers (384 dims)
to OpenAI text-embedding-3-small (1536 dims).

Usage:
    python scripts/migrate_qdrant_openai.py --dry-run   # Preview changes
    python scripts/migrate_qdrant_openai.py             # Execute migration

Environment Variables Required:
    QDRANT_URL      - Qdrant Cloud URL
    QDRANT_API_KEY  - Qdrant Cloud API key
    OPENAI_API_KEY  - OpenAI API key for embeddings

Author: Barrios A2I
Date: 2024
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

# Check dependencies
try:
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install qdrant-client httpx")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

COLLECTION_NAME = "nexus_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


@dataclass
class ChunkData:
    """Knowledge chunk to be embedded and stored."""
    content: str
    type: str  # pain_point, automation, objection, terminology, roi, script
    industry: str
    quality_score: float = 0.9
    citations: Optional[List[str]] = None


# =============================================================================
# DENTAL PRACTICE KNOWLEDGE BASE
# =============================================================================

DENTAL_PRACTICE_CHUNKS: List[ChunkData] = [
    # Pain Points - Staffing
    ChunkData(
        content="Dental practices spend $5,000-$10,000 per new hire on recruitment, training, and onboarding costs. High turnover in front desk and hygienist positions creates ongoing expense.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.95,
        citations=["ADA Practice Analysis 2023", "Dental Economics Staffing Report"]
    ),
    ChunkData(
        content="Staff turnover in dental practices averages 25-30% annually, with replacement costs reaching 50-200% of annual salary for skilled positions like hygienists.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.9,
        citations=["Dental Practice Management Association"]
    ),

    # Pain Points - Insurance & Revenue
    ChunkData(
        content="Insurance claim delays and denials cost dental practices $50,000-$150,000 annually in lost or delayed revenue. Average claim takes 14-30 days to process.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.95,
        citations=["Dental Claims Processing Study 2023"]
    ),
    ChunkData(
        content="Revenue leakage from missed appointments, unbilled procedures, and coding errors ranges from 10-30% of potential revenue in dental practices.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.9,
        citations=["Dental Revenue Cycle Management Report"]
    ),
    ChunkData(
        content="Dental practices lose an average of $150-$200 per missed appointment. No-show rates average 10-15% without automated reminder systems.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.85,
        citations=["Patient Engagement Analytics"]
    ),

    # Pain Points - Operations
    ChunkData(
        content="Front desk staff spend 40-60% of their time on phone calls for scheduling, confirmations, and insurance verification instead of patient care.",
        type="pain_point",
        industry="dental_practices",
        quality_score=0.85,
        citations=["Dental Office Workflow Study"]
    ),

    # Automation Opportunities
    ChunkData(
        content="AI-powered claims processing can reduce denial rates by 30-50% and accelerate reimbursement by 7-14 days through intelligent coding and pre-submission validation.",
        type="automation",
        industry="dental_practices",
        quality_score=0.95,
        citations=["Healthcare AI Implementation Guide"]
    ),
    ChunkData(
        content="Automated patient communication (appointment reminders, post-visit follow-ups, recall campaigns) can reduce no-show rates by 50-70% and increase patient retention by 25%.",
        type="automation",
        industry="dental_practices",
        quality_score=0.9,
        citations=["Patient Engagement Technology Report"]
    ),
    ChunkData(
        content="AI recruitment automation can reduce time-to-hire by 40-60% and improve candidate quality through intelligent screening and matching for dental positions.",
        type="automation",
        industry="dental_practices",
        quality_score=0.85,
        citations=["Healthcare Recruitment Technology Study"]
    ),
    ChunkData(
        content="Automated treatment cost estimation using AI can provide instant, accurate patient quotes, improving case acceptance rates by 20-35%.",
        type="automation",
        industry="dental_practices",
        quality_score=0.9,
        citations=["Dental Treatment Planning Analytics"]
    ),

    # ROI Data
    ChunkData(
        content="Dental practices implementing AI automation see average ROI of 300-500% within first year through reduced labor costs, improved collections, and increased patient volume.",
        type="roi",
        industry="dental_practices",
        quality_score=0.9,
        citations=["Dental Technology ROI Study 2023"]
    ),
    ChunkData(
        content="Practices using AI-powered revenue cycle management recover an additional $75,000-$200,000 annually in previously lost revenue from denied claims and missed billing.",
        type="roi",
        industry="dental_practices",
        quality_score=0.95,
        citations=["Revenue Cycle Optimization Report"]
    ),

    # Industry Terminology
    ChunkData(
        content="Key dental practice metrics: Production (total value of services), Collection Rate (% of production collected), Overhead (typically 60-65%), Case Acceptance Rate (% of treatment plans accepted).",
        type="terminology",
        industry="dental_practices",
        quality_score=0.85,
        citations=["Dental Practice Management Glossary"]
    ),
]


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_openai_embedding(text: str, api_key: str) -> List[float]:
    """Get embedding from OpenAI API."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
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


def embed_chunks(chunks: List[ChunkData], api_key: str, dry_run: bool = False) -> List[tuple]:
    """Embed all chunks and return (chunk, vector) pairs."""
    results = []

    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] Embedding: {chunk.content[:60]}...")

        if dry_run:
            # Fake vector for dry run
            vector = [0.0] * EMBEDDING_DIMENSIONS
        else:
            vector = get_openai_embedding(chunk.content, api_key)
            # Rate limit protection
            time.sleep(0.1)

        results.append((chunk, vector))

    return results


# =============================================================================
# QDRANT OPERATIONS
# =============================================================================

def connect_qdrant(url: str, api_key: Optional[str]) -> QdrantClient:
    """Connect to Qdrant Cloud."""
    return QdrantClient(url=url, api_key=api_key)


def delete_collection(client: QdrantClient, name: str, dry_run: bool = False) -> bool:
    """Delete existing collection."""
    try:
        collections = client.get_collections()
        exists = any(c.name == name for c in collections.collections)

        if not exists:
            print(f"  Collection '{name}' does not exist, nothing to delete")
            return True

        if dry_run:
            print(f"  [DRY RUN] Would delete collection '{name}'")
            return True

        client.delete_collection(name)
        print(f"  Deleted collection '{name}'")
        return True

    except Exception as e:
        print(f"  Error deleting collection: {e}")
        return False


def create_collection(client: QdrantClient, name: str, dry_run: bool = False) -> bool:
    """Create new collection with 1536 dimensions."""
    try:
        if dry_run:
            print(f"  [DRY RUN] Would create collection '{name}' with:")
            print(f"    - Vector size: {EMBEDDING_DIMENSIONS}")
            print(f"    - Distance: Cosine")
            return True

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE
            )
        )
        print(f"  Created collection '{name}' (size={EMBEDDING_DIMENSIONS}, distance=Cosine)")
        return True

    except Exception as e:
        print(f"  Error creating collection: {e}")
        return False


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    embedded_chunks: List[tuple],
    dry_run: bool = False
) -> bool:
    """Upsert embedded chunks to Qdrant."""
    try:
        points = []

        for chunk, vector in embedded_chunks:
            point = PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "content": chunk.content,
                    "type": chunk.type,
                    "industry": chunk.industry,
                    "quality_score": chunk.quality_score,
                    "citations": chunk.citations or [],
                }
            )
            points.append(point)

        if dry_run:
            print(f"  [DRY RUN] Would upsert {len(points)} points")
            for i, (chunk, _) in enumerate(embedded_chunks):
                print(f"    {i+1}. [{chunk.type}] {chunk.content[:50]}...")
            return True

        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"  Upserted {len(points)} points to '{collection_name}'")
        return True

    except Exception as e:
        print(f"  Error upserting chunks: {e}")
        return False


def verify_collection(client: QdrantClient, name: str) -> bool:
    """Verify collection exists and has correct config."""
    try:
        info = client.get_collection(name)
        print(f"\n  Collection '{name}' verified:")
        print(f"    - Points: {info.points_count}")
        print(f"    - Vector size: {info.config.params.vectors.size}")
        print(f"    - Distance: {info.config.params.vectors.distance}")
        return True
    except Exception as e:
        print(f"  Error verifying collection: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection to OpenAI embeddings (1536 dims)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing"
    )
    args = parser.parse_args()

    dry_run = args.dry_run

    print("=" * 60)
    print("QDRANT MIGRATION - OpenAI Embeddings")
    print("=" * 60)

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Load environment variables
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Validate
    missing = []
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("\nSet them with:")
        for var in missing:
            print(f"  export {var}=your_value")
        sys.exit(1)

    print(f"Qdrant URL: {qdrant_url[:50]}...")
    print(f"OpenAI API Key: {openai_api_key[:10]}...{openai_api_key[-4:]}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Vector Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Chunks to migrate: {len(DENTAL_PRACTICE_CHUNKS)}")

    # Step 1: Connect to Qdrant
    print("\n[1/5] Connecting to Qdrant Cloud...")
    try:
        client = connect_qdrant(qdrant_url, qdrant_api_key)
        print("  Connected successfully")
    except Exception as e:
        print(f"  ERROR: Failed to connect: {e}")
        sys.exit(1)

    # Step 2: Delete old collection
    print(f"\n[2/5] Deleting old collection '{COLLECTION_NAME}'...")
    if not delete_collection(client, COLLECTION_NAME, dry_run):
        sys.exit(1)

    # Step 3: Create new collection
    print(f"\n[3/5] Creating new collection '{COLLECTION_NAME}' (1536 dims)...")
    if not create_collection(client, COLLECTION_NAME, dry_run):
        sys.exit(1)

    # Step 4: Embed chunks
    print(f"\n[4/5] Embedding {len(DENTAL_PRACTICE_CHUNKS)} chunks with OpenAI...")
    embedded_chunks = embed_chunks(DENTAL_PRACTICE_CHUNKS, openai_api_key, dry_run)
    print(f"  Embedded {len(embedded_chunks)} chunks")

    # Step 5: Upsert to Qdrant
    print(f"\n[5/5] Upserting chunks to '{COLLECTION_NAME}'...")
    if not upsert_chunks(client, COLLECTION_NAME, embedded_chunks, dry_run):
        sys.exit(1)

    # Verify
    if not dry_run:
        print("\n[VERIFY] Checking collection...")
        verify_collection(client, COLLECTION_NAME)

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print("Run without --dry-run to execute migration")
    else:
        print("MIGRATION COMPLETE")
        print(f"Collection '{COLLECTION_NAME}' now has {len(DENTAL_PRACTICE_CHUNKS)} chunks")
        print(f"Using OpenAI {EMBEDDING_MODEL} embeddings ({EMBEDDING_DIMENSIONS} dims)")
    print("=" * 60)


if __name__ == "__main__":
    main()
