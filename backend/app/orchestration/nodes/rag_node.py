"""
NEXUS BRAIN v5.0 APEX - RAG Node
=================================
Third node in pipeline: retrieves knowledge chunks from Qdrant.

Implements dual-scope RAG:
1. Always retrieves barrios_a2i chunks (company knowledge)
2. Also retrieves detected industry chunks

This ensures Nexus ALWAYS knows about Barrios A2I services, pricing, etc.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..state import ConversationState, RAGResult, RetrievedChunk

logger = logging.getLogger("nexus.node.rag")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("NEXUS_KNOWLEDGE_COLLECTION", "nexus_knowledge")

# RAG settings
MAX_CHUNKS = 5
MIN_SCORE = 0.15


async def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    import httpx

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        return []

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-3-small",
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


async def search_qdrant(
    query_embedding: List[float],
    industries: List[str],
    limit: int = MAX_CHUNKS,
) -> List[RetrievedChunk]:
    """
    Search Qdrant for relevant chunks.

    Args:
        query_embedding: Query vector
        industries: Industries to filter by (dual-scope)
        limit: Max chunks to return

    Returns:
        List of RetrievedChunk objects
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("Qdrant credentials not configured")
        return []

    try:
        client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Build filter for dual-scope (always include barrios_a2i)
        search_filter = Filter(
            should=[
                FieldCondition(
                    key="industry",
                    match=MatchAny(any=industries),
                )
            ]
        )

        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=MIN_SCORE,
        )

        await client.close()

        chunks = []
        for result in results:
            payload = result.payload or {}
            chunks.append(RetrievedChunk(
                content=payload.get("content", ""),
                score=result.score,
                industry=payload.get("industry", "unknown"),
                chunk_type=payload.get("type", "unknown"),
                source_title=payload.get("source_title", ""),
                priority=payload.get("priority", "normal"),
                quality_score=payload.get("quality_score", 0.5),
            ))

        return chunks

    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []


async def rag_node(state: ConversationState) -> Dict[str, Any]:
    """
    RAG node: retrieve relevant knowledge chunks.

    Implements dual-scope retrieval:
    1. Always includes barrios_a2i industry (company knowledge)
    2. Also includes detected industry from classifier

    This ensures Nexus ALWAYS knows about Barrios A2I pricing, services, etc.

    Args:
        state: Current conversation state

    Returns:
        State updates with RAG results and context chunks
    """
    start_time = time.time()
    message = state["message"]
    detected_industry = state.get("detected_industry", "general")

    logger.info(f"RAG retrieval for: {message[:50]}...")

    # Build dual-scope industry filter
    industries = ["barrios_a2i"]  # Always include company knowledge
    if detected_industry != "barrios_a2i" and detected_industry != "general":
        industries.append(detected_industry)

    logger.debug(f"Dual-scope industries: {industries}")

    # Get embedding and search
    try:
        embedding = await get_embedding(message)
        if not embedding:
            logger.error("Failed to get embedding")
            return _empty_result(state, start_time)

        chunks = await search_qdrant(embedding, industries)

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return _empty_result(state, start_time)

    # Analyze results
    company_chunks = sum(1 for c in chunks if c.industry == "barrios_a2i")
    industry_chunks = sum(1 for c in chunks if c.industry != "barrios_a2i")
    avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        f"RAG complete: {len(chunks)} chunks "
        f"(company={company_chunks}, industry={industry_chunks}, "
        f"avg_score={avg_score:.3f}), elapsed={elapsed:.1f}ms"
    )

    rag_result = RAGResult(
        chunks=chunks,
        company_chunks=company_chunks,
        industry_chunks=industry_chunks,
        total_chunks=len(chunks),
        avg_score=avg_score,
    )

    return {
        "rag_result": rag_result,
        "context_chunks": chunks,  # Uses operator.add reducer
        "company_knowledge_found": company_chunks > 0,
        "node_timings": {
            **state.get("node_timings", {}),
            "rag": elapsed,
        },
    }


def _empty_result(state: ConversationState, start_time: float) -> Dict[str, Any]:
    """Return empty RAG result on error."""
    elapsed = (time.time() - start_time) * 1000
    return {
        "rag_result": RAGResult(),
        "context_chunks": [],
        "company_knowledge_found": False,
        "node_timings": {
            **state.get("node_timings", {}),
            "rag": elapsed,
        },
        "errors": [f"RAG retrieval failed"],
    }
