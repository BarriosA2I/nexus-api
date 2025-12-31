"""
NEXUS BRAIN v5.0 APEX - RAG Node
=================================
Third node in pipeline: retrieves knowledge chunks from Qdrant.

Implements dual-scope RAG + spreading activation:
1. Always retrieves barrios_a2i chunks (company knowledge)
2. Also retrieves detected industry chunks
3. For MODERATE+ complexity: graph traversal via spreading activation

This ensures Nexus ALWAYS knows about Barrios A2I services, pricing, etc.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..state import ConversationState, RAGResult, RetrievedChunk

# Spreading activation integration (optional, graceful fallback)
try:
    from ...services.spreading_activation import (
        SpreadingActivationRetriever,
        SpreadingActivationConfig,
        ActivatedFact,
    )
    SPREADING_ACTIVATION_AVAILABLE = True
except ImportError:
    SPREADING_ACTIVATION_AVAILABLE = False

logger = logging.getLogger("nexus.node.rag")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("NEXUS_KNOWLEDGE_COLLECTION", "nexus_knowledge")

# RAG settings
MAX_CHUNKS = 5
MIN_SCORE = 0.15

# Neo4j settings for spreading activation (optional)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Spreading activation singleton
_spreading_activation: Optional['SpreadingActivationRetriever'] = None


def _get_spreading_activation() -> Optional['SpreadingActivationRetriever']:
    """Get or create spreading activation retriever singleton."""
    global _spreading_activation

    if not SPREADING_ACTIVATION_AVAILABLE:
        return None

    if _spreading_activation is None and NEO4J_URI and NEO4J_PASSWORD:
        try:
            config = SpreadingActivationConfig(
                max_hops=3,
                decay_factor=0.7,
                activation_threshold=0.1,
                top_k_facts=10,
                use_neo4j=True,
            )
            _spreading_activation = SpreadingActivationRetriever(
                neo4j_uri=NEO4J_URI,
                neo4j_auth=(NEO4J_USER, NEO4J_PASSWORD),
                config=config,
            )
            logger.info("Spreading activation retriever initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize spreading activation: {e}")
            return None

    return _spreading_activation


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


def _rrf_fusion(
    list1: List[RetrievedChunk],
    list2: List[RetrievedChunk],
    k: int = 60
) -> List[RetrievedChunk]:
    """
    Reciprocal Rank Fusion to combine ranked lists.

    RRF score = sum(1 / (k + rank_i)) across all lists

    Args:
        list1: First ranked list (vector search)
        list2: Second ranked list (graph search)
        k: Smoothing constant (default 60)

    Returns:
        Combined and re-ranked list
    """
    # Build content -> chunk mapping (use first occurrence)
    chunk_map: Dict[str, RetrievedChunk] = {}
    rrf_scores: Dict[str, float] = {}

    # Score list1
    for rank, chunk in enumerate(list1):
        content = chunk.content
        if content not in chunk_map:
            chunk_map[content] = chunk
            rrf_scores[content] = 0.0
        rrf_scores[content] += 1.0 / (k + rank + 1)

    # Score list2
    for rank, chunk in enumerate(list2):
        content = chunk.content
        if content not in chunk_map:
            chunk_map[content] = chunk
            rrf_scores[content] = 0.0
        rrf_scores[content] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

    # Update scores and return
    result = []
    for content in sorted_contents:
        chunk = chunk_map[content]
        # Update chunk score to reflect RRF ranking
        chunk.score = rrf_scores[content]
        result.append(chunk)

    return result[:MAX_CHUNKS * 2]  # Return more than usual for RRF


async def rag_node(state: ConversationState) -> Dict[str, Any]:
    """
    RAG node: retrieve relevant knowledge chunks.

    Implements dual-scope retrieval + spreading activation:
    1. Always includes barrios_a2i industry (company knowledge)
    2. Also includes detected industry from classifier
    3. For MODERATE+ complexity: graph traversal via spreading activation
    4. RRF fusion combines vector + graph results

    This ensures Nexus ALWAYS knows about Barrios A2I pricing, services, etc.

    Args:
        state: Current conversation state

    Returns:
        State updates with RAG results and context chunks
    """
    start_time = time.time()
    message = state["message"]
    detected_industry = state.get("detected_industry", "general")
    complexity = state.get("complexity_level", "simple")

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

        # Vector search (primary)
        chunks = await search_qdrant(embedding, industries)

        # Spreading activation for MODERATE+ complexity
        graph_chunks: List[RetrievedChunk] = []
        if complexity in ("moderate", "complex") and SPREADING_ACTIVATION_AVAILABLE:
            sa_retriever = _get_spreading_activation()
            if sa_retriever:
                try:
                    # Extract seed entities from message
                    seed_entities = sa_retriever.extract_seed_entities(message)
                    if seed_entities:
                        facts = await sa_retriever.retrieve(
                            query=message,
                            seed_entities=seed_entities,
                            max_hops=2 if complexity == "moderate" else 3,
                        )
                        # Convert facts to chunks
                        for fact in facts:
                            graph_chunks.append(RetrievedChunk(
                                content=f"{fact.head} {fact.relation} {fact.tail}",
                                score=fact.activation_score,
                                industry="knowledge_graph",
                                chunk_type="graph_fact",
                                source_title=f"Graph hop {fact.hop_distance}",
                                priority="high" if fact.activation_score > 0.5 else "normal",
                                quality_score=fact.activation_score,
                            ))
                        logger.info(f"Spreading activation: {len(graph_chunks)} facts from {len(seed_entities)} seeds")
                except Exception as e:
                    logger.warning(f"Spreading activation failed: {e}")

        # RRF fusion if we have both sources
        if graph_chunks:
            chunks = _rrf_fusion(chunks, graph_chunks, k=60)
            logger.debug(f"RRF fusion: {len(chunks)} combined chunks")

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
