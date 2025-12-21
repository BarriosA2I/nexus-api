"""
NEXUS RAG CLIENT v2.0 - OpenAI Embeddings
==========================================
Async-friendly Qdrant retrieval using OpenAI Embeddings API.

CRITICAL CHANGE from v1.0:
- Removed sentence-transformers (2GB memory footprint)
- Now uses OpenAI text-embedding-3-small (1536 dimensions)
- Zero local ML dependencies = works on Render free tier

Features:
- Async Qdrant client (non-blocking)
- OpenAI Embeddings API (async httpx)
- Dual-scope retrieval (industry + company-core)
- Pydantic models for type safety
- Graceful degradation if Qdrant unavailable

Environment:
- QDRANT_URL (required)
- QDRANT_API_KEY (optional)
- OPENAI_API_KEY (required for embeddings)
- NEXUS_KNOWLEDGE_COLLECTION (default: nexus_knowledge)
- NEXUS_RAG_SCORE_THRESHOLD (default: 0.5)
- NEXUS_RAG_TOP_K (default: 5)
- NEXUS_RAG_INCLUDE_COMPANY (default: true)

Usage:
    rag = await get_rag_client()
    if rag.enabled:
        chunks = await rag.retrieve(query="I run a dental practice", industry="dental_practices")
        context = rag.format_context(chunks)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.rag")

# OpenAI Embeddings Config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


# =============================================================================
# OPTIONAL DEPENDENCIES (graceful degradation)
# =============================================================================

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed - RAG disabled")


# =============================================================================
# DATA MODELS
# =============================================================================

class RetrievedChunk(BaseModel):
    """Single knowledge chunk retrieved from Qdrant."""
    content: str = Field(default="")
    type: str = Field(default="unknown")
    industry: str = Field(default="")
    score: float = Field(default=0.0)
    quality_score: float = Field(default=0.8)
    citations: List[str] = Field(default_factory=list)


@dataclass
class _Config:
    """Internal configuration for RAG client."""
    qdrant_url: str
    qdrant_api_key: Optional[str]
    openai_api_key: Optional[str]
    collection: str
    score_threshold: float
    top_k: int
    include_company_scope: bool


# =============================================================================
# RAG CLIENT
# =============================================================================

class NexusRAGClient:
    """
    Retrieves industry/company knowledge from Qdrant vector store.

    Uses OpenAI Embeddings API instead of local sentence-transformers.
    Thread-safe singleton with async-friendly operations.
    """

    def __init__(self, config: _Config):
        self._cfg = config
        self._client: Optional[Any] = None  # AsyncQdrantClient
        self._http_client: Optional[httpx.AsyncClient] = None
        self._enabled: bool = False
        self._embedder_loaded: bool = False
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Check if RAG is enabled and ready."""
        return self._enabled

    @property
    def embedder_loaded(self) -> bool:
        """Check if embedder is ready (OpenAI API configured)."""
        return self._embedder_loaded

    @property
    def collection_name(self) -> str:
        """Get the configured collection name."""
        return self._cfg.collection

    def is_enabled(self) -> bool:
        """Alias for enabled property (backwards compatibility)."""
        return self._enabled

    async def initialize(self) -> bool:
        """
        Initialize Qdrant client and verify OpenAI API key.

        Returns True if successfully enabled, False otherwise.
        """
        async with self._lock:
            if self._enabled:
                return True

            # Check Qdrant dependency
            if not QDRANT_AVAILABLE:
                logger.warning("RAG disabled: qdrant-client not installed")
                return False

            # Check URL
            if not self._cfg.qdrant_url:
                logger.info("RAG disabled: QDRANT_URL not set")
                return False

            # Check OpenAI API key
            if not self._cfg.openai_api_key:
                logger.warning("RAG disabled: OPENAI_API_KEY not set (required for embeddings)")
                return False

            try:
                # Initialize HTTP client for OpenAI API
                self._http_client = httpx.AsyncClient(timeout=30.0)
                self._embedder_loaded = True

                # Initialize Qdrant client
                self._client = AsyncQdrantClient(
                    url=self._cfg.qdrant_url,
                    api_key=self._cfg.qdrant_api_key,
                )

                # Verify connection + collection
                collections = await self._client.get_collections()
                if not any(c.name == self._cfg.collection for c in collections.collections):
                    logger.warning(
                        "RAG disabled: collection '%s' not found",
                        self._cfg.collection
                    )
                    return False

                self._enabled = True
                logger.info(
                    "RAG enabled (url=%s collection=%s embedder=openai/%s)",
                    self._cfg.qdrant_url[:50] + "...",
                    self._cfg.collection,
                    EMBEDDING_MODEL
                )
                return True

            except Exception as e:
                logger.exception("RAG initialization failed: %s", e)
                self._enabled = False
                return False

    async def close(self) -> None:
        """Close Qdrant client and HTTP client connections."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
        self._enabled = False

    async def _embed(self, text: str) -> List[float]:
        """Embed text using OpenAI Embeddings API."""
        if not self._http_client or not self._cfg.openai_api_key:
            raise RuntimeError("OpenAI API not configured")

        response = await self._http_client.post(
            OPENAI_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self._cfg.openai_api_key}",
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

    def _get_scopes(self, industry: Optional[str]) -> List[Optional[str]]:
        """
        Get search scopes in priority order.

        Returns [industry, "barrios_a2i"] if include_company_scope is True.
        """
        scopes: List[Optional[str]] = []

        if industry:
            scopes.append(industry)

        if self._cfg.include_company_scope and "barrios_a2i" not in scopes:
            scopes.append("barrios_a2i")

        return scopes

    async def retrieve(
        self,
        query: str,
        industry: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve knowledge chunks from Qdrant.

        Strategy:
        1. Embed query using OpenAI API
        2. Search per scope (industry, then company-core)
        3. Merge and dedupe by score

        Args:
            query: User's question/message
            industry: Industry filter (e.g., "dental_practices")
            limit: Max chunks to return (default: config top_k)

        Returns:
            List of RetrievedChunk sorted by score (desc)
        """
        if not self._enabled or not self._client:
            return []

        top_k = limit or self._cfg.top_k
        if top_k <= 0:
            return []

        try:
            # Embed query using OpenAI
            query_vector = await self._embed(query)

            # Search each scope
            all_chunks: List[RetrievedChunk] = []
            seen_content: set = set()  # Dedupe

            for scope in self._get_scopes(industry):
                # Build filter
                search_filter: Optional[Filter] = None
                if scope:
                    search_filter = Filter(
                        must=[
                            FieldCondition(
                                key="industry",
                                match=MatchValue(value=scope)
                            )
                        ]
                    )

                # Search using query_points (async client API)
                response = await self._client.query_points(
                    collection_name=self._cfg.collection,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=self._cfg.score_threshold,
                    with_payload=True,
                )

                # Parse results from QueryResponse
                results = response.points if hasattr(response, 'points') else []
                for r in results:
                    payload = r.payload or {}
                    content = str(payload.get("content", ""))

                    # Skip duplicates
                    content_hash = hash(content[:100])
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)

                    all_chunks.append(
                        RetrievedChunk(
                            content=content,
                            type=str(payload.get("type", "unknown")),
                            industry=str(payload.get("industry", scope or "")),
                            score=float(r.score or 0.0),
                            quality_score=float(payload.get("quality_score", 0.8) or 0.8),
                            citations=list(payload.get("citations", []) or []),
                        )
                    )

            # Sort by score and limit
            all_chunks.sort(key=lambda c: c.score, reverse=True)
            return all_chunks[:top_k]

        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return []

    def format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into prompt context.

        Returns empty string if no chunks.
        """
        if not chunks:
            return ""

        lines: List[str] = [
            "## INDUSTRY INTELLIGENCE (from research database)\n",
            "Use this specific data to give actionable advice:\n"
        ]

        for chunk in chunks:
            chunk_type = (chunk.type or "unknown").lower()
            content = chunk.content.strip().replace("\n", " ")

            if not content:
                continue

            if chunk_type == "pain_point":
                lines.append(f"- **Pain Point**: {content}")
            elif chunk_type == "automation":
                lines.append(f"- **Automation Opportunity**: {content}")
            elif chunk_type == "objection":
                lines.append(f"- **Common Objection**: {content}")
            elif chunk_type == "terminology":
                lines.append(f"- **Industry Term**: {content}")
            elif chunk_type == "roi":
                lines.append(f"- **ROI Data**: {content}")
            elif chunk_type == "script":
                lines.append(f"- **Sales Angle**: {content}")
            else:
                lines.append(f"- **Insight**: {content}")

        return "\n".join(lines) + "\n"


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_rag_singleton: Optional[NexusRAGClient] = None


def _load_config() -> _Config:
    """Load configuration from environment variables."""
    return _Config(
        qdrant_url=os.getenv("QDRANT_URL", "").strip(),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        collection=os.getenv("NEXUS_KNOWLEDGE_COLLECTION", "nexus_knowledge"),
        score_threshold=float(os.getenv("NEXUS_RAG_SCORE_THRESHOLD", "0.5")),
        top_k=int(os.getenv("NEXUS_RAG_TOP_K", "5")),
        include_company_scope=os.getenv("NEXUS_RAG_INCLUDE_COMPANY", "true").lower() in {
            "1", "true", "yes", "y", "on"
        },
    )


async def get_rag_client() -> NexusRAGClient:
    """
    Get or initialize the process-wide RAG client singleton.

    Thread-safe and async-friendly.

    Usage:
        rag = await get_rag_client()
        if rag.enabled:
            chunks = await rag.retrieve("query", industry="dental_practices")
    """
    global _rag_singleton

    if _rag_singleton is None:
        _rag_singleton = NexusRAGClient(_load_config())
        await _rag_singleton.initialize()

    return _rag_singleton
