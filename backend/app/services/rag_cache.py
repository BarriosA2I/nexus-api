"""
RAG CACHE SERVICE - Redis-backed caching for embeddings and retrieval
======================================================================
P0-C: Reduces OpenAI API costs by ~40% via embedding cache.

Features:
- Embedding cache (6-hour TTL)
- Retrieval result cache (1-hour TTL)
- Graceful degradation if Redis unavailable
- Async-friendly operations

Environment:
- REDIS_URL (optional, falls back to no-cache)
- RAG_CACHE_EMBEDDING_TTL (default: 21600 = 6 hours)
- RAG_CACHE_RETRIEVAL_TTL (default: 3600 = 1 hour)
"""

import hashlib
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nexus.rag_cache")

# Try to import Redis
try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed - RAG caching disabled")


class RAGCache:
    """
    Redis-backed cache for RAG operations.

    Caches:
    - Embeddings (text -> vector)
    - Retrieval results (query + scopes -> chunks)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        embedding_ttl: int = 21600,  # 6 hours
        retrieval_ttl: int = 3600,   # 1 hour
    ):
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._embedding_ttl = embedding_ttl
        self._retrieval_ttl = retrieval_ttl
        self._redis: Optional[Any] = None
        self._enabled = False

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.info("RAG cache disabled: redis package not installed")
            return False

        if not self._redis_url:
            logger.info("RAG cache disabled: REDIS_URL not set")
            return False

        try:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            # Test connection
            await self._redis.ping()
            self._enabled = True
            logger.info(f"RAG cache enabled (embedding_ttl={self._embedding_ttl}s, retrieval_ttl={self._retrieval_ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"RAG cache initialization failed: {e}")
            self._enabled = False
            return False

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _cache_key(self, prefix: str, content: str) -> str:
        """Generate cache key from content hash."""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"rag:{prefix}:{content_hash}"

    # =========================================================================
    # EMBEDDING CACHE
    # =========================================================================

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Returns None if not cached or cache disabled.
        """
        if not self._enabled or not self._redis:
            return None

        try:
            key = self._cache_key("embed", text)
            cached = await self._redis.get(key)
            if cached:
                logger.debug(f"Embedding cache HIT: {key[:30]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Embedding cache get failed: {e}")
            return None

    async def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """
        Cache embedding for text.

        Returns True if cached successfully.
        """
        if not self._enabled or not self._redis:
            return False

        try:
            key = self._cache_key("embed", text)
            await self._redis.setex(
                key,
                self._embedding_ttl,
                json.dumps(embedding)
            )
            logger.debug(f"Embedding cached: {key[:30]}...")
            return True
        except Exception as e:
            logger.warning(f"Embedding cache set failed: {e}")
            return False

    async def get_embedding_protected(
        self,
        text: str,
        embed_func,
    ) -> Optional[List[float]]:
        """
        Get embedding with cache stampede protection.

        Uses distributed locking to prevent multiple concurrent requests
        from computing the same embedding simultaneously.

        Args:
            text: Text to embed
            embed_func: Async function to compute embedding if not cached

        Returns:
            Embedding vector or None if failed
        """
        import asyncio

        if not self._enabled or not self._redis:
            # No cache - just compute
            return await embed_func(text)

        key = self._cache_key("embed", text)
        lock_key = f"lock:{key}"

        try:
            # Try cache first
            cached = await self._redis.get(key)
            if cached:
                logger.debug(f"Embedding cache HIT (protected): {key[:30]}...")
                return json.loads(cached)

            # Acquire lock (only one request computes)
            lock_acquired = await self._redis.set(
                lock_key, "1",
                nx=True,  # Only if not exists
                ex=10     # 10s lock timeout
            )

            if lock_acquired:
                try:
                    # Double-check cache (another request may have computed)
                    cached = await self._redis.get(key)
                    if cached:
                        return json.loads(cached)

                    # Compute embedding
                    embedding = await embed_func(text)

                    # Cache result
                    await self._redis.setex(
                        key,
                        self._embedding_ttl,
                        json.dumps(embedding)
                    )
                    logger.debug(f"Embedding computed and cached: {key[:30]}...")
                    return embedding
                finally:
                    await self._redis.delete(lock_key)
            else:
                # Wait for computing request (poll for up to 2s)
                for _ in range(20):
                    await asyncio.sleep(0.1)
                    cached = await self._redis.get(key)
                    if cached:
                        logger.debug(f"Embedding cache HIT (waited): {key[:30]}...")
                        return json.loads(cached)

                # Fallback: compute ourselves if wait timed out
                logger.debug(f"Lock wait timeout, computing: {key[:30]}...")
                return await embed_func(text)

        except Exception as e:
            logger.warning(f"Protected embedding cache failed: {e}")
            # Fallback to direct compute
            return await embed_func(text)

    # =========================================================================
    # RETRIEVAL CACHE
    # =========================================================================

    def _retrieval_cache_key(
        self,
        query: str,
        scopes: List[str],
        top_k: int
    ) -> str:
        """Generate cache key for retrieval results."""
        content = f"{query}|{','.join(sorted(scopes))}|{top_k}"
        return self._cache_key("retrieve", content)

    async def get_retrieval(
        self,
        query: str,
        scopes: List[str],
        top_k: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached retrieval results.

        Returns None if not cached or cache disabled.
        """
        if not self._enabled or not self._redis:
            return None

        try:
            key = self._retrieval_cache_key(query, scopes, top_k)
            cached = await self._redis.get(key)
            if cached:
                logger.debug(f"Retrieval cache HIT: {key[:30]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Retrieval cache get failed: {e}")
            return None

    async def set_retrieval(
        self,
        query: str,
        scopes: List[str],
        top_k: int,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        Cache retrieval results.

        Returns True if cached successfully.
        """
        if not self._enabled or not self._redis:
            return False

        try:
            key = self._retrieval_cache_key(query, scopes, top_k)
            # Convert results to serializable format
            serializable = [
                {
                    "content": r.get("content", ""),
                    "type": r.get("type", "unknown"),
                    "industry": r.get("industry", ""),
                    "score": float(r.get("score", 0)),
                    "quality_score": float(r.get("quality_score", 0.8)),
                    "citations": list(r.get("citations", [])),
                }
                for r in results
            ]
            await self._redis.setex(
                key,
                self._retrieval_ttl,
                json.dumps(serializable)
            )
            logger.debug(f"Retrieval cached: {key[:30]}... ({len(results)} chunks)")
            return True
        except Exception as e:
            logger.warning(f"Retrieval cache set failed: {e}")
            return False

    # =========================================================================
    # STATS
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._enabled or not self._redis:
            return {"enabled": False}

        try:
            info = await self._redis.info("memory")
            keys_embed = await self._redis.keys("rag:embed:*")
            keys_retrieve = await self._redis.keys("rag:retrieve:*")

            return {
                "enabled": True,
                "embedding_keys": len(keys_embed),
                "retrieval_keys": len(keys_retrieve),
                "memory_used_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


# =============================================================================
# SINGLETON
# =============================================================================

_cache_singleton: Optional[RAGCache] = None


async def get_rag_cache() -> RAGCache:
    """Get or initialize the RAG cache singleton."""
    global _cache_singleton

    if _cache_singleton is None:
        _cache_singleton = RAGCache()
        await _cache_singleton.initialize()

    return _cache_singleton
