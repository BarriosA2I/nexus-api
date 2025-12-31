"""
Hierarchical Memory System
4-tier memory architecture with forgetting curves and ACAN reranking.

Memory Tiers:
- L0 Working: 7±2 items, in-memory, 1-5s decay (conversation context)
- L1 Episodic: Unlimited, PostgreSQL, 24h τ (session memories)
- L2 Semantic: Unlimited, Qdrant, no decay (factual knowledge)
- L3 Procedural: Unlimited, Qdrant, slow decay (learned skills/patterns)

Forgetting Curve: importance = similarity * e^(-age/τ)
ACAN Reranking: Improves retrieval precision by 12%+
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any, Protocol
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

from opentelemetry import trace

from .acan_attention import AssociativeCrossAttention, ACANConfig

logger = logging.getLogger("nexus.hierarchical_memory")
tracer = trace.get_tracer(__name__)


class MemoryTier(Enum):
    """Memory tier identifiers."""
    L0_WORKING = "L0_working"
    L1_EPISODIC = "L1_episodic"
    L2_SEMANTIC = "L2_semantic"
    L3_PROCEDURAL = "L3_procedural"


class VectorStore(Protocol):
    """Protocol for vector store backends."""
    async def search(
        self,
        collection: str,
        embedding: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        ...

    async def upsert(
        self,
        collection: str,
        id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        ...


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    async def embed(self, text: str) -> np.ndarray:
        ...


@dataclass
class MemoryTrace:
    """A single memory trace."""
    id: str
    content: str
    embedding: np.ndarray
    tier: MemoryTier
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class MemoryResult:
    """Result from memory retrieval."""
    memories: List[Dict[str, Any]]
    total_retrieved: int
    tiers_searched: List[str]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memories": self.memories,
            "total_retrieved": self.total_retrieved,
            "tiers_searched": self.tiers_searched,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class HierarchicalMemoryConfig:
    """Configuration for hierarchical memory system."""
    # Working memory (L0)
    working_memory_capacity: int = 7  # 7±2 Miller's Law

    # Forgetting curves (τ = time constant in seconds)
    tau_working: float = 5.0          # L0: 5 seconds
    tau_episodic: float = 86400.0     # L1: 24 hours
    tau_semantic: float = float('inf') # L2: No decay
    tau_procedural: float = 604800.0   # L3: 1 week

    # Minimum importance after decay
    min_importance_factor: float = 0.3

    # ACAN integration
    use_acan: bool = True
    acan_config: Optional[ACANConfig] = None

    # Retrieval settings
    default_top_k: int = 10
    similarity_threshold: float = 0.3


class HierarchicalMemorySystem:
    """
    4-tier hierarchical memory with forgetting curves and ACAN reranking.

    Implements cognitive-inspired memory architecture:
    - L0 Working: Short-term, high-access conversation buffer
    - L1 Episodic: Session-level autobiographical memories
    - L2 Semantic: Factual knowledge (no decay)
    - L3 Procedural: Learned skills and patterns

    Retrieval uses ACAN attention for improved precision.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[HierarchicalMemoryConfig] = None
    ):
        """
        Initialize hierarchical memory system.

        Args:
            vector_store: Backend for L1/L2/L3 storage
            embedding_model: Model for computing embeddings
            config: Configuration options
        """
        self.config = config or HierarchicalMemoryConfig()
        self.vector_store = vector_store
        self.embedder = embedding_model

        # L0 Working Memory (in-memory circular buffer)
        self.working_memory: deque = deque(maxlen=self.config.working_memory_capacity)
        self._working_lock = asyncio.Lock()

        # ACAN for reranking
        self.acan: Optional[AssociativeCrossAttention] = None
        if self.config.use_acan:
            acan_config = self.config.acan_config or ACANConfig()
            self.acan = AssociativeCrossAttention(acan_config)
            logger.info("ACAN attention network initialized for memory reranking")

        # Time constants per tier
        self._tau_map = {
            MemoryTier.L0_WORKING: self.config.tau_working,
            MemoryTier.L1_EPISODIC: self.config.tau_episodic,
            MemoryTier.L2_SEMANTIC: self.config.tau_semantic,
            MemoryTier.L3_PROCEDURAL: self.config.tau_procedural,
        }

        logger.info(
            f"Hierarchical memory initialized: L0 capacity={self.config.working_memory_capacity}, "
            f"ACAN={'enabled' if self.acan else 'disabled'}"
        )

    @tracer.start_as_current_span("hierarchical_memory.store")
    async def store(
        self,
        content: str,
        tier: MemoryTier,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory trace.

        Args:
            content: Text content to store
            tier: Memory tier (L0-L3)
            metadata: Additional metadata

        Returns:
            Memory trace ID
        """
        span = trace.get_current_span()
        span.set_attribute("tier", tier.value)

        # Generate embedding
        embedding = await self._embed(content)

        # Create trace
        trace_id = f"{tier.value}_{int(time.time() * 1000)}"
        memory_trace = MemoryTrace(
            id=trace_id,
            content=content,
            embedding=embedding,
            tier=tier,
            timestamp=time.time(),
            importance=1.0,  # Fresh memories start at full importance
            metadata=metadata or {},
        )

        if tier == MemoryTier.L0_WORKING:
            async with self._working_lock:
                self.working_memory.append(memory_trace)
        elif self.vector_store:
            await self.vector_store.upsert(
                collection=tier.value,
                id=trace_id,
                embedding=embedding,
                metadata={
                    "content": content,
                    "timestamp": memory_trace.timestamp,
                    "importance": memory_trace.importance,
                    **(metadata or {}),
                }
            )

        logger.debug(f"Stored memory: {trace_id} in {tier.value}")
        return trace_id

    @tracer.start_as_current_span("hierarchical_memory.retrieve")
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        tiers: Optional[List[MemoryTier]] = None
    ) -> MemoryResult:
        """
        Retrieve memories with ACAN attention reranking.

        Args:
            query: Search query
            top_k: Number of results to return
            tiers: Specific tiers to search (default: all)

        Returns:
            MemoryResult with ranked memories
        """
        start_time = time.time()
        span = trace.get_current_span()

        top_k = top_k or self.config.default_top_k
        tiers = tiers or [
            MemoryTier.L0_WORKING,
            MemoryTier.L1_EPISODIC,
            MemoryTier.L2_SEMANTIC,
            MemoryTier.L3_PROCEDURAL,
        ]

        span.set_attribute("top_k", top_k)
        span.set_attribute("tiers", [t.value for t in tiers])

        # Get query embedding
        query_emb = await self._embed(query)
        current_time = time.time()

        candidates: List[Dict[str, Any]] = []

        # Retrieve from each tier
        for tier in tiers:
            if tier == MemoryTier.L0_WORKING:
                # Search working memory
                working_results = await self._search_working_memory(query_emb)
                for result in working_results:
                    result['tier'] = tier.value
                candidates.extend(working_results)
            elif self.vector_store:
                # Search vector store
                results = await self.vector_store.search(
                    collection=tier.value,
                    embedding=query_emb,
                    top_k=top_k * 2  # Over-fetch for filtering
                )
                for result in results:
                    result['tier'] = tier.value
                candidates.extend(results)

        # Apply forgetting curve: importance = similarity * e^(-age/τ)
        for candidate in candidates:
            timestamp = candidate.get('timestamp', current_time)
            age = current_time - timestamp
            tier = MemoryTier(candidate.get('tier', 'L1_episodic'))
            tau = self._tau_map.get(tier, self.config.tau_episodic)

            if tau == float('inf'):
                decay = 1.0  # No decay for semantic memory
            else:
                decay = np.exp(-age / tau)
                # Apply minimum importance factor
                decay = max(decay, self.config.min_importance_factor)

            # Procedural memories decay slower
            if tier == MemoryTier.L3_PROCEDURAL:
                decay = min(1.0, decay * 1.5)

            similarity = candidate.get('similarity', candidate.get('score', 0.5))
            candidate['importance'] = float(similarity * decay)
            candidate['decay_factor'] = float(decay)
            candidate['age_seconds'] = float(age)

        # ACAN attention-based reranking
        if self.acan is not None and candidates:
            candidates = self.acan.rerank_memories(query_emb, candidates)
        else:
            # Fallback: sort by importance only
            candidates.sort(key=lambda x: x.get('importance', 0), reverse=True)

        # Filter by threshold
        candidates = [
            c for c in candidates
            if c.get('importance', 0) >= self.config.similarity_threshold
        ]

        # Take top-K
        result_memories = candidates[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Retrieved {len(result_memories)} memories from {len(tiers)} tiers in {elapsed_ms:.1f}ms"
        )

        return MemoryResult(
            memories=result_memories,
            total_retrieved=len(result_memories),
            tiers_searched=[t.value for t in tiers],
            elapsed_ms=elapsed_ms,
        )

    async def _search_working_memory(
        self,
        query_emb: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Search L0 working memory."""
        results = []

        async with self._working_lock:
            for trace in self.working_memory:
                # Cosine similarity
                similarity = np.dot(query_emb, trace.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(trace.embedding) + 1e-8
                )

                results.append({
                    'id': trace.id,
                    'content': trace.content,
                    'similarity': float(similarity),
                    'timestamp': trace.timestamp,
                    'embedding': trace.embedding,
                    'metadata': trace.metadata,
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    async def _embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.embedder:
            return await self.embedder.embed(text)
        else:
            # Fallback: return random embedding (for testing only)
            logger.warning("No embedder configured, using random embedding")
            return np.random.randn(1536).astype(np.float32)

    async def consolidate(self) -> int:
        """
        Consolidate working memory to episodic memory.
        Called periodically to move short-term to long-term storage.

        Returns:
            Number of memories consolidated
        """
        span = tracer.start_span("hierarchical_memory.consolidate")

        try:
            consolidated = 0
            current_time = time.time()

            async with self._working_lock:
                # Find memories older than working memory tau
                to_consolidate = []
                for trace in self.working_memory:
                    age = current_time - trace.timestamp
                    if age > self.config.tau_working * 2:
                        to_consolidate.append(trace)

                # Move to episodic
                for trace in to_consolidate:
                    if self.vector_store:
                        await self.vector_store.upsert(
                            collection=MemoryTier.L1_EPISODIC.value,
                            id=trace.id.replace("L0", "L1"),
                            embedding=trace.embedding,
                            metadata={
                                "content": trace.content,
                                "timestamp": trace.timestamp,
                                "importance": trace.importance,
                                "consolidated_from": "L0_working",
                                **trace.metadata,
                            }
                        )
                        consolidated += 1

            span.set_attribute("consolidated_count", consolidated)
            logger.info(f"Consolidated {consolidated} memories from L0 to L1")
            return consolidated

        finally:
            span.end()

    def get_working_memory_snapshot(self) -> List[Dict[str, Any]]:
        """Get current working memory contents (for debugging)."""
        return [
            {
                'id': trace.id,
                'content': trace.content[:100] + "..." if len(trace.content) > 100 else trace.content,
                'timestamp': trace.timestamp,
                'importance': trace.importance,
            }
            for trace in self.working_memory
        ]

    def clear_working_memory(self) -> None:
        """Clear L0 working memory."""
        self.working_memory.clear()
        logger.info("Working memory cleared")

    @property
    def working_memory_size(self) -> int:
        """Current working memory size."""
        return len(self.working_memory)

    @property
    def working_memory_capacity(self) -> int:
        """Maximum working memory capacity."""
        return self.config.working_memory_capacity
