"""
ACAN: Associative Cross-Attention Network
Learned memory retrieval ranking via cross-attention mechanism.

Based on: Attention Is All You Need (Vaswani et al., 2017)
Purpose: Improve hierarchical memory retrieval precision by 12%+

Architecture:
- Query encoder: Projects query into attention space
- Memory encoder: Projects memory traces into attention space
- Cross-attention: Computes relevance scores between query and memories
- Output: Attention-weighted scores for memory reranking
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from opentelemetry import trace

logger = logging.getLogger("nexus.acan_attention")
tracer = trace.get_tracer(__name__)


@dataclass
class ACANConfig:
    """Configuration for ACAN attention network."""
    embedding_dim: int = 1536  # Match OpenAI text-embedding-3-small
    attention_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    temperature: float = 1.0
    importance_weight: float = 0.6  # Weight for original importance score
    attention_weight: float = 0.4   # Weight for ACAN attention score


if TORCH_AVAILABLE:
    class QueryEncoder(nn.Module):
        """Encodes query into attention space."""

        def __init__(self, config: ACANConfig):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )

        def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
            """Project query embedding to attention space."""
            return self.projection(query_embedding)

    class MemoryEncoder(nn.Module):
        """Encodes memory traces into attention space with tier-aware projections."""

        def __init__(self, config: ACANConfig):
            super().__init__()
            # Separate projections per memory tier
            self.tier_projections = nn.ModuleDict({
                'L0_working': nn.Linear(config.embedding_dim, config.hidden_dim),
                'L1_episodic': nn.Linear(config.embedding_dim, config.hidden_dim),
                'L2_semantic': nn.Linear(config.embedding_dim, config.hidden_dim),
                'L3_procedural': nn.Linear(config.embedding_dim, config.hidden_dim),
            })
            self.layer_norm = nn.LayerNorm(config.hidden_dim)
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            memory_embeddings: torch.Tensor,
            memory_tiers: List[str]
        ) -> torch.Tensor:
            """Project memory embeddings with tier-specific transformations."""
            batch_size = memory_embeddings.shape[0]
            hidden_dim = list(self.tier_projections.values())[0].out_features

            outputs = torch.zeros(batch_size, hidden_dim, device=memory_embeddings.device)

            for i, tier in enumerate(memory_tiers):
                if tier in self.tier_projections:
                    outputs[i] = self.tier_projections[tier](memory_embeddings[i])
                else:
                    # Default to episodic if unknown tier
                    outputs[i] = self.tier_projections['L1_episodic'](memory_embeddings[i])

            return self.dropout(self.layer_norm(outputs))

    class CrossAttentionBlock(nn.Module):
        """Multi-head cross-attention between query and memories."""

        def __init__(self, config: ACANConfig):
            super().__init__()
            self.num_heads = config.attention_heads
            self.head_dim = config.hidden_dim // config.attention_heads
            self.temperature = config.temperature

            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            query: torch.Tensor,  # [1, hidden_dim]
            memories: torch.Tensor  # [num_memories, hidden_dim]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute cross-attention scores.

            Returns:
                output: Attention-weighted memory representation
                attention_weights: Raw attention scores for each memory
            """
            # Project to Q, K, V
            Q = self.q_proj(query)  # [1, hidden_dim]
            K = self.k_proj(memories)  # [num_memories, hidden_dim]
            V = self.v_proj(memories)  # [num_memories, hidden_dim]

            # Reshape for multi-head attention
            batch_size = 1
            num_memories = memories.shape[0]

            Q = Q.view(batch_size, self.num_heads, self.head_dim)
            K = K.view(num_memories, self.num_heads, self.head_dim)
            V = V.view(num_memories, self.num_heads, self.head_dim)

            # Compute attention scores: Q @ K^T / sqrt(d_k)
            attention_scores = torch.einsum('bhd,nhd->bnh', Q, K) / (self.head_dim ** 0.5)
            attention_scores = attention_scores / self.temperature

            # Softmax over memories
            attention_weights = F.softmax(attention_scores, dim=-1)  # [1, num_heads, num_memories]
            attention_weights = self.dropout(attention_weights)

            # Weighted sum of values
            output = torch.einsum('bnh,nhd->bhd', attention_weights, V)
            output = output.view(batch_size, -1)
            output = self.out_proj(output)

            # Average attention across heads for final scores
            final_weights = attention_weights.mean(dim=1).squeeze(0)  # [num_memories]

            return output, final_weights

    class AssociativeCrossAttentionModule(nn.Module):
        """
        ACAN: Full associative cross-attention network for memory ranking.
        PyTorch Module implementation.
        """

        def __init__(self, config: Optional[ACANConfig] = None):
            super().__init__()
            self.config = config or ACANConfig()

            self.query_encoder = QueryEncoder(self.config)
            self.memory_encoder = MemoryEncoder(self.config)
            self.cross_attention = CrossAttentionBlock(self.config)

            # Final scoring head
            self.score_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )


class AssociativeCrossAttention:
    """
    ACAN: Associative Cross-Attention Network wrapper.

    Usage:
        acan = AssociativeCrossAttention(config)
        scores = acan.compute_attention(query_emb, memory_candidates)
        # Combine with importance: final = 0.6 * importance + 0.4 * scores
    """

    def __init__(self, config: Optional[ACANConfig] = None):
        self.config = config or ACANConfig()
        self._model: Optional[Any] = None
        self._initialized = False

        if TORCH_AVAILABLE:
            self._model = AssociativeCrossAttentionModule(self.config)
            self._model.eval()  # Set to evaluation mode
            self._initialized = True
            logger.info("ACAN initialized with PyTorch backend")
        else:
            logger.warning("PyTorch not available, ACAN will use numpy fallback")

    @tracer.start_as_current_span("acan.compute_attention")
    def compute_attention(
        self,
        query_embedding: np.ndarray,
        memory_candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Compute attention scores for memory candidates.

        Args:
            query_embedding: Query vector [embedding_dim]
            memory_candidates: List of dicts with 'embedding', 'tier', 'importance'

        Returns:
            List of attention scores [0, 1] for each memory
        """
        span = trace.get_current_span()
        span.set_attribute("memory_count", len(memory_candidates))

        if not memory_candidates:
            return []

        if TORCH_AVAILABLE and self._model is not None:
            return self._compute_attention_torch(query_embedding, memory_candidates)
        else:
            return self._compute_attention_numpy(query_embedding, memory_candidates)

    def _compute_attention_torch(
        self,
        query_embedding: np.ndarray,
        memory_candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """PyTorch-based attention computation."""
        # Convert to tensors
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)

        memory_embeddings = torch.tensor(
            np.array([m.get('embedding', np.zeros(self.config.embedding_dim))
                     for m in memory_candidates]),
            dtype=torch.float32
        )
        memory_tiers = [m.get('tier', 'L1_episodic') for m in memory_candidates]

        # Forward pass
        with torch.no_grad():
            query_encoded = self._model.query_encoder(query_tensor)
            memories_encoded = self._model.memory_encoder(memory_embeddings, memory_tiers)

            # Cross-attention
            _, attention_weights = self._model.cross_attention(query_encoded, memories_encoded)

            # Additional scoring pass
            scores = self._model.score_head(memories_encoded).squeeze(-1)

            # Combine attention and learned scores
            final_scores = 0.5 * attention_weights + 0.5 * scores

        return final_scores.tolist()

    def _compute_attention_numpy(
        self,
        query_embedding: np.ndarray,
        memory_candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """
        NumPy fallback for attention computation.
        Uses scaled dot-product attention approximation.
        """
        scores = []

        for memory in memory_candidates:
            mem_emb = memory.get('embedding')
            if mem_emb is None:
                scores.append(0.5)
                continue

            mem_emb = np.array(mem_emb)

            # Scaled dot-product attention
            dot_product = np.dot(query_embedding, mem_emb)
            scale = np.sqrt(len(query_embedding))
            attention_score = dot_product / scale

            # Apply softmax-like normalization (sigmoid for single score)
            attention_score = 1 / (1 + np.exp(-attention_score / self.config.temperature))

            # Tier-based adjustment
            tier = memory.get('tier', 'L1_episodic')
            tier_weights = {
                'L0_working': 1.2,   # Recent, high relevance
                'L1_episodic': 1.0,  # Baseline
                'L2_semantic': 0.9,  # Factual, slightly lower
                'L3_procedural': 1.1 # Skills, slightly higher
            }
            tier_weight = tier_weights.get(tier, 1.0)

            final_score = float(attention_score * tier_weight)
            final_score = max(0.0, min(1.0, final_score))  # Clamp
            scores.append(final_score)

        return scores

    def rerank_memories(
        self,
        query_embedding: np.ndarray,
        memory_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank memories using ACAN attention + original importance.

        Args:
            query_embedding: Query vector
            memory_candidates: List with 'embedding', 'tier', 'importance', etc.

        Returns:
            Reranked list with added 'final_score' field
        """
        span = tracer.start_span("acan.rerank_memories")

        try:
            attention_scores = self.compute_attention(query_embedding, memory_candidates)

            for candidate, attn_score in zip(memory_candidates, attention_scores):
                original_importance = candidate.get('importance', 0.5)
                candidate['attention_score'] = attn_score
                candidate['final_score'] = (
                    self.config.importance_weight * original_importance +
                    self.config.attention_weight * attn_score
                )

            # Sort by final score
            memory_candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)

            span.set_attribute("reranked_count", len(memory_candidates))
            logger.debug(f"ACAN reranked {len(memory_candidates)} memories")

            return memory_candidates
        finally:
            span.end()

    def save_model(self, path: str) -> None:
        """Save PyTorch model weights."""
        if TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info(f"ACAN model saved to {path}")
        else:
            logger.warning("Cannot save model: PyTorch not available")

    def load_model(self, path: str) -> None:
        """Load PyTorch model weights."""
        if TORCH_AVAILABLE and self._model is not None:
            self._model.load_state_dict(torch.load(path, map_location='cpu'))
            self._model.eval()
            logger.info(f"ACAN model loaded from {path}")
        else:
            logger.warning("Cannot load model: PyTorch not available")

    @property
    def is_torch_available(self) -> bool:
        """Check if PyTorch backend is available."""
        return TORCH_AVAILABLE and self._initialized
