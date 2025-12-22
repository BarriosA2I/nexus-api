"""
CONFIDENCE SCORER - Real Signal-Based Confidence Calculation
=============================================================
P0-D: Replaces arbitrary "vibes" confidence with actual data signals.

Confidence is calculated from:
1. Industry detection quality (semantic router score)
2. RAG retrieval quality (chunk scores from Qdrant)
3. Company knowledge presence (barrios_a2i chunks found)
4. Conversation context depth (history length)

Output: 0.0-1.0 score with breakdown for debugging/transparency.

Usage:
    scorer = ConfidenceScorer()
    result = scorer.calculate(
        industry_confidence=0.75,
        rag_chunks=chunks,
        company_chunks_found=2,
        history_length=5
    )
    # result.score = 0.82
    # result.breakdown = {"industry": 0.3, "rag": 0.2, "company": 0.2, "context": 0.12}
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nexus.confidence")


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence score components."""
    industry: float = 0.0      # 0.0-0.30 (30% weight)
    rag_quality: float = 0.0   # 0.0-0.25 (25% weight)
    company_core: float = 0.0  # 0.0-0.25 (25% weight)
    context: float = 0.0       # 0.0-0.20 (20% weight)

    @property
    def total(self) -> float:
        """Sum of all components (0.0-1.0)."""
        return min(1.0, self.industry + self.rag_quality + self.company_core + self.context)


@dataclass
class ConfidenceResult:
    """
    Final confidence score with breakdown.

    Attributes:
        score: Overall confidence (0.0-1.0)
        level: Human-readable level (low/medium/high)
        breakdown: Component scores for transparency
        signals: Raw signals used in calculation
    """
    score: float
    level: str
    breakdown: ConfidenceBreakdown
    signals: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def low(cls, breakdown: ConfidenceBreakdown, signals: Dict) -> "ConfidenceResult":
        return cls(score=breakdown.total, level="low", breakdown=breakdown, signals=signals)

    @classmethod
    def medium(cls, breakdown: ConfidenceBreakdown, signals: Dict) -> "ConfidenceResult":
        return cls(score=breakdown.total, level="medium", breakdown=breakdown, signals=signals)

    @classmethod
    def high(cls, breakdown: ConfidenceBreakdown, signals: Dict) -> "ConfidenceResult":
        return cls(score=breakdown.total, level="high", breakdown=breakdown, signals=signals)


class ConfidenceScorer:
    """
    Calculate response confidence from real signals.

    Weight distribution:
    - Industry detection: 30% (most important for relevance)
    - RAG retrieval: 25% (quality of knowledge base match)
    - Company core: 25% (ability to speak about Barrios A2I)
    - Context depth: 20% (understanding from conversation)

    Thresholds:
    - 0.0-0.49: LOW - May need clarification, generic response
    - 0.50-0.74: MEDIUM - Good confidence, specific response
    - 0.75-1.0: HIGH - Expert response, full knowledge available
    """

    # Weight configuration
    WEIGHT_INDUSTRY = 0.30
    WEIGHT_RAG = 0.25
    WEIGHT_COMPANY = 0.25
    WEIGHT_CONTEXT = 0.20

    # Thresholds
    THRESHOLD_LOW = 0.50
    THRESHOLD_HIGH = 0.75

    def calculate(
        self,
        industry_confidence: float = 0.0,
        rag_chunks: Optional[List[Any]] = None,
        company_chunks_found: int = 0,
        history_length: int = 0,
    ) -> ConfidenceResult:
        """
        Calculate confidence from real signals.

        Args:
            industry_confidence: Semantic router confidence (0.0-1.0)
            rag_chunks: Retrieved knowledge chunks (RetrievedChunk objects)
            company_chunks_found: Number of barrios_a2i chunks in results
            history_length: Conversation history length

        Returns:
            ConfidenceResult with score, level, and breakdown
        """
        rag_chunks = rag_chunks or []

        # Calculate component scores
        breakdown = ConfidenceBreakdown()

        # 1. Industry component (0-0.30)
        # Strong industry match = high relevance
        breakdown.industry = self._score_industry(industry_confidence)

        # 2. RAG quality component (0-0.25)
        # Good retrieval = good knowledge base match
        breakdown.rag_quality = self._score_rag_quality(rag_chunks)

        # 3. Company core component (0-0.25)
        # Company knowledge = can speak authoritatively about Barrios A2I
        breakdown.company_core = self._score_company_presence(company_chunks_found)

        # 4. Context component (0-0.20)
        # Conversation history = better understanding of user
        breakdown.context = self._score_context_depth(history_length)

        # Calculate total
        total = breakdown.total

        # CRITICAL: Override high confidence if no RAG grounding
        # A response with no retrieved knowledge should never be "high" confidence
        if len(rag_chunks) == 0 and total > 0.5:
            logger.debug(
                f"Ungrounded override: {total:.2f} -> 0.40 (no RAG chunks)"
            )
            total = 0.40  # Force MEDIUM-LOW when ungrounded

        # Determine level
        if total >= self.THRESHOLD_HIGH:
            level = "high"
        elif total >= self.THRESHOLD_LOW:
            level = "medium"
        else:
            level = "low"

        # Build signals dict for debugging
        signals = {
            "industry_confidence": industry_confidence,
            "rag_chunk_count": len(rag_chunks),
            "avg_chunk_score": self._avg_chunk_score(rag_chunks),
            "company_chunks": company_chunks_found,
            "history_length": history_length,
        }

        result = ConfidenceResult(
            score=round(total, 3),
            level=level,
            breakdown=breakdown,
            signals=signals,
        )

        logger.debug(
            f"Confidence calculated: {result.score:.2f} ({result.level}) | "
            f"industry={breakdown.industry:.2f} rag={breakdown.rag_quality:.2f} "
            f"company={breakdown.company_core:.2f} context={breakdown.context:.2f}"
        )

        return result

    def _score_industry(self, confidence: float) -> float:
        """
        Score industry detection (max 0.30).

        Scoring:
        - 0.75+ semantic match: Full points (0.30)
        - 0.50-0.74: Scaled (0.15-0.29)
        - 0.45-0.50: Minimum viable (0.10)
        - <0.45: Regex fallback (0.05)
        - 0.0: No detection (0.0)
        """
        if confidence == 0.0:
            return 0.0
        elif confidence >= 0.75:
            return self.WEIGHT_INDUSTRY  # 0.30
        elif confidence >= 0.50:
            # Scale from 0.15 to 0.29
            ratio = (confidence - 0.50) / 0.25
            return 0.15 + (ratio * 0.15)
        elif confidence >= 0.45:
            return 0.10
        else:
            # Regex fallback gives 0.5 confidence, treat as partial
            return 0.05

    def _score_rag_quality(self, chunks: List[Any]) -> float:
        """
        Score RAG retrieval quality (max 0.25).

        Factors:
        - Number of chunks retrieved (more = better coverage)
        - Average similarity score (higher = more relevant)
        - Quality scores from ingestion (higher = better source)
        """
        if not chunks:
            return 0.0

        chunk_count = len(chunks)
        avg_score = self._avg_chunk_score(chunks)

        # Chunk count bonus (0-0.10)
        # 5 chunks = full bonus, scale linearly
        count_score = min(0.10, chunk_count * 0.02)

        # Average similarity bonus (0-0.15)
        # Score > 0.7 = full bonus
        if avg_score >= 0.70:
            similarity_score = 0.15
        elif avg_score >= 0.50:
            similarity_score = 0.08 + ((avg_score - 0.50) / 0.20) * 0.07
        else:
            similarity_score = avg_score * 0.16

        return count_score + similarity_score

    def _score_company_presence(self, company_chunks: int) -> float:
        """
        Score company knowledge presence (max 0.25).

        Having barrios_a2i chunks means we can speak authoritatively
        about our own company, pricing, and offerings.

        Scoring:
        - 3+ chunks: Full points (0.25)
        - 2 chunks: Good (0.20)
        - 1 chunk: Partial (0.10)
        - 0 chunks: Base (0.05) - still know static prompt
        """
        if company_chunks >= 3:
            return 0.25
        elif company_chunks == 2:
            return 0.20
        elif company_chunks == 1:
            return 0.10
        else:
            # Base knowledge from system prompt
            return 0.05

    def _score_context_depth(self, history_length: int) -> float:
        """
        Score conversation context depth (max 0.20).

        More history = better understanding of user needs.

        Scoring:
        - 10+ turns: Full (0.20)
        - 5-9 turns: Good (0.15)
        - 2-4 turns: Building (0.10)
        - 1 turn: Minimal (0.05)
        - 0 turns: Cold start (0.02)
        """
        if history_length >= 10:
            return 0.20
        elif history_length >= 5:
            return 0.15
        elif history_length >= 2:
            return 0.10
        elif history_length == 1:
            return 0.05
        else:
            return 0.02

    def _avg_chunk_score(self, chunks: List[Any]) -> float:
        """Get average similarity score from chunks."""
        if not chunks:
            return 0.0

        scores = []
        for chunk in chunks:
            if hasattr(chunk, 'score'):
                scores.append(chunk.score)
            elif isinstance(chunk, dict) and 'score' in chunk:
                scores.append(chunk['score'])

        if not scores:
            return 0.5  # Default if no score attribute

        return sum(scores) / len(scores)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_scorer: Optional[ConfidenceScorer] = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get singleton confidence scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer


def calculate_confidence(
    industry_confidence: float = 0.0,
    rag_chunks: Optional[List[Any]] = None,
    company_chunks_found: int = 0,
    history_length: int = 0,
) -> ConfidenceResult:
    """
    Convenience function to calculate confidence.

    Usage:
        result = calculate_confidence(
            industry_confidence=0.75,
            rag_chunks=chunks,
            company_chunks_found=2,
            history_length=5
        )
        print(f"Confidence: {result.score} ({result.level})")
    """
    scorer = get_confidence_scorer()
    return scorer.calculate(
        industry_confidence=industry_confidence,
        rag_chunks=rag_chunks,
        company_chunks_found=company_chunks_found,
        history_length=history_length,
    )
