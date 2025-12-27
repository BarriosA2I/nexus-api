"""
NEXUS BRAIN v5.0 APEX - Classifier Node
========================================
First node in pipeline: determines complexity and detects industry.

Outputs:
- complexity: ClassifierResult with System 1/2 level
- detected_industry: Industry classification for RAG filtering
- industry_confidence: Confidence score for industry detection
"""

import logging
import time
from typing import Any, Dict

from ..state import ConversationState, ClassifierResult, ComplexityLevel
from ..complexity_classifier import classify_complexity

logger = logging.getLogger("nexus.node.classifier")


# Industry detection patterns (simplified from SemanticRouter)
INDUSTRY_PATTERNS: Dict[str, list] = {
    "barrios_a2i": [
        r"\b(barrios|a2i|nexus|ragnarok|trinity)\b",
        r"\b(marketing\s+overlord|neural\s+ad\s+forge|cinesite|total\s+command)\b",
        r"\b(your|this)\s+(company|business|service|product)s?\b",
    ],
    "technology": [
        r"\b(software|app|application|platform|saas)\b",
        r"\b(api|sdk|developer|programming|code)\b",
        r"\b(cloud|server|database|infrastructure)\b",
    ],
    "marketing": [
        r"\b(marketing|advertising|campaign|brand|content)\b",
        r"\b(social\s+media|seo|ppc|email\s+marketing)\b",
        r"\b(lead\s+generation|conversion|funnel)\b",
    ],
    "ecommerce": [
        r"\b(ecommerce|e-commerce|online\s+store|shop)\b",
        r"\b(product|inventory|checkout|cart|payment)\b",
        r"\b(shipping|fulfillment|order)\b",
    ],
    "healthcare": [
        r"\b(health|medical|patient|clinical|healthcare)\b",
        r"\b(hospital|clinic|doctor|nurse|treatment)\b",
        r"\b(insurance|medicare|medicaid)\b",
    ],
    "finance": [
        r"\b(finance|financial|banking|investment)\b",
        r"\b(loan|credit|mortgage|trading)\b",
        r"\b(insurance|risk|compliance)\b",
    ],
}


def detect_industry(message: str) -> tuple[str, float]:
    """
    Detect industry from message using pattern matching.

    Args:
        message: User's input message

    Returns:
        Tuple of (industry, confidence)
    """
    import re

    message_lower = message.lower()
    scores: Dict[str, int] = {}

    for industry, patterns in INDUSTRY_PATTERNS.items():
        match_count = 0
        for pattern in patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                match_count += 1
        scores[industry] = match_count

    # Find best match
    if not scores or max(scores.values()) == 0:
        return "general", 0.3

    best_industry = max(scores, key=scores.get)
    max_score = scores[best_industry]

    # Calculate confidence based on match density
    confidence = min(0.95, 0.4 + (max_score * 0.15))

    return best_industry, confidence


async def classifier_node(state: ConversationState) -> Dict[str, Any]:
    """
    Classifier node: analyze message complexity and industry.

    This is the first node in the pipeline. It determines:
    1. Whether this is a System 1 (fast) or System 2 (slow) query
    2. What industry context to use for RAG retrieval

    Args:
        state: Current conversation state

    Returns:
        State updates with complexity and industry info
    """
    start_time = time.time()
    message = state["message"]

    logger.info(f"Classifying message: {message[:50]}...")

    # 1. Classify complexity
    complexity_result = classify_complexity(message)

    # 2. Detect industry
    industry, industry_conf = detect_industry(message)

    # Always include barrios_a2i for dual-scope RAG
    if industry != "barrios_a2i":
        logger.debug(f"Dual-scope: {industry} + barrios_a2i")

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        f"Classification complete: "
        f"complexity={complexity_result.level.value} "
        f"(conf={complexity_result.confidence:.2%}), "
        f"industry={industry} (conf={industry_conf:.2%}), "
        f"elapsed={elapsed:.1f}ms"
    )

    return {
        "complexity": complexity_result,
        "detected_industry": industry,
        "industry_confidence": industry_conf,
        "node_timings": {
            **state.get("node_timings", {}),
            "classifier": elapsed,
        },
    }
