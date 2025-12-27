"""
NEXUS SuperGraph - Router Node
Classifies intent and routes to appropriate subgraph
"""
import re
import time
import logging
import os
from typing import Literal, Tuple

from .state import NexusState

logger = logging.getLogger("nexus.router")

# ============================================================================
# INTENT PATTERNS (System 1 - Fast Path)
# ============================================================================
INTENT_PATTERNS = {
    "video_creation": [
        r"\b(video|commercial|ad|advertisement|promo|trailer|film|clip)\b",
        r"\b(create|make|produce|generate|build)\b.{0,20}\b(video|ad|commercial)\b",
        r"\bragnarok\b",
        r"\b(neural|ai).{0,10}(ad|video|commercial)\b",
    ],
    "market_research": [
        r"\b(competitor|competition|market|research|analyze|analysis)\b",
        r"\b(trends?|sentiment|industry)\b.{0,20}\b(research|analysis|report)\b",
        r"\btrinity\b",
        r"\bwhat.{0,20}(competitors?|market)\b",
    ],
    "intake": [
        r"\b(consultation|consult|discuss|talk about|help me with)\b",
        r"\b(project|business|company|startup)\b.{0,20}\b(help|advice|guidance)\b",
        r"\b(what|how).{0,20}(services?|offerings?|do you do)\b",
    ],
    "escalate": [
        r"\b(human|person|real person|speak to someone|manager)\b",
        r"\b(not helpful|frustrated|angry|upset)\b",
    ],
}

# Confidence thresholds
FAST_PATH_CONFIDENCE = 0.85  # Use System 1 if above this
ESCALATION_THRESHOLD = 0.5   # Escalate to System 2 if below this


async def router_node(state: NexusState) -> dict:
    """
    Router Node - Entry point for all user messages.
    Uses dual-process routing:
      - System 1 (Fast): Regex pattern matching
      - System 2 (Slow): LLM classification for ambiguous cases
    """
    start_time = time.perf_counter()

    last_message = state["messages"][-1]["content"].lower()

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM 1: Fast Pattern Matching
    # ─────────────────────────────────────────────────────────────────────────
    intent_scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, last_message, re.IGNORECASE))
        if matches > 0:
            # Score based on number of pattern matches
            intent_scores[intent] = min(0.95, 0.6 + (matches * 0.15))

    # Check if we have a confident fast-path result
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        best_confidence = intent_scores[best_intent]

        if best_confidence >= FAST_PATH_CONFIDENCE:
            logger.info(f"Router: System 1 fast path -> {best_intent} ({best_confidence:.2f})")

            latency_ms = (time.perf_counter() - start_time) * 1000

            return {
                "current_intent": best_intent,
                "previous_intent": state.get("current_intent"),
                "intent_confidence": best_confidence,
                "intent_history": [{
                    "intent": best_intent,
                    "confidence": best_confidence,
                    "method": "system1_regex",
                    "message_preview": last_message[:50],
                }],
                "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM 2: LLM Classification (for ambiguous cases)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Router: System 2 LLM classification")
    intent, confidence = await _llm_classify_intent(last_message, state)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "current_intent": intent,
        "previous_intent": state.get("current_intent"),
        "intent_confidence": confidence,
        "intent_history": [{
            "intent": intent,
            "confidence": confidence,
            "method": "system2_llm",
            "message_preview": last_message[:50],
        }],
        "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        "model_calls": [{
            "node": "router",
            "model": "claude-3-haiku-20240307",
            "latency_ms": latency_ms,
            "purpose": "intent_classification",
        }],
    }


async def _llm_classify_intent(message: str, state: NexusState) -> Tuple[str, float]:
    """Use Claude Haiku for fast, cheap intent classification"""

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set, defaulting to general_chat")
        return "general_chat", 0.6

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        # Include conversation context for better classification
        recent_messages = state.get("messages", [])[-5:]  # Last 5 messages
        context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages])

        prompt = f"""Classify the user's intent based on their message and conversation context.

CONVERSATION CONTEXT:
{context}

LATEST MESSAGE: {message}

POSSIBLE INTENTS:
1. general_chat - General questions, conversation, information requests
2. video_creation - User wants to create a video, commercial, ad, or promotional content
3. market_research - User wants competitor analysis, market trends, or industry research
4. intake - User wants consultation, help with a project, or general business guidance
5. escalate - User is frustrated, wants human help, or the request is unclear

Respond with ONLY the intent name and confidence (0.0-1.0) in this format:
INTENT: <intent_name>
CONFIDENCE: <0.0-1.0>"""

        response = await client.messages.create(
            model="claude-3-haiku-20240307",  # Fast & cheap for classification
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text

        # Parse response
        intent_match = re.search(r"INTENT:\s*(\w+)", text)
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)

        intent = intent_match.group(1) if intent_match else "general_chat"
        confidence = float(conf_match.group(1)) if conf_match else 0.7

        # Validate intent
        valid_intents = ["general_chat", "video_creation", "market_research", "intake", "escalate"]
        if intent not in valid_intents:
            intent = "general_chat"
            confidence = 0.5

        return intent, confidence

    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        return "general_chat", 0.5


def route_by_intent(state: NexusState) -> str:
    """
    Conditional edge function - determines which subgraph to route to.
    Used by supergraph.add_conditional_edges()
    """
    intent = state.get("current_intent", "general_chat")
    confidence = state.get("intent_confidence", 0.0)

    # Low confidence -> default to general chat
    if confidence < ESCALATION_THRESHOLD:
        logger.warning(f"Low confidence ({confidence:.2f}) - defaulting to general_chat")
        return "rag_subgraph"

    routing_map = {
        "video_creation": "creative_director_subgraph",
        "market_research": "trinity_subgraph",
        "intake": "rag_subgraph",  # Use RAG with intake mode
        "escalate": "escalation_subgraph",
        "general_chat": "rag_subgraph",
    }

    return routing_map.get(intent, "rag_subgraph")
