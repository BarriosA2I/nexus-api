"""
NEXUS BRAIN v5.0 APEX - Publisher Node
=======================================
Fifth (final) node in pipeline: formats and emits response.

Handles:
- SSE event formatting (meta, delta, final, error)
- Support code generation
- Thompson Sampling feedback preparation
- Final state cleanup
"""

import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict

from ..state import ConversationState
from ..thompson_router import update_model_outcome, ModelTier

logger = logging.getLogger("nexus.node.publisher")


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format SSE event for streaming.

    Args:
        event_type: Event type (meta, delta, final, error)
        data: Event payload

    Returns:
        SSE-formatted string
    """
    import json
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"


async def publisher_node(state: ConversationState) -> Dict[str, Any]:
    """
    Publisher node: finalize response and prepare for emission.

    This is the final node in the pipeline. It:
    1. Generates support code for tracking
    2. Calculates final timing metrics
    3. Prepares Thompson Sampling feedback
    4. Marks state as published

    Args:
        state: Current conversation state

    Returns:
        State updates with publisher metadata
    """
    start_time = time.time()

    # Generate support code
    support_code = f"NX-{uuid.uuid4().hex[:8].upper()}"

    # Calculate total pipeline time
    pipeline_start = state.get("pipeline_start", time.time())
    pipeline_end = time.time()
    total_time = (pipeline_end - pipeline_start) * 1000

    # Get response quality indicators
    response = state.get("response", "")
    confidence = state.get("response_confidence", 0.0)
    errors = state.get("errors", [])

    # Determine Thompson success (for feedback)
    # Success if: response generated, no errors, reasonable confidence
    thompson_success = (
        len(response) > 20 and
        len(errors) == 0 and
        confidence > 0.3
    )

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        f"Publishing response: {len(response)} chars, "
        f"confidence={confidence:.2%}, "
        f"success={thompson_success}, "
        f"total_pipeline={total_time:.1f}ms"
    )

    return {
        "published": True,
        "support_code": support_code,
        "trace_id": state.get("trace_id", f"trace-{uuid.uuid4().hex[:12]}"),
        "pipeline_end": pipeline_end,
        "thompson_success": thompson_success,
        "thompson_reward": confidence if thompson_success else 0.0,
        "node_timings": {
            **state.get("node_timings", {}),
            "publisher": elapsed,
            "total": total_time,
        },
    }


async def stream_response(
    state: ConversationState,
) -> AsyncGenerator[str, None]:
    """
    Stream complete response with SSE events.

    Emits:
    1. meta event (thinking/working status)
    2. delta events (text chunks)
    3. final event (completion with support code)

    Args:
        state: Current conversation state

    Yields:
        SSE-formatted event strings
    """
    from .agent_node import stream_agent_response

    # Emit meta event
    yield format_sse_event("meta", {"state": "thinking"})

    # Stream response chunks
    full_response = ""
    try:
        async for chunk in stream_agent_response(state):
            full_response += chunk
            yield format_sse_event("delta", {"text": chunk})

        # Update Thompson Sampling with outcome
        router_result = state.get("router_result")
        if router_result:
            # Consider success if we got a reasonable response
            success = len(full_response) > 20
            try:
                model_tier = ModelTier(router_result.selected_model.value)
                update_model_outcome(model_tier, success)
            except (ValueError, AttributeError):
                pass  # Ignore if model tier parsing fails

        # Emit final event
        support_code = f"NX-{uuid.uuid4().hex[:8].upper()}"
        yield format_sse_event("final", {"support_code": support_code})

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield format_sse_event("error", {"message": "Connection issue. Try again."})


def apply_thompson_feedback(state: ConversationState) -> None:
    """
    Apply Thompson Sampling feedback after response evaluation.

    Call this after the response quality has been evaluated
    (e.g., user feedback, automated quality check).

    Args:
        state: Final conversation state with thompson_success
    """
    router_result = state.get("router_result")
    thompson_success = state.get("thompson_success")

    if router_result is None or thompson_success is None:
        logger.debug("Skipping Thompson feedback: missing data")
        return

    try:
        model_tier = router_result.selected_model
        update_model_outcome(model_tier, thompson_success)
        logger.info(
            f"Thompson feedback applied: {model_tier.value} "
            f"{'SUCCESS' if thompson_success else 'FAILURE'}"
        )
    except Exception as e:
        logger.error(f"Thompson feedback failed: {e}")
