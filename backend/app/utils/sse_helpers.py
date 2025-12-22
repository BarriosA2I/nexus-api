"""
Sales-Safe SSE Protocol Helpers
P0-D: Now includes optional confidence metadata for frontend display.

Exposed to frontend:
- Confidence level (low/medium/high) for user feedback
- Support code (obfuscated trace_id)
- Next action suggestion

NOT exposed:
- Raw confidence score (0-1)
- Internal trace_id format
- Model names
- Source citations
"""
import json
from typing import Any, Dict, Optional


def make_support_code(trace_id: str) -> str:
    """
    Convert internal trace_id to customer-safe support code.
    "nxs_1766188864_53483d2f" -> "NX-53483D"
    """
    if not trace_id:
        return "NX-GENERIC"
    parts = trace_id.split("_")
    tail = parts[-1] if len(parts) > 1 else trace_id
    return f"NX-{tail[:6].upper()}"


def sse_pack(event_type: str, data: dict) -> str:
    """
    Pack event into SSE format.
    Valid event_type: meta, delta, final, error
    """
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# Pre-built events for common cases
def sse_thinking(message: str = "One sec—getting context.") -> str:
    """Send thinking status"""
    return sse_pack("meta", {"state": "thinking", "message": message})


def sse_working(message: str = "Still with you...") -> str:
    """Send working status"""
    return sse_pack("meta", {"state": "working", "message": message})


def sse_delta(text: str) -> str:
    """Send text chunk"""
    return sse_pack("delta", {"text": text})


def sse_final(
    trace_id: str,
    next_action: str = "question",
    confidence: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Send final event with support code and optional confidence metadata.

    Args:
        trace_id: Internal trace ID (will be obfuscated)
        next_action: "question" | "intake" | "booking" | "offer"
        confidence: Optional confidence result dict with:
            - level: "low" | "medium" | "high"
            - score: 0.0-1.0 (for monitoring, not display)
            - industry: Detected industry name

    P0-D: Confidence level is exposed for frontend to show response quality indicator.
    """
    payload = {
        "support_code": make_support_code(trace_id),
        "next": next_action,
    }

    # P0-D: Add confidence metadata if provided
    if confidence:
        payload["confidence"] = {
            "level": confidence.get("level", "medium"),
            # Include industry for context (sales-safe)
            "industry": confidence.get("industry"),
        }
        # Include score only if explicitly enabled (for debugging/monitoring)
        if confidence.get("include_score", False):
            payload["confidence"]["score"] = confidence.get("score", 0.5)

    return sse_pack("final", payload)


def sse_error(message: str = "I'm having trouble loading right now. Try again in a moment.") -> str:
    """Send error event"""
    return sse_pack("error", {"message": message})
