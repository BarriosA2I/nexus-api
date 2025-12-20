"""
Sales-Safe SSE Protocol Helpers
Never expose: confidence, trace_id, sources, internal step names, model names
"""
import json
from typing import Optional


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


def sse_final(trace_id: str, next_action: str = "question") -> str:
    """
    Send final event with support code.
    next_action: "question" | "intake" | "booking" | "offer"
    """
    return sse_pack("final", {
        "support_code": make_support_code(trace_id),
        "next": next_action
    })


def sse_error(message: str = "I'm having trouble loading right now. Try again in a moment.") -> str:
    """Send error event"""
    return sse_pack("error", {"message": message})
