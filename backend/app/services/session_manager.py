"""
SESSION MANAGER
===============
Redis-backed conversation history management.
Fixes CRITICAL bug: Bot was passing empty [] for history every request.

Features:
- Sliding window (last 20 messages)
- 1 hour TTL per session
- In-memory fallback for dev safety
"""

import os
import json
import logging
from typing import List, Dict, Optional

import redis

logger = logging.getLogger("nexus.session")

# Configuration
REDIS_URL = os.getenv("REDIS_URL")
SESSION_TTL = 3600  # 1 hour
MAX_HISTORY_LENGTH = 20  # 10 conversation turns

# Initialize Redis client
_redis_client: Optional[redis.Redis] = None
_local_fallback: Dict[str, List[Dict]] = {}  # Dev fallback

if REDIS_URL:
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()  # Test connection
        logger.info("Session Manager: Connected to Redis")
    except Exception as e:
        logger.error(f"Session Manager: Redis connection failed: {e}")
        _redis_client = None
else:
    logger.warning("Session Manager: REDIS_URL not set - using in-memory fallback (resets on restart)")


async def get_history(session_id: str) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for a session.

    Returns:
        List of message dicts: [{"role": "user", "content": "..."}, ...]
    """
    if not session_id:
        return []

    key = f"nexus:session:{session_id}"

    # Try Redis first
    if _redis_client:
        try:
            data = _redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis read error: {e}")

    # Fallback to local memory
    return _local_fallback.get(session_id, [])


async def save_turn(session_id: str, user_message: str, assistant_message: str):
    """
    Append a new conversation turn to history.
    Applies sliding window truncation.

    Args:
        session_id: Unique session identifier
        user_message: The user's message
        assistant_message: Nexus's response
    """
    if not session_id:
        return

    # Get current history
    history = await get_history(session_id)

    # Append new messages
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})

    # Sliding window - keep last N messages
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    key = f"nexus:session:{session_id}"

    # Save to Redis
    if _redis_client:
        try:
            _redis_client.setex(key, SESSION_TTL, json.dumps(history))
        except Exception as e:
            logger.error(f"Redis write error: {e}")

    # Always sync to local fallback
    _local_fallback[session_id] = history

    # Memory management - cap fallback dict size
    if len(_local_fallback) > 1000:
        # Remove oldest entries
        oldest_key = next(iter(_local_fallback))
        del _local_fallback[oldest_key]


async def clear_session(session_id: str):
    """Clear a session's history (for testing or user request)."""
    if not session_id:
        return

    key = f"nexus:session:{session_id}"

    if _redis_client:
        try:
            _redis_client.delete(key)
        except Exception:
            pass

    if session_id in _local_fallback:
        del _local_fallback[session_id]
