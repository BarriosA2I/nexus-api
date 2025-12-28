"""
NEXUS SuperGraph - Main Orchestrator
Wires together all subgraphs with unified state

Production features:
- PostgreSQL checkpointing for state persistence
- OpenTelemetry tracing for observability
- Prometheus metrics for monitoring
- Circuit breaker protection
"""
import logging
import os
import asyncio
from typing import Optional

from .state import NexusState, create_initial_state
from .router import router_node, route_by_intent
from .tracing import traced_node, traced_subgraph, get_tracer
from .metrics import (
    record_intent,
    record_request,
    record_node_latency,
    update_circuit_state,
    ACTIVE_SESSIONS,
)

logger = logging.getLogger("nexus.supergraph")

# PostgreSQL checkpointer singleton
_checkpointer = None
_checkpointer_lock = asyncio.Lock()

# Check if LangGraph is available
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph available - SuperGraph enabled")
except ImportError:
    logger.warning("LangGraph not available - using fallback orchestration")


async def build_nexus_supergraph():
    """
    Build the complete NEXUS SuperGraph with all subgraphs.

    Graph Structure:

        START
          |
          v
       [router]
          |
          +--- video_creation ---> [creative_director_subgraph] ---+
          |                                                        |
          +--- market_research ---> [trinity_subgraph] -----------+
          |                                                        |
          +--- general_chat ---> [rag_subgraph] ------------------+
          |                                                        |
          +--- escalate ---> [escalation_node] -------------------+
                                                                   |
                                                                   v
                                                          [format_response]
                                                                   |
                                                                   v
                                                                  END

    Features:
    - PostgreSQL checkpointing for session state persistence
    - OpenTelemetry tracing on all nodes
    - Prometheus metrics collection
    - Circuit breaker protection on external calls
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("Cannot build SuperGraph - LangGraph not installed")
        return None

    from langgraph.graph import StateGraph, START, END
    from .subgraphs.rag import build_rag_subgraph
    from .subgraphs.creative_director import build_creative_director_subgraph
    from .subgraphs.trinity import build_trinity_subgraph

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD MAIN SUPERGRAPH
    # ─────────────────────────────────────────────────────────────────────────

    supergraph = StateGraph(NexusState)

    # Add nodes
    supergraph.add_node("router", router_node)
    supergraph.add_node("rag_subgraph", build_rag_subgraph())
    supergraph.add_node("creative_director_subgraph", build_creative_director_subgraph())
    supergraph.add_node("trinity_subgraph", build_trinity_subgraph())
    supergraph.add_node("escalation_node", escalation_node)
    supergraph.add_node("format_response", format_response_node)

    # Entry point
    supergraph.add_edge(START, "router")

    # Conditional routing based on intent
    supergraph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "rag_subgraph": "rag_subgraph",
            "creative_director_subgraph": "creative_director_subgraph",
            "trinity_subgraph": "trinity_subgraph",
            "escalation_subgraph": "escalation_node",
        }
    )

    # All subgraphs -> Format Response -> END
    supergraph.add_edge("rag_subgraph", "format_response")
    supergraph.add_edge("creative_director_subgraph", "format_response")
    supergraph.add_edge("trinity_subgraph", "format_response")
    supergraph.add_edge("escalation_node", "format_response")
    supergraph.add_edge("format_response", END)

    # ─────────────────────────────────────────────────────────────────────────
    # COMPILE WITH OPTIONAL CHECKPOINTER
    # ─────────────────────────────────────────────────────────────────────────

    checkpointer = await get_checkpointer()

    compiled = supergraph.compile(
        checkpointer=checkpointer,
        # Interrupt before expensive operations (human-in-the-loop)
        interrupt_before=["render_agent"] if checkpointer else None,
    )

    logger.info(f"NEXUS SuperGraph compiled (checkpointer={'enabled' if checkpointer else 'disabled'})")
    return compiled


async def get_checkpointer():
    """
    Get or create PostgreSQL checkpointer with connection pooling.

    Supports:
    - Async connection pooling via psycopg[pool]
    - Automatic schema creation
    - Graceful degradation if DB unavailable

    Environment Variables:
        DATABASE_URL: PostgreSQL connection string (required)
        CHECKPOINT_POOL_SIZE: Connection pool size (default: 10)
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    async with _checkpointer_lock:
        # Double-check after acquiring lock
        if _checkpointer is not None:
            return _checkpointer

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.info("DATABASE_URL not set - checkpointing disabled")
            return None

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # Create checkpointer with connection pooling
            _checkpointer = AsyncPostgresSaver.from_conn_string(
                database_url,
                # pool_size=int(os.getenv("CHECKPOINT_POOL_SIZE", "10")),
            )

            # Initialize schema (creates tables if not exist)
            await _checkpointer.setup()

            logger.info("PostgreSQL checkpointer initialized with schema")
            return _checkpointer

        except ImportError:
            logger.warning(
                "PostgreSQL checkpointer not available. "
                "Install: pip install langgraph-checkpoint-postgres psycopg[binary,pool]"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
            return None


async def close_checkpointer():
    """
    Gracefully close the checkpointer connection pool.
    Call this during application shutdown.
    """
    global _checkpointer

    if _checkpointer is not None:
        try:
            # Close the connection pool if available
            if hasattr(_checkpointer, 'conn') and _checkpointer.conn:
                await _checkpointer.conn.close()
            logger.info("PostgreSQL checkpointer closed")
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")
        finally:
            _checkpointer = None


@traced_node("escalation")
async def escalation_node(state: NexusState) -> dict:
    """Handle escalation to human support"""
    logger.info(f"Escalating session {state.get('session_id', 'unknown')} to human support")

    return {
        "messages": [{
            "role": "assistant",
            "content": (
                "I understand you'd like to speak with a human. "
                "I'm connecting you with our team now. "
                "In the meantime, you can also reach us at support@barriosa2i.com "
                "or schedule a call at calendly.com/barriosa2i"
            ),
        }],
        "escalated": True,
        "last_successful_node": "escalation_node",
    }


@traced_node("format_response")
async def format_response_node(state: NexusState) -> dict:
    """Format final response with metadata and record metrics"""
    import time

    # Record intent metrics
    intent = state.get("current_intent", "unknown")
    method = "system1_regex" if state.get("intent_confidence", 0) >= 0.85 else "system2_llm"
    confidence = state.get("intent_confidence", 0)
    record_intent(intent, method, confidence)

    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_successful_node": "format_response",
    }


# ============================================================================
# FALLBACK ORCHESTRATOR (when LangGraph not available)
# ============================================================================
class FallbackOrchestrator:
    """Simple orchestrator when LangGraph is not installed"""

    def __init__(self):
        self.sessions = {}

    async def ainvoke(self, state: NexusState, config: dict = None) -> NexusState:
        """Process a message through the fallback pipeline"""
        from .router import router_node, route_by_intent

        # Step 1: Route the message
        router_result = await router_node(state)
        state.update(router_result)

        # Step 2: Get the target subgraph
        target = route_by_intent(state)

        # Step 3: Execute appropriate handler
        if target == "creative_director_subgraph":
            from .subgraphs.creative_director import fallback_creative_director
            result = await fallback_creative_director(state)
        elif target == "trinity_subgraph":
            from .subgraphs.trinity import fallback_trinity
            result = await fallback_trinity(state)
        elif target == "escalation_subgraph":
            result = await escalation_node(state)
        else:  # rag_subgraph
            from .subgraphs.rag import fallback_rag
            result = await fallback_rag(state)

        state.update(result)

        # Step 4: Format response
        format_result = await format_response_node(state)
        state.update(format_result)

        return state

    async def aget_state(self, config: dict) -> Optional[dict]:
        """Get state for a session"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id in self.sessions:
            return type('State', (), {'values': self.sessions[thread_id]})()
        return None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
_supergraph = None
_supergraph_lock = asyncio.Lock()
_fallback = None


async def get_supergraph():
    """
    Get or create the SuperGraph singleton.

    Thread-safe initialization with double-check locking pattern.
    Falls back to FallbackOrchestrator if LangGraph is not available.
    """
    global _supergraph, _fallback

    if LANGGRAPH_AVAILABLE:
        if _supergraph is not None:
            return _supergraph

        async with _supergraph_lock:
            # Double-check after acquiring lock
            if _supergraph is None:
                _supergraph = await build_nexus_supergraph()
            return _supergraph
    else:
        if _fallback is None:
            _fallback = FallbackOrchestrator()
            logger.info("Using FallbackOrchestrator (LangGraph not available)")
        return _fallback


async def shutdown_supergraph():
    """
    Gracefully shutdown the SuperGraph.
    Call during application shutdown.
    """
    global _supergraph, _fallback

    await close_checkpointer()

    _supergraph = None
    _fallback = None
    logger.info("SuperGraph shutdown complete")
