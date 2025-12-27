"""
NEXUS SuperGraph - Main Orchestrator
Wires together all subgraphs with unified state
"""
import logging
import os
from typing import Optional

from .state import NexusState, create_initial_state
from .router import router_node, route_by_intent

logger = logging.getLogger("nexus.supergraph")

# Check if LangGraph is available
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph available - SuperGraph enabled")
except ImportError:
    logger.warning("LangGraph not available - using fallback orchestration")


def build_nexus_supergraph():
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

    checkpointer = None
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
            logger.info("PostgreSQL checkpointer enabled")
        except ImportError:
            logger.warning("PostgreSQL checkpointer not available")
        except Exception as e:
            logger.warning(f"Failed to initialize checkpointer: {e}")

    compiled = supergraph.compile(
        checkpointer=checkpointer,
        # Interrupt before expensive operations (human-in-the-loop)
        interrupt_before=["render_agent"] if checkpointer else None,
    )

    logger.info("NEXUS SuperGraph compiled successfully")
    return compiled


async def escalation_node(state: NexusState) -> dict:
    """Handle escalation to human support"""
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
        "last_successful_node": "escalation_node",
    }


async def format_response_node(state: NexusState) -> dict:
    """Format final response with metadata"""
    import time

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
_fallback = None


async def get_supergraph():
    """Get or create the SuperGraph singleton"""
    global _supergraph, _fallback

    if LANGGRAPH_AVAILABLE:
        if _supergraph is None:
            _supergraph = build_nexus_supergraph()
        return _supergraph
    else:
        if _fallback is None:
            _fallback = FallbackOrchestrator()
            logger.info("Using FallbackOrchestrator (LangGraph not available)")
        return _fallback
