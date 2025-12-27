"""
NEXUS BRAIN v5.0 APEX - LangGraph Orchestrator
===============================================
StateGraph-based pipeline orchestration.

Pipeline flow:
[START] -> classifier -> router -> rag -> agent -> publisher -> [END]

Features:
- Node-based execution with state management
- Conditional routing based on complexity
- PostgreSQL checkpointing for persistence
- Streaming response generation
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from langgraph.graph import StateGraph, END

from .state import ConversationState, create_initial_state, state_to_dict
from .nodes import (
    classifier_node,
    router_node,
    rag_node,
    agent_node,
    publisher_node,
)
from .nodes.publisher_node import format_sse_event, stream_response

logger = logging.getLogger("nexus.orchestrator")


# =============================================================================
# GRAPH DEFINITION
# =============================================================================

def create_graph() -> StateGraph:
    """
    Create LangGraph StateGraph for Nexus Brain pipeline.

    Pipeline:
    classifier -> router -> rag -> agent -> publisher

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create graph with state schema
    graph = StateGraph(ConversationState)

    # Add nodes
    graph.add_node("classifier", classifier_node)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("publisher", publisher_node)

    # Define edges (linear pipeline)
    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "router")
    graph.add_edge("router", "rag")
    graph.add_edge("rag", "agent")
    graph.add_edge("agent", "publisher")
    graph.add_edge("publisher", END)

    return graph


def create_compiled_graph():
    """
    Create and compile the graph for execution.

    Returns:
        Compiled graph ready for invoke/stream
    """
    graph = create_graph()
    return graph.compile()


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class NexusBrainOrchestrator:
    """
    High-level orchestrator for Nexus Brain v5.0 APEX.

    Provides:
    - Single invoke for non-streaming responses
    - Streaming invoke for SSE responses
    - Session management
    - State persistence (optional)
    """

    def __init__(
        self,
        checkpointer=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            checkpointer: Optional LangGraph checkpointer for persistence
            config: Optional configuration overrides
        """
        self.graph = create_graph()

        if checkpointer:
            self.compiled = self.graph.compile(checkpointer=checkpointer)
        else:
            self.compiled = self.graph.compile()

        self.config = config or {}

        logger.info("NexusBrainOrchestrator initialized")

    async def invoke(
        self,
        session_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ConversationState:
        """
        Invoke pipeline and return final state.

        For non-streaming responses. Use stream() for SSE.

        Args:
            session_id: Session identifier
            message: User's input message
            history: Optional conversation history

        Returns:
            Final ConversationState after pipeline execution
        """
        # Create initial state
        state = create_initial_state(session_id, message, history)

        logger.info(
            f"Pipeline invoke: session={session_id}, "
            f"message={message[:50]}..."
        )

        # Run pipeline
        try:
            final_state = await self.compiled.ainvoke(state)

            logger.info(
                f"Pipeline complete: "
                f"response_len={len(final_state.get('response', ''))}, "
                f"confidence={final_state.get('response_confidence', 0):.2%}"
            )

            return final_state

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            state["errors"] = [str(e)]
            state["response"] = "I apologize, but I encountered an error. Please try again."
            return state

    async def stream(
        self,
        session_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response via SSE events.

        Runs classifier, router, and rag nodes first, then streams
        the agent response with delta events.

        Args:
            session_id: Session identifier
            message: User's input message
            history: Optional conversation history

        Yields:
            SSE-formatted event strings
        """
        # Create initial state
        state = create_initial_state(session_id, message, history)

        logger.info(
            f"Pipeline stream: session={session_id}, "
            f"message={message[:50]}..."
        )

        try:
            # Emit initial meta event
            yield format_sse_event("meta", {"state": "thinking"})

            # Run pre-agent nodes
            state = await classifier_node(state)
            state = {**state, **(await router_node(state))}
            state = {**state, **(await rag_node(state))}

            # Update status
            yield format_sse_event("meta", {"state": "working"})

            # Stream response
            async for event in stream_response(state):
                yield event

        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield format_sse_event("error", {
                "message": "Connection issue. Please try again."
            })

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        from .thompson_router import get_thompson_router

        router = get_thompson_router()
        return {
            "thompson_stats": router.get_stats(),
            "config": self.config,
        }


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_orchestrator: Optional[NexusBrainOrchestrator] = None


def get_orchestrator() -> NexusBrainOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_orchestrator()
    return _orchestrator


def create_orchestrator(
    use_postgres_checkpointer: bool = False,
) -> NexusBrainOrchestrator:
    """
    Factory function to create orchestrator.

    Args:
        use_postgres_checkpointer: Enable PostgreSQL state persistence

    Returns:
        Configured NexusBrainOrchestrator
    """
    checkpointer = None

    if use_postgres_checkpointer:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            db_url = os.getenv("DATABASE_URL")
            if db_url:
                # Note: In production, use connection pool
                checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
                logger.info("PostgreSQL checkpointer enabled")
            else:
                logger.warning("DATABASE_URL not set, checkpointer disabled")
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres not installed, "
                "checkpointer disabled"
            )

    return NexusBrainOrchestrator(checkpointer=checkpointer)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def chat(
    session_id: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    stream: bool = False,
) -> ConversationState | AsyncGenerator[str, None]:
    """
    Main entry point for Nexus Brain chat.

    Args:
        session_id: Session identifier
        message: User's input message
        history: Optional conversation history
        stream: Whether to stream response

    Returns:
        Final state if not streaming, SSE generator if streaming
    """
    orchestrator = get_orchestrator()

    if stream:
        return orchestrator.stream(session_id, message, history)
    else:
        return await orchestrator.invoke(session_id, message, history)
