"""
RAG Subgraph - General chat with retrieval-augmented generation

Production features:
- Circuit breaker protection for Qdrant and Anthropic
- OpenTelemetry tracing for full observability
- Prometheus metrics for monitoring
"""
import time
import logging
import os
from typing import Optional, Dict, Any

from ..state import NexusState
from ..circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    with_circuit_breaker,
)
from ..tracing import traced_node, traced_external_call, traced_llm_call
from ..metrics import (
    record_rag_retrieval,
    record_llm_call,
    record_node_latency,
    metered_node,
    CONTEXT_LENGTH,
)

logger = logging.getLogger("nexus.rag")

# Check if LangGraph is available
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    pass

# Get circuit breakers
_registry = CircuitBreakerRegistry.get_instance()
_anthropic_breaker = _registry.register("anthropic", threshold=5, reset_timeout=30)
_qdrant_breaker = _registry.register("qdrant", threshold=3, reset_timeout=60)


@traced_node("retrieve")
@metered_node("retrieve")
async def retrieve_node(state: NexusState) -> dict:
    """Retrieve relevant documents from vector store with circuit breaker"""
    start_time = time.perf_counter()

    query = state.get("query", state["messages"][-1]["content"])

    # Check circuit breaker
    if not await _qdrant_breaker.can_execute():
        logger.warning("Qdrant circuit OPEN - skipping retrieval")
        record_rag_retrieval("qdrant", "circuit_open")
        return {
            "retrieved_docs": [],
            "errors": [{"node": "retrieve", "error": "Vector store unavailable", "recoverable": True}],
            "last_successful_node": state.get("last_successful_node", "start"),
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    # Try to use RAG service if available
    docs = []
    try:
        from ...services.rag_local import get_rag_service
        rag = get_rag_service()
        if rag:
            results = rag.search(query, top_k=5)
            docs = [{"content": r["content"], "source": r.get("source", "unknown"), "score": r.get("score", 0)} for r in results]

        _qdrant_breaker.record_success()
        record_rag_retrieval("qdrant", "success", context_length=sum(len(d.get("content", "")) for d in docs))

    except Exception as e:
        _qdrant_breaker.record_failure()
        record_rag_retrieval("qdrant", "error")
        logger.warning(f"RAG retrieval failed: {e}")

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "retrieved_docs": docs,
        "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        "last_successful_node": "retrieve",
    }


@traced_node("compress")
@metered_node("compress")
async def compress_node(state: NexusState) -> dict:
    """Compress retrieved context using LLMLingua-style compression"""
    start_time = time.perf_counter()

    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"compressed_context": "", "last_successful_node": "compress"}

    # Concatenate documents
    full_context = "\n\n---\n\n".join([
        f"[Source: {doc.get('source', 'unknown')}]\n{doc.get('content', '')}"
        for doc in docs
    ])

    # Simple compression: truncate to max tokens
    max_chars = 8000
    if len(full_context) > max_chars:
        compressed = full_context[:max_chars] + "..."
    else:
        compressed = full_context

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Record context length metric
    CONTEXT_LENGTH.observe(len(compressed))

    return {
        "compressed_context": compressed,
        "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        "last_successful_node": "compress",
    }


@traced_node("generate")
@metered_node("generate")
async def generate_node(state: NexusState) -> dict:
    """Generate response using Claude with circuit breaker"""
    start_time = time.perf_counter()

    # Check circuit breaker FIRST
    if not await _anthropic_breaker.can_execute():
        logger.warning("Anthropic circuit OPEN - using fallback response")
        record_llm_call("claude-sonnet-4-20250514", "circuit_open")
        return {
            "messages": [{
                "role": "assistant",
                "content": "I'm experiencing high demand right now. Please try again in a moment, or reach out to support@barriosa2i.com for immediate assistance."
            }],
            "errors": [{"node": "generate", "error": "LLM service unavailable", "recoverable": True}],
            "last_successful_node": "compress",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    query = state.get("query", state["messages"][-1]["content"])
    context = state.get("compressed_context", "")
    industry = state.get("industry", "general")

    # Build system prompt
    system_prompt = f"""You are NEXUS, the AI assistant for Barrios A2I (Alienation 2 Innovation).

You help businesses with:
- AI automation solutions
- Video commercial creation (RAGNAROK)
- Market research and competitor analysis (Trinity)
- Custom AI agent development

Current user industry: {industry or 'Not yet determined'}

CONTEXT FROM KNOWLEDGE BASE:
{context if context else 'No specific context retrieved.'}

Be helpful, professional, and guide users toward our services when relevant.
Keep responses concise but informative."""

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set, using fallback response")
        return {
            "messages": [{"role": "assistant", "content": "I'm NEXUS, your AI assistant. How can I help you today?"}],
            "rag_confidence": 0.5,
            "last_successful_node": "generate",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        # Build conversation messages
        messages = []
        for msg in state.get("messages", [])[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )

        _anthropic_breaker.record_success()

        assistant_message = response.content[0].text
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)  # Sonnet pricing

        return {
            "messages": [{"role": "assistant", "content": assistant_message}],
            "rag_confidence": 0.85,
            "total_cost_usd": state.get("total_cost_usd", 0) + cost,
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
            "token_usage": {
                "input": state.get("token_usage", {}).get("input", 0) + input_tokens,
                "output": state.get("token_usage", {}).get("output", 0) + output_tokens,
            },
            "model_calls": [{
                "node": "generate",
                "model": "claude-sonnet-4-20250514",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost,
            }],
            "last_successful_node": "generate",
        }

    except Exception as e:
        _anthropic_breaker.record_failure()
        logger.error(f"Generation failed: {e}")

        return {
            "messages": [{"role": "assistant", "content": "I apologize, but I encountered an error. Please try again."}],
            "errors": [{"node": "generate", "error": str(e), "recoverable": True}],
            "last_successful_node": "compress",
        }


def build_rag_subgraph():
    """Build the RAG subgraph - returns compiled graph or None"""

    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available for RAG subgraph")
        return None

    from langgraph.graph import StateGraph, START, END

    rag = StateGraph(NexusState)

    rag.add_node("retrieve", retrieve_node)
    rag.add_node("compress", compress_node)
    rag.add_node("generate", generate_node)

    rag.add_edge(START, "retrieve")
    rag.add_edge("retrieve", "compress")
    rag.add_edge("compress", "generate")
    rag.add_edge("generate", END)

    return rag.compile()


async def fallback_rag(state: NexusState) -> dict:
    """Fallback RAG pipeline when LangGraph not available"""
    result = {}

    # Step 1: Retrieve
    retrieve_result = await retrieve_node(state)
    result.update(retrieve_result)
    state.update(retrieve_result)

    # Step 2: Compress
    compress_result = await compress_node(state)
    result.update(compress_result)
    state.update(compress_result)

    # Step 3: Generate
    generate_result = await generate_node(state)
    result.update(generate_result)

    return result
