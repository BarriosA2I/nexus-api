"""
NEXUS BRAIN v5.0 APEX - Agent Node
===================================
Fourth node in pipeline: generates response using selected LLM.

Uses the model selected by Thompson router and context from RAG.
Implements streaming response generation with Anthropic API.
"""

import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import anthropic

from ..state import ConversationState, AgentResult, RetrievedChunk

logger = logging.getLogger("nexus.node.agent")

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 0.7


def build_system_prompt(chunks: List[RetrievedChunk]) -> str:
    """
    Build system prompt with RAG context.

    Args:
        chunks: Retrieved knowledge chunks

    Returns:
        System prompt string
    """
    base_prompt = """You are Nexus, the AI assistant for Barrios A2I - an AI automation consultancy.

CORE IDENTITY:
- Company: Barrios A2I (www.barriosa2i.com)
- Tagline: "This is not automation. This is operational intelligence."
- Philosophy: "Your Business. With a Nervous System."
- Founder: Gary Barrios

COMMUNICATION STYLE:
- Professional yet approachable
- Confident but not arrogant
- Technical when needed, simple when possible
- Always helpful and solution-oriented

IMPORTANT:
- ALWAYS know Barrios A2I pricing, services, and offerings
- If asked about pricing, give specific numbers from context
- If asked about services, describe them accurately
- Direct users to cal.com/barriosa2i/discovery for consultations"""

    if not chunks:
        return base_prompt

    # Add context from RAG
    context_section = "\n\nCONTEXT (Use this information to answer accurately):\n"
    for i, chunk in enumerate(chunks[:5], 1):
        context_section += f"\n[{i}] {chunk.content}\n"

    return base_prompt + context_section


def build_messages(
    state: ConversationState,
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Build message history for LLM call.

    Args:
        state: Current conversation state
        system_prompt: System prompt with context

    Returns:
        List of message dicts
    """
    messages = []

    # Add conversation history
    history = state.get("history", [])
    for msg in history[-10:]:  # Keep last 10 messages
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": state["message"],
    })

    return messages


async def generate_response(
    state: ConversationState,
    stream: bool = False,
) -> AsyncGenerator[str, None] | str:
    """
    Generate response using Anthropic API.

    Args:
        state: Current conversation state
        stream: Whether to stream response

    Yields/Returns:
        Response chunks if streaming, full response if not
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    # Get selected model
    model = state.get("selected_model", DEFAULT_MODEL)
    chunks = state.get("context_chunks", [])

    # Build prompts
    system_prompt = build_system_prompt(chunks)
    messages = build_messages(state, system_prompt)

    # Create client
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    if stream:
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        ) as response:
            async for text in response.text_stream:
                yield text
    else:
        response = await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text


async def agent_node(state: ConversationState) -> Dict[str, Any]:
    """
    Agent node: generate response using selected LLM.

    Uses the model selected by Thompson router and context from RAG.
    This node does NOT stream - streaming is handled by publisher_node.

    Args:
        state: Current conversation state

    Returns:
        State updates with agent result and response
    """
    start_time = time.time()

    model = state.get("selected_model", DEFAULT_MODEL)
    chunks = state.get("context_chunks", [])

    logger.info(
        f"Generating response with {model} "
        f"({len(chunks)} context chunks)"
    )

    try:
        # Build prompts
        system_prompt = build_system_prompt(chunks)
        messages = build_messages(state, system_prompt)

        # Create client and generate
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        response = await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        )

        elapsed = (time.time() - start_time) * 1000
        response_text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        logger.info(
            f"Response generated: {len(response_text)} chars, "
            f"{tokens_used} tokens, {elapsed:.1f}ms"
        )

        # Calculate confidence from RAG quality
        avg_score = state.get("rag_result", {})
        if hasattr(avg_score, "avg_score"):
            avg_score = avg_score.avg_score
        else:
            avg_score = 0.5

        company_found = state.get("company_knowledge_found", False)
        confidence = min(0.95, avg_score + (0.2 if company_found else 0))

        agent_result = AgentResult(
            response=response_text,
            model_used=model,
            tokens_used=tokens_used,
            latency_ms=elapsed,
            confidence=confidence,
        )

        return {
            "agent_result": agent_result,
            "response": response_text,
            "response_confidence": confidence,
            "node_timings": {
                **state.get("node_timings", {}),
                "agent": elapsed,
            },
        }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"Agent generation failed: {e}")

        return {
            "agent_result": AgentResult(
                response="I apologize, but I'm having trouble generating a response right now. Please try again.",
                model_used=model,
                latency_ms=elapsed,
            ),
            "response": "I apologize, but I'm having trouble generating a response right now. Please try again.",
            "response_confidence": 0.0,
            "errors": [f"Agent error: {str(e)}"],
            "node_timings": {
                **state.get("node_timings", {}),
                "agent": elapsed,
            },
        }


async def stream_agent_response(
    state: ConversationState,
) -> AsyncGenerator[str, None]:
    """
    Stream response from agent (for SSE).

    Args:
        state: Current conversation state

    Yields:
        Response text chunks
    """
    if not ANTHROPIC_API_KEY:
        yield "I apologize, but the AI service is not configured."
        return

    model = state.get("selected_model", DEFAULT_MODEL)
    chunks = state.get("context_chunks", [])

    system_prompt = build_system_prompt(chunks)
    messages = build_messages(state, system_prompt)

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        ) as response:
            async for text in response.text_stream:
                yield text
    except Exception as e:
        logger.error(f"Stream generation failed: {e}")
        yield "I apologize, but I'm having trouble right now. Please try again."
