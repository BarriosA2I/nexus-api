"""
Trinity Subgraph - Market research and competitor analysis
"""
import time
import logging
import os
from typing import Optional

from ..state import NexusState

logger = logging.getLogger("nexus.trinity")

# Check if LangGraph is available
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# COMPETITOR ANALYST AGENT
# ============================================================================
async def competitor_analyst_node(state: NexusState) -> dict:
    """Analyze competitors in the user's industry"""
    start_time = time.perf_counter()

    query = state.get("trinity_query") or state.get("query", state["messages"][-1]["content"])
    industry = state.get("industry", "technology")

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "trinity_competitor_data": {
                "competitors": ["Competitor A", "Competitor B", "Competitor C"],
                "analysis": "Basic competitor analysis - enable ANTHROPIC_API_KEY for detailed insights"
            },
            "last_successful_node": "competitor_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        prompt = f"""Analyze the competitive landscape for this query:

Query: {query}
Industry: {industry}

Provide:
1. Top 3-5 main competitors
2. Their key strengths and weaknesses
3. Market positioning
4. Differentiation opportunities

Format as JSON with structure:
{{
    "competitors": [
        {{
            "name": "...",
            "strengths": ["..."],
            "weaknesses": ["..."],
            "market_share_estimate": "..."
        }}
    ],
    "opportunities": ["..."],
    "threats": ["..."]
}}"""

        response = await client.messages.create(
            model="claude-3-haiku-20240307",  # Fast for research
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        try:
            competitor_data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            competitor_data = {"raw_analysis": response.content[0].text}

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "trinity_competitor_data": competitor_data,
            "last_successful_node": "competitor_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        return {
            "trinity_competitor_data": {"error": str(e)},
            "errors": [{"node": "competitor_analyst", "error": str(e), "recoverable": True}],
        }


# ============================================================================
# SENTIMENT ANALYST AGENT
# ============================================================================
async def sentiment_analyst_node(state: NexusState) -> dict:
    """Analyze market sentiment and brand perception"""
    start_time = time.perf_counter()

    query = state.get("trinity_query") or state.get("query", state["messages"][-1]["content"])

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "trinity_sentiment_data": {
                "overall_sentiment": "neutral",
                "confidence": 0.5,
                "analysis": "Basic sentiment analysis - enable ANTHROPIC_API_KEY for detailed insights"
            },
            "last_successful_node": "sentiment_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        prompt = f"""Analyze market sentiment for this query:

Query: {query}

Provide:
1. Overall market sentiment (positive/neutral/negative)
2. Key sentiment drivers
3. Brand perception insights
4. Customer pain points

Format as JSON with structure:
{{
    "overall_sentiment": "positive|neutral|negative",
    "sentiment_score": 0.0-1.0,
    "drivers": ["..."],
    "pain_points": ["..."],
    "opportunities": ["..."]
}}"""

        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        try:
            sentiment_data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            sentiment_data = {"raw_analysis": response.content[0].text}

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "trinity_sentiment_data": sentiment_data,
            "last_successful_node": "sentiment_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "trinity_sentiment_data": {"error": str(e)},
            "errors": [{"node": "sentiment_analyst", "error": str(e), "recoverable": True}],
        }


# ============================================================================
# TRENDS ANALYST AGENT
# ============================================================================
async def trends_analyst_node(state: NexusState) -> dict:
    """Analyze market trends and predictions"""
    start_time = time.perf_counter()

    query = state.get("trinity_query") or state.get("query", state["messages"][-1]["content"])
    industry = state.get("industry", "technology")

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "trinity_trends_data": {
                "trends": ["Trend 1", "Trend 2", "Trend 3"],
                "analysis": "Basic trends analysis - enable ANTHROPIC_API_KEY for detailed insights"
            },
            "last_successful_node": "trends_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        prompt = f"""Analyze market trends for this query:

Query: {query}
Industry: {industry}

Provide:
1. Current market trends
2. Emerging technologies/approaches
3. Market growth predictions
4. Strategic recommendations

Format as JSON with structure:
{{
    "current_trends": ["..."],
    "emerging_trends": ["..."],
    "growth_predictions": {{
        "short_term": "...",
        "long_term": "..."
    }},
    "recommendations": ["..."]
}}"""

        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        try:
            trends_data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            trends_data = {"raw_analysis": response.content[0].text}

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "trinity_trends_data": trends_data,
            "last_successful_node": "trends_analyst",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Trends analysis failed: {e}")
        return {
            "trinity_trends_data": {"error": str(e)},
            "errors": [{"node": "trends_analyst", "error": str(e), "recoverable": True}],
        }


# ============================================================================
# SYNTHESIS AGENT - Combine all research
# ============================================================================
async def synthesis_node(state: NexusState) -> dict:
    """Synthesize all research into actionable insights"""
    start_time = time.perf_counter()

    competitor_data = state.get("trinity_competitor_data", {})
    sentiment_data = state.get("trinity_sentiment_data", {})
    trends_data = state.get("trinity_trends_data", {})

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        synthesis = f"""## Market Research Summary

### Competitive Landscape
{competitor_data.get('analysis', 'No competitor data available')}

### Market Sentiment
{sentiment_data.get('analysis', 'No sentiment data available')}

### Trends & Predictions
{trends_data.get('analysis', 'No trends data available')}

---
*Enable ANTHROPIC_API_KEY for detailed AI-powered insights*"""

        return {
            "trinity_synthesis": synthesis,
            "messages": [{"role": "assistant", "content": synthesis}],
            "last_successful_node": "synthesis",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        import json
        client = AsyncAnthropic()

        prompt = f"""Synthesize this market research into a clear, actionable executive summary:

COMPETITOR ANALYSIS:
{json.dumps(competitor_data, indent=2)}

SENTIMENT ANALYSIS:
{json.dumps(sentiment_data, indent=2)}

TRENDS ANALYSIS:
{json.dumps(trends_data, indent=2)}

Create a concise executive summary with:
1. Key findings (3-5 bullet points)
2. Strategic recommendations (3-5 bullet points)
3. Immediate action items (2-3 bullet points)
4. Risk factors to consider

Use markdown formatting for readability."""

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        synthesis = response.content[0].text
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "trinity_synthesis": synthesis,
            "messages": [{"role": "assistant", "content": synthesis}],
            "last_successful_node": "synthesis",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {
            "trinity_synthesis": f"Error generating synthesis: {e}",
            "messages": [{"role": "assistant", "content": "I encountered an error generating the research synthesis. Please try again."}],
            "errors": [{"node": "synthesis", "error": str(e), "recoverable": True}],
        }


# ============================================================================
# BUILD SUBGRAPH
# ============================================================================
def build_trinity_subgraph():
    """Build the Trinity subgraph"""

    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available for Trinity subgraph")
        return None

    from langgraph.graph import StateGraph, START, END

    trinity = StateGraph(NexusState)

    # Add all agent nodes
    trinity.add_node("competitor_analyst", competitor_analyst_node)
    trinity.add_node("sentiment_analyst", sentiment_analyst_node)
    trinity.add_node("trends_analyst", trends_analyst_node)
    trinity.add_node("synthesis", synthesis_node)

    # Parallel execution of analysts, then synthesis
    trinity.add_edge(START, "competitor_analyst")
    trinity.add_edge(START, "sentiment_analyst")
    trinity.add_edge(START, "trends_analyst")

    trinity.add_edge("competitor_analyst", "synthesis")
    trinity.add_edge("sentiment_analyst", "synthesis")
    trinity.add_edge("trends_analyst", "synthesis")

    trinity.add_edge("synthesis", END)

    return trinity.compile()


async def fallback_trinity(state: NexusState) -> dict:
    """Fallback Trinity pipeline when LangGraph not available"""
    result = {}

    # Run all analysts sequentially (parallel not available in fallback)
    competitor_result = await competitor_analyst_node(state)
    result.update(competitor_result)
    state.update(competitor_result)

    sentiment_result = await sentiment_analyst_node(state)
    result.update(sentiment_result)
    state.update(sentiment_result)

    trends_result = await trends_analyst_node(state)
    result.update(trends_result)
    state.update(trends_result)

    # Synthesize
    synthesis_result = await synthesis_node(state)
    result.update(synthesis_result)

    return result
