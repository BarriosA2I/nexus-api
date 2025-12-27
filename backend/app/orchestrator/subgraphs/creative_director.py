"""
Creative Director Subgraph - 6-agent video creation pipeline
"""
import time
import logging
import os
import json
from typing import Optional

from ..state import NexusState

logger = logging.getLogger("nexus.creative_director")

# Check if LangGraph is available
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# INTAKE AGENT - Conversational brief gathering
# ============================================================================
async def intake_agent_node(state: NexusState) -> dict:
    """Gather video requirements through conversation"""
    start_time = time.perf_counter()

    # Check if intake is complete
    required_fields = ["business_name", "product_service", "target_audience", "video_goal", "tone"]
    answers = state.get("cd_intake_answers", {})
    missing = [f for f in required_fields if f not in answers]

    if not missing:
        # Intake complete - move to brief generation
        return {
            "cd_phase": "brief",
            "cd_intake_complete": True,
            "cd_intake_missing_info": [],
            "last_successful_node": "intake_agent",
        }

    # Generate next question
    next_field = missing[0]
    questions = {
        "business_name": "What's the name of your business or brand?",
        "product_service": "What product or service would you like to promote in this video?",
        "target_audience": "Who is your target audience? (demographics, interests, pain points)",
        "video_goal": "What's the main goal of this video? (brand awareness, lead generation, sales, education)",
        "tone": "What tone should the video have? (professional, playful, emotional, energetic, luxurious)",
    }

    question = questions.get(next_field, f"Please tell me about your {next_field}")

    # Check if user already provided an answer in their message
    last_message = state["messages"][-1]["content"]

    # Simple extraction - in production, use LLM for better extraction
    if len(last_message) > 10:  # User provided substantial input
        # Store the answer and ask next question
        return {
            "cd_intake_answers": {**answers, next_field: last_message},
            "cd_intake_questions_asked": state.get("cd_intake_questions_asked", []) + [next_field],
            "cd_intake_missing_info": missing[1:],
            "messages": [{
                "role": "assistant",
                "content": f"Got it! {questions.get(missing[1], 'Thanks! Let me prepare your brief.')}" if len(missing) > 1
                          else "Perfect! I have everything I need. Let me create your creative brief..."
            }],
            "cd_phase": "intake" if len(missing) > 1 else "brief",
            "cd_intake_complete": len(missing) <= 1,
            "last_successful_node": "intake_agent",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    # First interaction - greet and ask first question
    return {
        "messages": [{
            "role": "assistant",
            "content": f"Welcome to Creative Director mode! I'll help you create an amazing video.\n\n{question}"
        }],
        "cd_phase": "intake",
        "cd_intake_missing_info": missing,
        "last_successful_node": "intake_agent",
        "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
    }


# ============================================================================
# BRIEF AGENT - Generate structured brief from intake
# ============================================================================
async def brief_agent_node(state: NexusState) -> dict:
    """Generate structured creative brief from intake answers"""
    start_time = time.perf_counter()

    answers = state.get("cd_intake_answers", {})

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        # Create a basic brief without LLM
        brief = {
            "title": f"Video for {answers.get('business_name', 'Your Business')}",
            "business_name": answers.get("business_name", "Unknown"),
            "product_service": answers.get("product_service", "Unknown"),
            "target_audience": {"demographics": answers.get("target_audience", "Unknown")},
            "video_goal": answers.get("video_goal", "Brand awareness"),
            "tone": answers.get("tone", "Professional"),
            "duration_seconds": 30,
            "key_messages": ["Key message 1", "Key message 2", "Key message 3"],
            "call_to_action": "Learn more",
            "visual_style": "Modern and clean",
        }

        return {
            "cd_brief": brief,
            "cd_phase": "script",
            "messages": [{
                "role": "assistant",
                "content": f"Creative brief generated!\n\n**{brief.get('title')}**\n\nGenerating script..."
            }],
            "last_successful_node": "brief_agent",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        prompt = f"""Based on the following intake information, generate a structured creative brief:

INTAKE INFORMATION:
- Business: {answers.get('business_name', 'Unknown')}
- Product/Service: {answers.get('product_service', 'Unknown')}
- Target Audience: {answers.get('target_audience', 'Unknown')}
- Video Goal: {answers.get('video_goal', 'Unknown')}
- Tone: {answers.get('tone', 'Professional')}

Generate a JSON creative brief with the following structure:
{{
    "title": "Video title",
    "business_name": "...",
    "product_service": "...",
    "target_audience": {{
        "demographics": "...",
        "pain_points": ["..."],
        "desires": ["..."]
    }},
    "video_goal": "...",
    "tone": "...",
    "duration_seconds": 30,
    "key_messages": ["...", "...", "..."],
    "call_to_action": "...",
    "visual_style": "..."
}}

Respond with ONLY the JSON, no other text."""

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        brief_text = response.content[0].text
        brief = json.loads(brief_text)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "cd_brief": brief,
            "cd_phase": "script",
            "messages": [{
                "role": "assistant",
                "content": f"Creative brief generated!\n\n**{brief.get('title', 'Your Video')}**\n\nKey messages:\n" +
                          "\n".join([f"- {msg}" for msg in brief.get('key_messages', [])]) +
                          f"\n\nDuration: {brief.get('duration_seconds', 30)} seconds\n\nGenerating script..."
            }],
            "last_successful_node": "brief_agent",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        return {
            "errors": [{"node": "brief_agent", "error": str(e), "recoverable": True}],
            "cd_phase": "intake",  # Go back to intake
            "messages": [{
                "role": "assistant",
                "content": "I had trouble creating the brief. Let me ask a few more questions to clarify."
            }],
        }


# ============================================================================
# SCRIPT AGENT - Generate video script
# ============================================================================
async def script_agent_node(state: NexusState) -> dict:
    """Generate video script from brief"""
    start_time = time.perf_counter()

    brief = state.get("cd_brief", {})

    if not brief:
        return {
            "errors": [{"node": "script_agent", "error": "No brief available", "recoverable": True}],
            "cd_phase": "brief",
        }

    # Check if Anthropic is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        script_text = f"""SCENE 1: Opening Hook
VOICEOVER: "Tired of [problem]?"
VISUAL: Eye-catching imagery
DURATION: 3 seconds

SCENE 2: Problem Statement
VOICEOVER: "You're not alone..."
VISUAL: Relatable scenario
DURATION: 5 seconds

SCENE 3: Solution Introduction
VOICEOVER: "Introducing {brief.get('business_name', 'our solution')}..."
VISUAL: Product reveal
DURATION: 7 seconds

SCENE 4: Benefits
VOICEOVER: "{brief.get('key_messages', ['Your key benefit'])[0]}"
VISUAL: Feature demonstration
DURATION: 10 seconds

SCENE 5: Call to Action
VOICEOVER: "{brief.get('call_to_action', 'Learn more today!')}"
VISUAL: Logo and contact info
DURATION: 5 seconds"""

        return {
            "cd_script": {"raw_script": script_text, "brief_id": brief.get("title", "unknown")},
            "cd_phase": "review",
            "messages": [{
                "role": "assistant",
                "content": f"Script generated!\n\n```\n{script_text[:500]}...\n```\n\nWould you like me to proceed with video generation, or would you like to make changes?"
            }],
            "last_successful_node": "script_agent",
            "total_latency_ms": state.get("total_latency_ms", 0) + (time.perf_counter() - start_time) * 1000,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        prompt = f"""Generate a video script based on this creative brief:

{json.dumps(brief, indent=2)}

Create a script with the following structure:
1. Hook (first 3 seconds) - Grab attention
2. Problem statement - Identify the pain point
3. Solution introduction - Present the product/service
4. Benefits - 3 key benefits with visuals
5. Social proof (optional) - Testimonial or statistic
6. Call to action - Clear next step

Format as:
SCENE 1: [Description]
VOICEOVER: "..."
VISUAL: [Description]
DURATION: X seconds

Keep total duration around {brief.get('duration_seconds', 30)} seconds."""

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        script_text = response.content[0].text

        script = {
            "raw_script": script_text,
            "brief_id": brief.get("title", "unknown"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "cd_script": script,
            "cd_phase": "review",
            "messages": [{
                "role": "assistant",
                "content": f"Script generated!\n\n```\n{script_text[:500]}...\n```\n\nWould you like me to proceed with video generation, or would you like to make changes?"
            }],
            "last_successful_node": "script_agent",
            "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
        }

    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return {
            "errors": [{"node": "script_agent", "error": str(e), "recoverable": True}],
        }


# ============================================================================
# REVIEW AGENT - Script approval gate
# ============================================================================
async def review_agent_node(state: NexusState) -> dict:
    """Handle script review and approval"""

    last_message = state["messages"][-1]["content"].lower()

    # Check for approval signals
    approval_signals = ["yes", "proceed", "looks good", "approved", "generate", "create", "let's do it", "go ahead"]
    rejection_signals = ["no", "change", "modify", "edit", "revise", "different"]

    if any(signal in last_message for signal in approval_signals):
        return {
            "cd_script_approved": True,
            "cd_phase": "render",
            "messages": [{
                "role": "assistant",
                "content": "Approved! Starting video generation with RAGNAROK...\n\nThis will take a few minutes. I'll notify you when it's ready."
            }],
            "last_successful_node": "review_agent",
        }
    elif any(signal in last_message for signal in rejection_signals):
        return {
            "cd_phase": "script",
            "messages": [{
                "role": "assistant",
                "content": "No problem! What changes would you like me to make to the script?"
            }],
            "last_successful_node": "review_agent",
        }
    else:
        # Unclear response - ask for clarification
        return {
            "messages": [{
                "role": "assistant",
                "content": "Would you like me to proceed with generating the video, or would you prefer to make changes to the script first?"
            }],
            "last_successful_node": "review_agent",
        }


# ============================================================================
# RENDER AGENT - Trigger RAGNAROK pipeline
# ============================================================================
async def render_agent_node(state: NexusState) -> dict:
    """Trigger video rendering via RAGNAROK"""
    start_time = time.perf_counter()

    script = state.get("cd_script", {})
    brief = state.get("cd_brief", {})

    # TODO: Replace with actual RAGNAROK API call
    import uuid
    job_id = f"ragnarok-{uuid.uuid4().hex[:8]}"

    # Simulate job submission
    # In production: await ragnarok_client.submit_job(script, brief)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "cd_render_job_id": job_id,
        "cd_phase": "deliver",
        "messages": [{
            "role": "assistant",
            "content": f"Video generation started!\n\nJob ID: `{job_id}`\n\nEstimated time: 3-5 minutes\n\nI'll let you know as soon as it's ready!"
        }],
        "last_successful_node": "render_agent",
        "total_latency_ms": state.get("total_latency_ms", 0) + latency_ms,
    }


# ============================================================================
# DELIVER AGENT - Handle completed video delivery
# ============================================================================
async def deliver_agent_node(state: NexusState) -> dict:
    """Handle video delivery"""

    job_id = state.get("cd_render_job_id")

    # TODO: Check actual job status from RAGNAROK
    # Simulate completed video URL
    video_url = f"https://storage.barriosa2i.com/videos/{job_id}.mp4"

    return {
        "cd_video_url": video_url,
        "cd_delivery_status": "delivered",
        "cd_phase": "complete",
        "messages": [{
            "role": "assistant",
            "content": f"Your video is ready!\n\n**Download:** [Click here]({video_url})\n\nWould you like to:\n- Create another video\n- Make revisions to this one\n- Get help with something else"
        }],
        "last_successful_node": "deliver_agent",
    }


# ============================================================================
# BUILD SUBGRAPH
# ============================================================================
def build_creative_director_subgraph():
    """Build the Creative Director subgraph"""

    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available for Creative Director subgraph")
        return None

    from langgraph.graph import StateGraph, START, END

    cd = StateGraph(NexusState)

    # Add all agent nodes
    cd.add_node("intake_agent", intake_agent_node)
    cd.add_node("brief_agent", brief_agent_node)
    cd.add_node("script_agent", script_agent_node)
    cd.add_node("review_agent", review_agent_node)
    cd.add_node("render_agent", render_agent_node)
    cd.add_node("deliver_agent", deliver_agent_node)

    # Dynamic routing based on cd_phase
    def route_by_phase(state: NexusState) -> str:
        phase = state.get("cd_phase")

        if phase == "brief":
            return "brief_agent"
        elif phase == "script":
            return "script_agent"
        elif phase == "review":
            return "review_agent"
        elif phase == "render":
            return "render_agent"
        elif phase == "deliver":
            return "deliver_agent"
        elif phase == "complete":
            return END
        else:
            return "intake_agent"  # Default to intake

    # Entry point - route by phase
    cd.add_conditional_edges(
        START,
        route_by_phase,
        {
            "intake_agent": "intake_agent",
            "brief_agent": "brief_agent",
            "script_agent": "script_agent",
            "review_agent": "review_agent",
            "render_agent": "render_agent",
            "deliver_agent": "deliver_agent",
            END: END,
        }
    )

    # Intake -> Brief (when complete)
    cd.add_conditional_edges(
        "intake_agent",
        lambda s: "brief_agent" if s.get("cd_intake_complete") else END,
        {"brief_agent": "brief_agent", END: END}
    )

    # Brief -> Script
    cd.add_conditional_edges(
        "brief_agent",
        lambda s: "script_agent" if s.get("cd_brief") else END,
        {"script_agent": "script_agent", END: END}
    )

    # Script -> Review
    cd.add_edge("script_agent", "review_agent")

    # Review -> Render (if approved) or END (if needs changes)
    cd.add_conditional_edges(
        "review_agent",
        lambda s: "render_agent" if s.get("cd_script_approved") else END,
        {"render_agent": "render_agent", END: END}
    )

    # Render -> Deliver
    cd.add_edge("render_agent", "deliver_agent")

    # Deliver -> END
    cd.add_edge("deliver_agent", END)

    return cd.compile()


async def fallback_creative_director(state: NexusState) -> dict:
    """Fallback Creative Director pipeline when LangGraph not available"""
    phase = state.get("cd_phase", "intake")

    if phase == "intake" or not phase:
        return await intake_agent_node(state)
    elif phase == "brief":
        return await brief_agent_node(state)
    elif phase == "script":
        return await script_agent_node(state)
    elif phase == "review":
        return await review_agent_node(state)
    elif phase == "render":
        return await render_agent_node(state)
    elif phase == "deliver":
        return await deliver_agent_node(state)
    else:
        return await intake_agent_node(state)
