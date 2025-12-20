"""
Voiceover Service - Bridges Nexus Intake → VoiceoverMaster
Generates voiceovers after commercial intake completes.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from .voiceover_master_agent import (
    VoiceoverMasterAgent,
    VoiceoverRequest,
    VoiceoverResult,
    VoiceoverConfig,
    SceneType,
    VoiceEmotion,
)

logger = logging.getLogger("voiceover_service")

# Singleton instance
_agent: Optional[VoiceoverMasterAgent] = None
_initialized: bool = False


def get_voiceover_agent() -> VoiceoverMasterAgent:
    """Get or create VoiceoverMaster agent singleton"""
    global _agent, _initialized
    if _agent is None:
        config = VoiceoverConfig.from_env()
        _agent = VoiceoverMasterAgent(config)
        _initialized = True
        logger.info("VoiceoverMasterAgent initialized")
    return _agent


def is_initialized() -> bool:
    """Check if agent is initialized"""
    return _initialized


def transform_intake_to_scenes(client_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform Nexus intake client_config into VoiceoverMaster scene format.

    Intake format:
    {
        "client_id": "demo_001",
        "commercial_id": "comm_001",
        "business_name": "Acme Corp",
        "product_description": "AI automation",
        "target_audience": "B2B SaaS companies",
        "brand_voice": "Professional, confident",
        "video_architecture": {
            "scene_plan": [
                {"type": "hook", "text": "...", "duration": 3},
                {"type": "problem", "text": "...", "duration": 5},
                ...
            ]
        }
    }

    VoiceoverMaster format:
    [
        {"type": "hook", "text": "...", "duration": 3.0, "emotion": "urgent"},
        {"type": "solution", "text": "...", "duration": 6.0, "emotion": "confident"},
        ...
    ]
    """
    scenes = []

    # Get scene plan from video architecture
    video_arch = client_config.get("video_architecture", {})
    scene_plan = video_arch.get("scene_plan", [])

    # If no scene plan, generate default from intake data
    if not scene_plan:
        scene_plan = generate_default_scenes(client_config)

    # Map scene types to emotions
    emotion_map = {
        "hook": "urgent",
        "problem": "empathetic",
        "solution": "confident",
        "benefits": "warm",
        "proof": "professional",
        "cta": "urgent",
    }

    for scene in scene_plan:
        scene_type = scene.get("type", "hook")
        scenes.append({
            "type": scene_type,
            "text": scene.get("text", ""),
            "duration": float(scene.get("duration", 5.0)),
            "emotion": scene.get("emotion", emotion_map.get(scene_type, "confident")),
            "pacing": scene.get("pacing", "normal"),
            "emphasis": scene.get("emphasis_words", []),
        })

    return scenes


def generate_default_scenes(client_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate default scene structure from intake data if no scene plan exists"""
    business = client_config.get("business_name", "Your Business")
    product = client_config.get("product_description", "our solution")
    audience = client_config.get("target_audience", "businesses")
    problem = client_config.get("main_problem", "wasting time on manual tasks")
    benefit = client_config.get("key_benefit", "save time and money")

    return [
        {
            "type": "hook",
            "text": f"Tired of {problem}?",
            "duration": 3.0,
        },
        {
            "type": "problem",
            "text": f"Most {audience} struggle with inefficiency every single day.",
            "duration": 5.0,
        },
        {
            "type": "solution",
            "text": f"{business} changes everything with {product}.",
            "duration": 6.0,
        },
        {
            "type": "benefits",
            "text": f"You'll {benefit} starting from day one.",
            "duration": 5.0,
        },
        {
            "type": "cta",
            "text": "Book your strategy call today.",
            "duration": 4.0,
        },
    ]


async def generate_voiceover_from_intake(
    client_config: Dict[str, Any],
    quality_tier: str = "premium",
) -> Dict[str, Any]:
    """
    Main integration function: Generate voiceovers from intake client_config.

    Args:
        client_config: The completed intake configuration
        quality_tier: "premium" (ElevenLabs), "standard" (Azure), "budget" (gTTS)

    Returns:
        Updated client_config with voiceover data
    """
    agent = get_voiceover_agent()

    # Transform intake data to scenes
    scenes = transform_intake_to_scenes(client_config)

    if not scenes:
        logger.warning("No scenes to generate voiceovers for")
        return client_config

    # Create voiceover request
    request = VoiceoverRequest(
        client_id=client_config.get("client_id", "unknown"),
        commercial_id=client_config.get("commercial_id", f"comm_{int(time.time())}"),
        scenes=scenes,
        brand_voice_guidelines=client_config.get("brand_voice", ""),
        total_duration_seconds=sum(s.get("duration", 5.0) for s in scenes),
        quality_tier=quality_tier,
    )

    logger.info(f"Generating voiceovers for {len(scenes)} scenes...")

    # Generate voiceovers
    result: VoiceoverResult = await agent.generate(request)

    # Add voiceover data to client config
    client_config["voiceover"] = {
        "enabled": True,
        "status": "completed",
        "provider": result.provider_used,
        "voice_profile": result.voice_profile,
        "scenes": result.scenes,
        "drive_assets": result.drive_assets,
        "quality_metrics": result.quality_metrics,
        "total_cost_usd": result.total_cost_usd,
        "processing_time_ms": result.processing_time_ms,
    }

    logger.info(
        f"Voiceovers generated: {len(result.scenes)} scenes, "
        f"${result.total_cost_usd:.3f} cost, "
        f"{result.processing_time_ms:.0f}ms"
    )

    return client_config


async def generate_single_voiceover(
    text: str,
    scene_type: str = "hook",
    duration: float = 5.0,
    emotion: str = "confident",
    client_id: str = "single",
    quality_tier: str = "premium",
) -> Dict[str, Any]:
    """
    Generate a single scene voiceover.

    Useful for testing or regenerating individual scenes.
    """
    agent = get_voiceover_agent()

    result = await agent.generate_scene_voiceover(
        client_id=client_id,
        scene_text=text,
        scene_type=scene_type,
        duration_seconds=duration,
        emotion=emotion,
    )

    return result


def get_service_health() -> Dict[str, Any]:
    """Get voiceover service health info"""
    if not _initialized or _agent is None:
        return {
            "status": "unavailable",
            "initialized": False,
        }

    status = _agent.get_status()

    return {
        "status": "available",
        "initialized": True,
        "agent_status": status.get("status", "unknown"),
        "providers": status.get("providers", {}),
        "memory": status.get("memory", {}),
    }
