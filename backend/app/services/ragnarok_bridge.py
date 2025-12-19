"""
Nexus Assistant Unified - Ragnarok Bridge
Integration bridge for RAGNAROK video generation pipeline
"""
import asyncio
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..config import settings
from .job_store import Job, get_job_store

logger = logging.getLogger(__name__)


class RagnarokMode(str, Enum):
    """Ragnarok operation mode"""
    MOCK = "mock"      # Mock outputs for development
    PATH = "path"      # Import from local path
    API = "api"        # Call external API


@dataclass
class RagnarokResult:
    """Ragnarok generation result"""
    success: bool
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    script: Optional[str] = None
    duration_seconds: Optional[int] = None
    cost_usd: Optional[float] = None
    generation_time_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RagnarokBridge:
    """
    Bridge to RAGNAROK video generation pipeline.

    Supports three modes:
    - mock: Return simulated outputs (for development)
    - path: Import Ragnarok modules from local path
    - api: Call external Ragnarok API
    """

    def __init__(
        self,
        mode: str = None,
        ragnarok_path: Optional[Path] = None,
        api_url: Optional[str] = None,
    ):
        self.mode = RagnarokMode(mode or settings.RAGNAROK_MODE)
        self.ragnarok_path = ragnarok_path or settings.ragnarok_path
        self.api_url = api_url or settings.RAGNAROK_API_URL

        self._pipeline = None
        self._orchestrator_class = None
        self._orchestrator_type = None
        self._commercial_format = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    async def initialize(self) -> bool:
        """Initialize Ragnarok bridge based on mode"""
        try:
            if self.mode == RagnarokMode.MOCK:
                logger.info("Ragnarok bridge initialized in MOCK mode")
                self._is_initialized = True
                return True

            elif self.mode == RagnarokMode.PATH:
                if not self.ragnarok_path or not self.ragnarok_path.exists():
                    logger.warning(f"Ragnarok path not found: {self.ragnarok_path}")
                    logger.info("Falling back to MOCK mode")
                    self.mode = RagnarokMode.MOCK
                    self._is_initialized = True
                    return True

                # Add to Python path
                sys.path.insert(0, str(self.ragnarok_path))

                # Try importing RAGNAROK v7.0 modules in priority order
                imported = False

                # Option 1: CommercialWorkflowOrchestrator (9-agent pipeline)
                try:
                    from commercial_agent_bridge import CommercialWorkflowOrchestrator, CommercialFormat
                    logger.info(f"CommercialWorkflowOrchestrator found at {self.ragnarok_path}")
                    self._orchestrator_class = CommercialWorkflowOrchestrator
                    self._orchestrator_type = "commercial_workflow"
                    self._commercial_format = CommercialFormat
                    imported = True
                except ImportError as e:
                    logger.debug(f"CommercialWorkflowOrchestrator not found: {e}")

                # Option 2: RAGNAROKv7Orchestrator
                if not imported:
                    try:
                        from ragnarok_v7_apex import RAGNAROKv7Orchestrator
                        logger.info(f"RAGNAROKv7Orchestrator found at {self.ragnarok_path}")
                        self._orchestrator_class = RAGNAROKv7Orchestrator
                        self._orchestrator_type = "ragnarok_v7"
                        imported = True
                    except ImportError as e:
                        logger.debug(f"RAGNAROKv7Orchestrator not found: {e}")

                # Option 3: Trinity orchestrator
                if not imported:
                    try:
                        from trinity_orchestrator_legendary import TrinityOrchestrator
                        logger.info(f"TrinityOrchestrator found at {self.ragnarok_path}")
                        self._orchestrator_class = TrinityOrchestrator
                        self._orchestrator_type = "trinity"
                        imported = True
                    except ImportError as e:
                        logger.debug(f"TrinityOrchestrator not found: {e}")

                if imported:
                    logger.info(f"RAGNAROK {self._orchestrator_type} loaded from {self.ragnarok_path}")
                    self._is_initialized = True
                    return True
                else:
                    logger.warning(f"No compatible orchestrator found in {self.ragnarok_path}")
                    logger.info("Falling back to MOCK mode")
                    self.mode = RagnarokMode.MOCK
                    self._is_initialized = True
                    return True

            elif self.mode == RagnarokMode.API:
                if not self.api_url:
                    logger.warning("Ragnarok API URL not configured")
                    logger.info("Falling back to MOCK mode")
                    self.mode = RagnarokMode.MOCK

                self._is_initialized = True
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to initialize Ragnarok bridge: {e}")
            self.mode = RagnarokMode.MOCK
            self._is_initialized = True
            return True

    async def generate(
        self,
        brief: str,
        industry: str = "technology",
        duration_seconds: int = 30,
        platform: str = "youtube_1080p",
        style: Optional[str] = None,
        voice_style: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RagnarokResult:
        """
        Generate a commercial video.

        Returns RagnarokResult with video URL and metadata.
        """
        start_time = time.time()

        try:
            if self.mode == RagnarokMode.MOCK:
                return await self._mock_generate(
                    brief, industry, duration_seconds, platform, start_time
                )

            elif self.mode == RagnarokMode.PATH:
                return await self._path_generate(
                    brief, industry, duration_seconds, platform, style, voice_style, start_time
                )

            elif self.mode == RagnarokMode.API:
                return await self._api_generate(
                    brief, industry, duration_seconds, platform, style, voice_style, metadata, start_time
                )

            else:
                return RagnarokResult(
                    success=False,
                    error=f"Unknown Ragnarok mode: {self.mode}"
                )

        except Exception as e:
            logger.error(f"Ragnarok generation failed: {e}")
            return RagnarokResult(
                success=False,
                error=str(e)
            )

    async def _mock_generate(
        self,
        brief: str,
        industry: str,
        duration_seconds: int,
        platform: str,
        start_time: float,
    ) -> RagnarokResult:
        """Generate mock result for development"""
        # Simulate processing time
        await asyncio.sleep(2.0)

        generation_time = time.time() - start_time

        # Generate mock script
        mock_script = f"""
[SCENE 1 - OPENING]
Visual: Dynamic tech montage
VO: "In a world driven by innovation..."

[SCENE 2 - PROBLEM]
Visual: Pain point illustration
VO: "Businesses struggle with {industry} challenges..."

[SCENE 3 - SOLUTION]
Visual: Product showcase
VO: "{brief[:50]}..."

[SCENE 4 - CTA]
Visual: Logo + call to action
VO: "Transform your future today."

Duration: {duration_seconds} seconds
Platform: {platform}
"""

        return RagnarokResult(
            success=True,
            video_url=f"https://storage.barriosa2i.com/mock/commercial_{int(time.time())}.mp4",
            thumbnail_url=f"https://storage.barriosa2i.com/mock/thumb_{int(time.time())}.jpg",
            script=mock_script.strip(),
            duration_seconds=duration_seconds,
            cost_usd=2.60,
            generation_time_seconds=generation_time,
            metadata={
                "mode": "mock",
                "industry": industry,
                "platform": platform,
                "model_version": "RAGNAROK_v7.0_APEX_MOCK",
            }
        )

    async def _path_generate(
        self,
        brief: str,
        industry: str,
        duration_seconds: int,
        platform: str,
        style: Optional[str],
        voice_style: Optional[str],
        start_time: float,
    ) -> RagnarokResult:
        """Generate using local RAGNAROK v7.0 APEX installation"""
        import os

        if not self._orchestrator_class:
            logger.warning("No orchestrator class loaded, falling back to mock")
            return await self._mock_generate(brief, industry, duration_seconds, platform, start_time)

        try:
            # Handle based on orchestrator type
            if self._orchestrator_type == "commercial_workflow":
                # CommercialWorkflowOrchestrator requires API keys
                orchestrator = self._orchestrator_class(
                    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                    google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                    laozhang_api_key=os.environ.get("LAOZHANG_API_KEY", ""),
                    elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
                    cache_enabled=True,
                )

                # Execute the 9-agent workflow
                state = await orchestrator.execute_workflow(
                    business_name=brief,
                    user_id="nexus_assistant",
                    formats=[self._commercial_format.HD_1080P] if self._commercial_format else None,
                    priority=5,
                )

                generation_time = time.time() - start_time

                # Extract results from workflow state
                return RagnarokResult(
                    success=state.status == "completed" if hasattr(state, 'status') else True,
                    video_url=getattr(state, 'video_url', None),
                    thumbnail_url=getattr(state, 'thumbnail_url', None),
                    script=getattr(state, 'script', None) or getattr(state, 'story', None),
                    duration_seconds=duration_seconds,
                    cost_usd=getattr(state, 'total_cost', 2.60),
                    generation_time_seconds=generation_time,
                    metadata={
                        "mode": "path_commercial_workflow",
                        "orchestrator_type": self._orchestrator_type,
                        "workflow_id": getattr(state, 'workflow_id', None),
                        "model_version": "RAGNAROK_v7.0_APEX",
                        "agents_used": 9,
                    }
                )

            elif self._orchestrator_type == "ragnarok_v7":
                # RAGNAROKv7Orchestrator
                orchestrator = self._orchestrator_class()

                result = await orchestrator.process_request(
                    query=brief,
                    context={
                        "industry": industry,
                        "duration": duration_seconds,
                        "platform": platform,
                        "style": style,
                        "voice_style": voice_style,
                    }
                )

                generation_time = time.time() - start_time

                return RagnarokResult(
                    success=result.get("success", True),
                    video_url=result.get("video_url"),
                    thumbnail_url=result.get("thumbnail_url"),
                    script=result.get("script") or result.get("response"),
                    duration_seconds=result.get("duration", duration_seconds),
                    cost_usd=result.get("cost", 2.60),
                    generation_time_seconds=generation_time,
                    metadata={
                        "mode": "path_ragnarok_v7",
                        "orchestrator_type": self._orchestrator_type,
                        "model_version": "RAGNAROK_v7.0_APEX",
                        **result.get("metadata", {}),
                    }
                )

            elif self._orchestrator_type == "trinity":
                # Trinity orchestrator for market intelligence
                orchestrator = self._orchestrator_class()

                result = await orchestrator.execute(
                    query=brief,
                    context={"industry": industry}
                )

                generation_time = time.time() - start_time

                return RagnarokResult(
                    success=True,
                    script=result.get("analysis") or result.get("response"),
                    duration_seconds=duration_seconds,
                    cost_usd=result.get("cost", 0.50),
                    generation_time_seconds=generation_time,
                    metadata={
                        "mode": "path_trinity",
                        "orchestrator_type": self._orchestrator_type,
                        "model_version": "Trinity_Legendary",
                    }
                )

            else:
                logger.warning(f"Unknown orchestrator type: {self._orchestrator_type}")
                return await self._mock_generate(brief, industry, duration_seconds, platform, start_time)

        except Exception as e:
            logger.error(f"Path generation failed: {e}", exc_info=True)
            return await self._mock_generate(brief, industry, duration_seconds, platform, start_time)

    async def _api_generate(
        self,
        brief: str,
        industry: str,
        duration_seconds: int,
        platform: str,
        style: Optional[str],
        voice_style: Optional[str],
        metadata: Optional[Dict[str, Any]],
        start_time: float,
    ) -> RagnarokResult:
        """
        Generate using external RAGNAROK v7.0 APEX API.

        RAGNAROK API endpoint: POST /api/v1/commercial/generate
        Expected request: { business_name, user_id, formats, priority }
        """
        import httpx

        try:
            # Map platform to RAGNAROK format
            format_map = {
                "youtube_1080p": "1080p",
                "tiktok": "tiktok_vertical",
                "instagram": "instagram_square",
                "shorts": "youtube_shorts",
            }
            ragnarok_format = format_map.get(platform, "1080p")

            async with httpx.AsyncClient(timeout=settings.RAGNAROK_TIMEOUT) as client:
                # Call RAGNAROK commercial generate endpoint
                response = await client.post(
                    f"{self.api_url}/api/v1/commercial/generate",
                    json={
                        "business_name": brief,  # Use brief as business name
                        "user_id": "nexus_assistant",
                        "formats": [ragnarok_format],
                        "priority": 5,
                    }
                )
                response.raise_for_status()
                result = response.json()

            generation_time = time.time() - start_time

            # RAGNAROK returns workflow_id for async tracking
            return RagnarokResult(
                success=True,
                video_url=result.get("tracking_url"),  # Tracking URL until video ready
                thumbnail_url=None,
                script=None,  # Script generated async in workflow
                duration_seconds=result.get("estimated_duration_seconds", 243),
                cost_usd=result.get("estimated_cost_dollars", 2.60),
                generation_time_seconds=generation_time,
                metadata={
                    "mode": "api",
                    "workflow_id": result.get("workflow_id"),
                    "status": result.get("status"),
                    "tracking_url": result.get("tracking_url"),
                    "model_version": "RAGNAROK_v7.0_APEX",
                }
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"RAGNAROK API error: {e.response.status_code} - {e.response.text}")
            return await self._mock_generate(brief, industry, duration_seconds, platform, start_time)
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return await self._mock_generate(brief, industry, duration_seconds, platform, start_time)

    def health_info(self) -> Dict[str, Any]:
        """Get health information"""
        return {
            "status": "available" if self._is_initialized else "unavailable",
            "mode": self.mode.value,
            "ragnarok_path": str(self.ragnarok_path) if self.ragnarok_path else None,
            "api_url": self.api_url,
            "pipeline_loaded": self._pipeline is not None,
            "orchestrator_type": self._orchestrator_type,
            "orchestrator_loaded": self._orchestrator_class is not None,
        }


# Job handler for async processing
async def ragnarok_job_handler(job: Job) -> Dict[str, Any]:
    """Handle Ragnarok generation job"""
    bridge = get_ragnarok_bridge()

    if not bridge.is_initialized:
        await bridge.initialize()

    # Update progress
    job_store = get_job_store()
    await job_store.update_progress(job.id, 10)

    result = await bridge.generate(
        brief=job.payload.get("brief", ""),
        industry=job.payload.get("industry", "technology"),
        duration_seconds=job.payload.get("duration_seconds", 30),
        platform=job.payload.get("platform", "youtube_1080p"),
        style=job.payload.get("style"),
        voice_style=job.payload.get("voice_style"),
        metadata=job.payload.get("metadata"),
    )

    await job_store.update_progress(job.id, 100)

    if result.success:
        return {
            "video_url": result.video_url,
            "thumbnail_url": result.thumbnail_url,
            "script": result.script,
            "duration_seconds": result.duration_seconds,
            "cost_usd": result.cost_usd,
            "generation_time_seconds": result.generation_time_seconds,
            "metadata": result.metadata,
        }
    else:
        raise Exception(result.error or "Generation failed")


# Global Ragnarok bridge instance
_ragnarok_bridge: Optional[RagnarokBridge] = None


def get_ragnarok_bridge() -> RagnarokBridge:
    """Get or create Ragnarok bridge instance"""
    global _ragnarok_bridge
    if _ragnarok_bridge is None:
        _ragnarok_bridge = RagnarokBridge()
    return _ragnarok_bridge
