"""
================================================================================
NEXUS - Unified Router with Creative Director Integration
================================================================================
Intelligent routing layer that detects video creation intent and seamlessly
hands off to Creative Director while maintaining standard NEXUS functionality.

Flow:
1. User sends message to /api/chat/message
2. Router checks for active Creative Director session
3. If in CD mode → Route to Creative Director
4. If not → Check intent detection
5. If video intent detected → Start CD session, handoff
6. If general → Route to standard NEXUS Brain

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from opentelemetry import trace

from app.creative_director.intent_detector import CreativeIntentDetector, CreativeIntent, IntentResult
from app.creative_director.session_manager import CreativeSessionManager, WorkflowPhase

logger = logging.getLogger("nexus.router.unified")
tracer = trace.get_tracer("nexus.router.unified")


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    route_to: str  # "nexus" | "creative_director" | "creative_director_pipeline"
    session_id: Optional[str] = None
    handoff_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedRouter:
    """
    Unified routing layer for NEXUS + Creative Director.
    
    Determines where to route messages based on:
    1. Active session state (already in CD mode?)
    2. Intent detection (user wants video?)
    3. Explicit commands (/video, /creative-director, etc.)
    
    Usage:
        router = UnifiedRouter(cd_session_manager, intent_detector)
        decision = router.route(message, nexus_session_id, session_context)
        
        if decision.route_to == "creative_director":
            # Handle CD intake
        elif decision.route_to == "creative_director_pipeline":
            # Handle CD pipeline execution
        else:
            # Handle standard NEXUS
    """
    
    # Explicit command triggers
    CD_COMMANDS = {
        "/video": True,
        "/creative-director": True,
        "/cd": True,
        "/commercial": True,
        "/ad": True,
        "/neural-ad-forge": True,
    }
    
    # Exit commands
    EXIT_COMMANDS = {
        "/exit-cd": True,
        "/back": True,
        "/exit-video": True,
        "/cancel": True,
    }
    
    def __init__(
        self,
        cd_session_manager: CreativeSessionManager,
        intent_detector: CreativeIntentDetector,
        confidence_threshold: float = 0.7,
    ):
        self.cd_session_manager = cd_session_manager
        self.intent_detector = intent_detector
        self.confidence_threshold = confidence_threshold
    
    def route(
        self,
        message: str,
        nexus_session_id: str,
        session_context: Optional[Dict[str, Any]] = None,
        user_id: str = "anonymous",
    ) -> RoutingDecision:
        """
        Determine where to route a message.
        
        Args:
            message: User's message
            nexus_session_id: NEXUS session identifier
            session_context: Optional context with session state
            user_id: User identifier
            
        Returns:
            RoutingDecision with route and any handoff info
        """
        with tracer.start_as_current_span("unified_router.route") as span:
            span.set_attribute("message_length", len(message))
            span.set_attribute("nexus_session_id", nexus_session_id)
            
            message_lower = message.lower().strip()
            
            # Check for explicit exit command first
            if any(cmd in message_lower for cmd in self.EXIT_COMMANDS):
                span.set_attribute("route", "nexus")
                span.set_attribute("reason", "exit_command")
                
                # Close CD session if exists
                cd_session = self.cd_session_manager.get_session_by_nexus_id(nexus_session_id)
                if cd_session:
                    self.cd_session_manager.close_session(cd_session.session_id)
                
                return RoutingDecision(
                    route_to="nexus",
                    handoff_message="Exited Creative Director mode. How can I help you?",
                    metadata={"reason": "exit_command"},
                )
            
            # Check for explicit CD command
            if any(cmd in message_lower for cmd in self.CD_COMMANDS):
                span.set_attribute("route", "creative_director")
                span.set_attribute("reason", "explicit_command")
                
                return self._create_or_get_cd_session(
                    nexus_session_id, user_id, "explicit_command"
                )
            
            # Check for active CD session
            cd_session = self.cd_session_manager.get_session_by_nexus_id(nexus_session_id)
            
            if cd_session and cd_session.is_active:
                span.set_attribute("cd_session_id", cd_session.session_id)
                span.set_attribute("workflow_phase", cd_session.workflow_phase.value)
                
                # Route based on workflow phase
                if cd_session.workflow_phase == WorkflowPhase.INTAKE:
                    span.set_attribute("route", "creative_director")
                    span.set_attribute("reason", "active_intake")
                    
                    return RoutingDecision(
                        route_to="creative_director",
                        session_id=cd_session.session_id,
                        metadata={
                            "reason": "active_intake",
                            "phase": cd_session.intake_phase.value,
                        },
                    )
                
                elif cd_session.workflow_phase in [
                    WorkflowPhase.RESEARCH,
                    WorkflowPhase.IDEATION,
                    WorkflowPhase.SCRIPTING,
                    WorkflowPhase.REVIEW,
                    WorkflowPhase.PRODUCTION,
                ]:
                    span.set_attribute("route", "creative_director_pipeline")
                    span.set_attribute("reason", "active_pipeline")
                    
                    return RoutingDecision(
                        route_to="creative_director_pipeline",
                        session_id=cd_session.session_id,
                        metadata={
                            "reason": "active_pipeline",
                            "phase": cd_session.workflow_phase.value,
                        },
                    )
                
                elif cd_session.workflow_phase == WorkflowPhase.COMPLETE:
                    # Video complete, check if user wants revision
                    if any(word in message_lower for word in ["revision", "change", "redo", "again", "modify"]):
                        span.set_attribute("route", "creative_director")
                        span.set_attribute("reason", "revision_request")
                        
                        return RoutingDecision(
                            route_to="creative_director",
                            session_id=cd_session.session_id,
                            handoff_message="I can help with revisions! What would you like to change about the video?",
                            metadata={"reason": "revision_request"},
                        )
                    else:
                        # Session complete, route to NEXUS but keep session for reference
                        span.set_attribute("route", "nexus")
                        span.set_attribute("reason", "session_complete")
                        
                        return RoutingDecision(
                            route_to="nexus",
                            metadata={
                                "reason": "session_complete",
                                "video_url": cd_session.video_url,
                            },
                        )
            
            # No active CD session - check intent
            intent_result = self.intent_detector.detect(message, session_context)
            span.set_attribute("intent", intent_result.intent.value)
            span.set_attribute("intent_confidence", intent_result.confidence)
            
            if intent_result.should_handoff:
                span.set_attribute("route", "creative_director")
                span.set_attribute("reason", "intent_detection")
                
                return self._create_or_get_cd_session(
                    nexus_session_id, user_id, "intent_detection",
                    intent_result,
                )
            
            # Default: route to NEXUS
            span.set_attribute("route", "nexus")
            span.set_attribute("reason", "default")
            
            return RoutingDecision(
                route_to="nexus",
                metadata={
                    "reason": "default",
                    "intent": intent_result.intent.value,
                    "confidence": intent_result.confidence,
                },
            )
    
    def _create_or_get_cd_session(
        self,
        nexus_session_id: str,
        user_id: str,
        reason: str,
        intent_result: Optional[IntentResult] = None,
    ) -> RoutingDecision:
        """Create or get existing CD session"""
        cd_session = self.cd_session_manager.get_session_by_nexus_id(nexus_session_id)
        
        if not cd_session:
            cd_session = self.cd_session_manager.create_session(
                nexus_session_id, user_id
            )
        
        handoff_message = None
        if intent_result and intent_result.suggested_response:
            handoff_message = intent_result.suggested_response
        elif reason == "explicit_command":
            handoff_message = (
                "🎬 **Creative Director Mode Activated!**\n\n"
                "I'll help you create a professional video ad. Let's start with your business.\n\n"
                "**What's your company name?**"
            )
        
        return RoutingDecision(
            route_to="creative_director",
            session_id=cd_session.session_id,
            handoff_message=handoff_message,
            metadata={
                "reason": reason,
                "is_new_session": cd_session.created_at == cd_session.updated_at,
            },
        )
    
    def get_session_context_for_cd(
        self,
        nexus_session_id: str,
    ) -> Dict[str, Any]:
        """Get context dict for session that includes CD state"""
        cd_session = self.cd_session_manager.get_session_by_nexus_id(nexus_session_id)
        
        if not cd_session:
            return {}
        
        return cd_session.get_context_for_nexus()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

class NEXUSWithCreativeDirector:
    """
    High-level integration class for NEXUS + Creative Director.
    
    Provides a simple interface for handling messages with automatic
    routing between NEXUS Brain and Creative Director.
    
    Usage:
        nexus = NEXUSWithCreativeDirector(brain, cd_manager, cd_bridge)
        response = await nexus.handle_message(session_id, user_id, message)
    """
    
    def __init__(
        self,
        nexus_brain,  # Your existing NEXUS Brain instance
        cd_session_manager: CreativeSessionManager,
        cd_pipeline_bridge: 'CreativeDirectorBridge',
        intent_detector: Optional[CreativeIntentDetector] = None,
    ):
        self.nexus_brain = nexus_brain
        self.cd_session_manager = cd_session_manager
        self.cd_pipeline_bridge = cd_pipeline_bridge
        self.intent_detector = intent_detector or CreativeIntentDetector()
        
        self.router = UnifiedRouter(
            cd_session_manager=cd_session_manager,
            intent_detector=self.intent_detector,
        )
    
    async def handle_message(
        self,
        session_id: str,
        user_id: str,
        message: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a user message with intelligent routing.
        
        Returns:
            Dict with response and metadata including routing info
        """
        with tracer.start_as_current_span("nexus_cd.handle_message") as span:
            span.set_attribute("session_id", session_id)
            
            # Get routing decision
            decision = self.router.route(
                message=message,
                nexus_session_id=session_id,
                session_context=session_context,
                user_id=user_id,
            )
            
            span.set_attribute("route_to", decision.route_to)
            
            # Route to appropriate handler
            if decision.route_to == "creative_director":
                return await self._handle_creative_director(
                    decision, message, session_id, user_id
                )
            
            elif decision.route_to == "creative_director_pipeline":
                return await self._handle_creative_director_pipeline(
                    decision, message, session_id
                )
            
            else:  # "nexus"
                return await self._handle_nexus(
                    decision, message, session_id
                )
    
    async def _handle_creative_director(
        self,
        decision: RoutingDecision,
        message: str,
        session_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Handle Creative Director intake conversation"""
        
        # If this is a new session with handoff message, return that
        if decision.handoff_message and decision.metadata.get("is_new_session"):
            return {
                "response": decision.handoff_message,
                "mode": "creative_director",
                "phase": "greeting",
                "session_id": decision.session_id,
            }
        
        # Process intake message
        result = await self.cd_session_manager.process_intake_message(
            decision.session_id,
            message,
        )
        
        response = {
            "response": result["response"],
            "mode": "creative_director",
            "phase": result["phase"],
            "session_id": decision.session_id,
            "is_complete": result.get("is_complete", False),
        }
        
        # If intake complete, include brief and trigger pipeline
        if result.get("is_complete"):
            response["brief"] = result.get("brief")
            response["next_action"] = "pipeline_ready"
        
        return response
    
    async def _handle_creative_director_pipeline(
        self,
        decision: RoutingDecision,
        message: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Handle messages during active pipeline execution"""
        
        cd_session = self.cd_session_manager.get_session(decision.session_id)
        phase = cd_session.workflow_phase.value if cd_session else "unknown"
        
        # During pipeline, provide status updates
        if cd_session:
            status = await self.cd_pipeline_bridge.get_pipeline_status(cd_session)
            
            status_message = self._get_pipeline_status_message(cd_session, status)
            
            return {
                "response": status_message,
                "mode": "creative_director_pipeline",
                "phase": phase,
                "session_id": decision.session_id,
                "status": status,
            }
        
        return {
            "response": "Your video is being processed. I'll let you know when it's ready!",
            "mode": "creative_director_pipeline",
            "phase": phase,
            "session_id": decision.session_id,
        }
    
    async def _handle_nexus(
        self,
        decision: RoutingDecision,
        message: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Handle standard NEXUS Brain message"""
        
        # If there's a handoff message (e.g., exiting CD mode), prepend it
        if decision.handoff_message:
            prefix = decision.handoff_message + "\n\n"
        else:
            prefix = ""
        
        # Call NEXUS Brain (your existing implementation)
        # This is a placeholder - integrate with your actual NEXUS Brain
        if self.nexus_brain:
            response = await self.nexus_brain.process(session_id, message)
        else:
            response = {
                "response": f"{prefix}I'm the NEXUS assistant. How can I help you today?",
            }
        
        return {
            **response,
            "mode": "nexus",
            "metadata": decision.metadata,
        }
    
    def _get_pipeline_status_message(
        self,
        session: 'CreativeDirectorSession',
        status: Dict[str, Any],
    ) -> str:
        """Generate human-readable pipeline status message"""
        phase = session.workflow_phase
        
        messages = {
            WorkflowPhase.RESEARCH: "🔍 Our Research Agent is analyzing your market, competitors, and audience insights...",
            WorkflowPhase.IDEATION: "💡 Our Ideation Agent is generating creative concepts for your video...",
            WorkflowPhase.SCRIPTING: "✍️ Our Script Agent is writing your video script...",
            WorkflowPhase.REVIEW: "✅ Our Review Agent is quality-checking the script...",
            WorkflowPhase.PRODUCTION: f"🎬 Your video is being rendered... {status.get('production_job', {}).get('progress', 0)}% complete",
            WorkflowPhase.COMPLETE: f"🎉 Your video is ready! Watch it here: {session.video_url}",
            WorkflowPhase.ERROR: "❌ There was an issue with your video. Our team has been notified.",
        }
        
        return messages.get(phase, "Processing your request...")


# =============================================================================
# MIDDLEWARE FOR FASTAPI
# =============================================================================

async def creative_director_middleware(
    request,
    call_next,
    router: UnifiedRouter,
):
    """
    FastAPI middleware that injects CD routing context.
    
    Usage:
        app.middleware("http")(lambda req, call_next: 
            creative_director_middleware(req, call_next, router))
    """
    # Add router to request state for use in endpoints
    request.state.cd_router = router
    
    response = await call_next(request)
    return response
