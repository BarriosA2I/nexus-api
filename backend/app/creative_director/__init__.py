"""
================================================================================
NEXUS - Creative Director Integration
================================================================================
Integrates the 6-agent Creative Director pipeline into NEXUS Unified.

Components:
- Intent Detector: Routes to Creative Director when video creation detected
- Session Manager: Handles Creative Director workflow state
- Pipeline Bridge: Connects NEXUS → Creative Director → RAGNAROK
- Unified Router: Intelligent routing between NEXUS and Creative Director

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

from app.creative_director.intent_detector import CreativeIntentDetector, CreativeIntent, IntentResult
from app.creative_director.session_manager import (
    CreativeDirectorSession,
    CreativeSessionManager,
    BriefData,
    WorkflowPhase,
    IntakePhase,
    IntakeConversationEngine,
)
from app.creative_director.pipeline_bridge import (
    CreativeDirectorBridge,
    PipelineEvent,
    PipelineEventData,
)
from app.creative_director.unified_router import (
    UnifiedRouter,
    RoutingDecision,
    NEXUSWithCreativeDirector,
)

__all__ = [
    # Intent Detection
    "CreativeIntentDetector",
    "CreativeIntent",
    "IntentResult",
    
    # Session Management
    "CreativeDirectorSession",
    "CreativeSessionManager",
    "BriefData",
    "WorkflowPhase",
    "IntakePhase",
    "IntakeConversationEngine",
    
    # Pipeline
    "CreativeDirectorBridge",
    "PipelineEvent",
    "PipelineEventData",
    
    # Routing
    "UnifiedRouter",
    "RoutingDecision",
    "NEXUSWithCreativeDirector",
]
