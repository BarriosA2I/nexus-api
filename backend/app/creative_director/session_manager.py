"""
================================================================================
NEXUS - Creative Director Session Manager
================================================================================
Manages Creative Director workflow sessions within NEXUS.

Features:
- Intake conversation state machine
- Brief data collection
- Production job tracking
- Session persistence

Workflow Phases:
1. GREETING → Welcome and explain process
2. BUSINESS → Collect business name/industry
3. AUDIENCE → Define target audience
4. PLATFORM → Select video platform
5. DURATION → Set video length
6. MESSAGE → Key message and USPs
7. TONE → Brand tone and budget tier
8. COMPETITORS → Competitor analysis
9. CONFIRM → Review and confirm brief
10. RESEARCH → 6-agent pipeline running
11. CONCEPTS → Concept selection
12. PRODUCTION → Video generation
13. DELIVERY → Final delivery

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("nexus.creative_director.session")


class IntakePhase(Enum):
    """Intake conversation phases"""
    GREETING = "greeting"
    BUSINESS = "business"
    INDUSTRY = "industry"
    AUDIENCE = "audience"
    PLATFORM = "platform"
    DURATION = "duration"
    MESSAGE = "message"
    USPS = "usps"
    TONE = "tone"
    BUDGET = "budget"
    COMPETITORS = "competitors"
    CONFIRM = "confirm"
    COMPLETE = "complete"


class WorkflowPhase(Enum):
    """Overall workflow phases"""
    INTAKE = "intake"
    RESEARCH = "research"
    IDEATION = "ideation"
    SCRIPTING = "scripting"
    REVIEW = "review"
    PRODUCTION = "production"
    DELIVERY = "delivery"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class BriefData:
    """Creative brief data model"""
    session_id: str
    business_name: Optional[str] = None
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    target_platform: Optional[str] = None
    video_duration: int = 30
    key_message: Optional[str] = None
    unique_selling_points: List[str] = field(default_factory=list)
    brand_tone: str = "professional"
    budget_tier: str = "standard"
    competitors: List[str] = field(default_factory=list)
    additional_notes: Optional[str] = None
    is_complete: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "business_name": self.business_name,
            "industry": self.industry,
            "target_audience": self.target_audience,
            "target_platform": self.target_platform,
            "video_duration": self.video_duration,
            "key_message": self.key_message,
            "unique_selling_points": self.unique_selling_points,
            "brand_tone": self.brand_tone,
            "budget_tier": self.budget_tier,
            "competitors": self.competitors,
            "additional_notes": self.additional_notes,
            "is_complete": self.is_complete,
        }
    
    def get_completion_percentage(self) -> float:
        """Calculate brief completion percentage"""
        required_fields = [
            self.business_name,
            self.industry,
            self.target_audience,
            self.target_platform,
            self.key_message,
        ]
        filled = sum(1 for f in required_fields if f)
        return (filled / len(required_fields)) * 100


@dataclass
class ProductionJob:
    """Production job tracking"""
    job_id: str
    session_id: str
    status: str = "pending"
    progress: int = 0
    video_url: Optional[str] = None
    preview_url: Optional[str] = None
    script: Optional[Dict] = None
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CreativeDirectorSession:
    """
    Creative Director session state.
    
    Tracks the entire workflow from intake to delivery.
    """
    session_id: str
    user_id: str
    
    # Mode flags
    is_active: bool = True
    workflow_phase: WorkflowPhase = WorkflowPhase.INTAKE
    intake_phase: IntakePhase = IntakePhase.GREETING
    
    # Data
    brief: BriefData = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Research results
    market_research: Optional[Dict] = None
    audience_insights: Optional[Dict] = None
    competitor_analysis: Optional[Dict] = None
    
    # Creative outputs
    concepts: List[Dict] = field(default_factory=list)
    selected_concept_id: Optional[str] = None
    script: Optional[Dict] = None
    
    # Production
    production_job: Optional[ProductionJob] = None
    video_url: Optional[str] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    revision_count: int = 0
    
    def __post_init__(self):
        if self.brief is None:
            self.brief = BriefData(session_id=self.session_id)
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": self.intake_phase.value,
        })
        self.updated_at = datetime.utcnow()
    
    def get_context_for_nexus(self) -> Dict[str, Any]:
        """Get context to inject into NEXUS session"""
        return {
            "creative_director_mode": self.workflow_phase.value,
            "creative_director_phase": self.intake_phase.value if self.workflow_phase == WorkflowPhase.INTAKE else None,
            "creative_director_session_id": self.session_id,
            "brief_completion": self.brief.get_completion_percentage(),
        }


# =============================================================================
# INTAKE CONVERSATION ENGINE
# =============================================================================

class IntakeConversationEngine:
    """
    Manages the intake conversation flow.
    
    Guides users through brief collection with smart prompts
    and validation.
    """
    
    # Phase prompts
    PHASE_PROMPTS = {
        IntakePhase.GREETING: (
            "Welcome to the AI Creative Director! 🎬\n\n"
            "I'll guide you through creating a professional video ad. "
            "This usually takes about 5 minutes, and then our 6-agent AI pipeline "
            "will research your market, generate creative concepts, write the script, "
            "and produce your video.\n\n"
            "Let's start with the basics. **What's your business name?**"
        ),
        IntakePhase.BUSINESS: "Great! **What industry is {business_name} in?** (e.g., fitness tech, SaaS, e-commerce)",
        IntakePhase.INDUSTRY: "Perfect. **Who's your target audience?** Be specific about demographics, interests, or job titles.",
        IntakePhase.AUDIENCE: "Got it. **Which platform is this video for?**\n\n• TikTok\n• Instagram Reels\n• YouTube\n• LinkedIn\n• Facebook\n• General/Multi-platform",
        IntakePhase.PLATFORM: "**How long should the video be?**\n\n• 15 seconds (quick hook)\n• 30 seconds (standard)\n• 60 seconds (story-driven)\n• 90+ seconds (explainer)",
        IntakePhase.DURATION: "Now for the creative. **What's the ONE key message** you want viewers to remember?",
        IntakePhase.MESSAGE: "**What are your top 3 unique selling points?** What makes you different from competitors?",
        IntakePhase.USPS: "**What's your brand tone?**\n\n• Bold & energetic\n• Professional & trustworthy\n• Friendly & approachable\n• Luxurious & premium\n• Playful & fun",
        IntakePhase.TONE: "**What's your budget tier?** This affects production quality.\n\n• Standard ($199) - AI-generated visuals\n• Premium ($499) - Enhanced quality + music\n• Enterprise ($999+) - Custom everything",
        IntakePhase.BUDGET: "**Who are your main competitors?** (Optional - helps us differentiate your message)",
        IntakePhase.COMPETITORS: "Let me summarize your brief:\n\n{brief_summary}\n\n**Does this look correct?** (yes to confirm, or tell me what to change)",
        IntakePhase.CONFIRM: "Your brief is confirmed! 🎯\n\nI'm now handing you off to our 6-agent pipeline:\n\n1. 🔍 Research Agent - Analyzing your market\n2. 💡 Ideation Agent - Generating concepts\n3. ✍️ Script Agent - Writing your script\n4. ✅ Review Agent - Quality assurance\n\nThis usually takes 2-3 minutes. I'll keep you updated!",
    }
    
    # Phase transitions
    PHASE_ORDER = [
        IntakePhase.GREETING,
        IntakePhase.BUSINESS,
        IntakePhase.INDUSTRY,
        IntakePhase.AUDIENCE,
        IntakePhase.PLATFORM,
        IntakePhase.DURATION,
        IntakePhase.MESSAGE,
        IntakePhase.USPS,
        IntakePhase.TONE,
        IntakePhase.BUDGET,
        IntakePhase.COMPETITORS,
        IntakePhase.CONFIRM,
        IntakePhase.COMPLETE,
    ]
    
    # Platform normalization
    PLATFORM_MAP = {
        "tiktok": "tiktok",
        "tik tok": "tiktok",
        "instagram": "instagram",
        "ig": "instagram",
        "reels": "instagram",
        "youtube": "youtube",
        "yt": "youtube",
        "shorts": "youtube",
        "linkedin": "linkedin",
        "facebook": "facebook",
        "fb": "facebook",
        "general": "multi-platform",
        "multi": "multi-platform",
    }
    
    # Duration parsing
    DURATION_MAP = {
        "15": 15,
        "30": 30,
        "60": 60,
        "90": 90,
        "quick": 15,
        "short": 15,
        "standard": 30,
        "story": 60,
        "explainer": 90,
    }
    
    # Tone normalization
    TONE_MAP = {
        "bold": "bold",
        "energetic": "bold",
        "professional": "professional",
        "trustworthy": "professional",
        "friendly": "friendly",
        "approachable": "friendly",
        "luxury": "luxurious",
        "luxurious": "luxurious",
        "premium": "luxurious",
        "playful": "playful",
        "fun": "playful",
    }
    
    # Budget tier parsing
    BUDGET_MAP = {
        "standard": "standard",
        "199": "standard",
        "premium": "premium",
        "499": "premium",
        "enterprise": "enterprise",
        "999": "enterprise",
        "custom": "enterprise",
    }
    
    def __init__(self):
        pass
    
    def process_message(
        self,
        session: CreativeDirectorSession,
        message: str,
    ) -> Dict[str, Any]:
        """
        Process user message and advance intake conversation.
        
        Args:
            session: Current Creative Director session
            message: User's message
            
        Returns:
            Dict with response, new phase, and completion status
        """
        current_phase = session.intake_phase
        brief = session.brief
        message_lower = message.lower().strip()
        
        # Handle greeting (start)
        if current_phase == IntakePhase.GREETING:
            session.intake_phase = IntakePhase.BUSINESS
            return self._create_response(session, IntakePhase.BUSINESS)
        
        # Process based on current phase
        if current_phase == IntakePhase.BUSINESS:
            brief.business_name = message.strip()
            session.intake_phase = IntakePhase.INDUSTRY
            return self._create_response(session, IntakePhase.INDUSTRY)
        
        elif current_phase == IntakePhase.INDUSTRY:
            brief.industry = message.strip()
            session.intake_phase = IntakePhase.AUDIENCE
            return self._create_response(session, IntakePhase.AUDIENCE)
        
        elif current_phase == IntakePhase.AUDIENCE:
            brief.target_audience = message.strip()
            session.intake_phase = IntakePhase.PLATFORM
            return self._create_response(session, IntakePhase.PLATFORM)
        
        elif current_phase == IntakePhase.PLATFORM:
            # Normalize platform
            for key, value in self.PLATFORM_MAP.items():
                if key in message_lower:
                    brief.target_platform = value
                    break
            else:
                brief.target_platform = message.strip().lower()
            
            session.intake_phase = IntakePhase.DURATION
            return self._create_response(session, IntakePhase.DURATION)
        
        elif current_phase == IntakePhase.DURATION:
            # Parse duration
            for key, value in self.DURATION_MAP.items():
                if key in message_lower:
                    brief.video_duration = value
                    break
            else:
                try:
                    brief.video_duration = int(''.join(filter(str.isdigit, message)) or 30)
                except:
                    brief.video_duration = 30
            
            session.intake_phase = IntakePhase.MESSAGE
            return self._create_response(session, IntakePhase.MESSAGE)
        
        elif current_phase == IntakePhase.MESSAGE:
            brief.key_message = message.strip()
            session.intake_phase = IntakePhase.USPS
            return self._create_response(session, IntakePhase.USPS)
        
        elif current_phase == IntakePhase.USPS:
            # Parse USPs (comma or newline separated)
            usps = [u.strip() for u in message.replace('\n', ',').split(',') if u.strip()]
            brief.unique_selling_points = usps[:5]  # Max 5
            session.intake_phase = IntakePhase.TONE
            return self._create_response(session, IntakePhase.TONE)
        
        elif current_phase == IntakePhase.TONE:
            # Normalize tone
            for key, value in self.TONE_MAP.items():
                if key in message_lower:
                    brief.brand_tone = value
                    break
            else:
                brief.brand_tone = "professional"
            
            session.intake_phase = IntakePhase.BUDGET
            return self._create_response(session, IntakePhase.BUDGET)
        
        elif current_phase == IntakePhase.BUDGET:
            # Parse budget tier
            for key, value in self.BUDGET_MAP.items():
                if key in message_lower:
                    brief.budget_tier = value
                    break
            else:
                brief.budget_tier = "standard"
            
            session.intake_phase = IntakePhase.COMPETITORS
            return self._create_response(session, IntakePhase.COMPETITORS)
        
        elif current_phase == IntakePhase.COMPETITORS:
            # Parse competitors (comma separated)
            if message_lower not in ["skip", "none", "n/a", ""]:
                competitors = [c.strip() for c in message.split(',') if c.strip()]
                brief.competitors = competitors[:5]  # Max 5
            
            session.intake_phase = IntakePhase.CONFIRM
            return self._create_response(session, IntakePhase.CONFIRM)
        
        elif current_phase == IntakePhase.CONFIRM:
            # Check confirmation
            if any(word in message_lower for word in ["yes", "correct", "confirm", "looks good", "perfect"]):
                brief.is_complete = True
                session.intake_phase = IntakePhase.COMPLETE
                session.workflow_phase = WorkflowPhase.RESEARCH
                
                return {
                    "response": self.PHASE_PROMPTS[IntakePhase.CONFIRM],
                    "phase": IntakePhase.COMPLETE.value,
                    "is_complete": True,
                    "brief": brief.to_dict(),
                    "next_action": "start_pipeline",
                }
            else:
                # User wants to change something
                return {
                    "response": "No problem! What would you like to change? Just tell me which field and the new value.",
                    "phase": IntakePhase.CONFIRM.value,
                    "is_complete": False,
                    "brief": brief.to_dict(),
                }
        
        # Default fallback
        return {
            "response": "I didn't quite catch that. Could you rephrase?",
            "phase": session.intake_phase.value,
            "is_complete": False,
        }
    
    def _create_response(
        self,
        session: CreativeDirectorSession,
        next_phase: IntakePhase,
    ) -> Dict[str, Any]:
        """Create response for phase transition"""
        prompt_template = self.PHASE_PROMPTS.get(next_phase, "")
        
        # Format with brief data
        brief = session.brief
        prompt = prompt_template.format(
            business_name=brief.business_name or "your business",
            brief_summary=self._format_brief_summary(brief),
        )
        
        return {
            "response": prompt,
            "phase": next_phase.value,
            "is_complete": False,
            "brief": brief.to_dict(),
        }
    
    def _format_brief_summary(self, brief: BriefData) -> str:
        """Format brief for confirmation"""
        usps = "\n".join(f"  • {usp}" for usp in brief.unique_selling_points) if brief.unique_selling_points else "  (none specified)"
        competitors = ", ".join(brief.competitors) if brief.competitors else "(none specified)"
        
        return f"""
**Business:** {brief.business_name}
**Industry:** {brief.industry}
**Target Audience:** {brief.target_audience}
**Platform:** {brief.target_platform}
**Duration:** {brief.video_duration} seconds
**Key Message:** {brief.key_message}
**USPs:**
{usps}
**Tone:** {brief.brand_tone}
**Budget Tier:** {brief.budget_tier}
**Competitors:** {competitors}
"""


# =============================================================================
# SESSION MANAGER
# =============================================================================

class CreativeSessionManager:
    """
    Manages Creative Director sessions.
    
    Handles session lifecycle, persistence, and cleanup.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.sessions: Dict[str, CreativeDirectorSession] = {}
        self.intake_engine = IntakeConversationEngine()
    
    def create_session(
        self,
        nexus_session_id: str,
        user_id: str,
    ) -> CreativeDirectorSession:
        """Create new Creative Director session"""
        session_id = f"cd-{nexus_session_id}"
        
        session = CreativeDirectorSession(
            session_id=session_id,
            user_id=user_id,
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created Creative Director session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[CreativeDirectorSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_session_by_nexus_id(self, nexus_session_id: str) -> Optional[CreativeDirectorSession]:
        """Get session by NEXUS session ID"""
        session_id = f"cd-{nexus_session_id}"
        return self.sessions.get(session_id)
    
    async def process_intake_message(
        self,
        session_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """Process intake conversation message"""
        session = self.get_session(session_id)
        
        if not session:
            logger.error(f"Session not found: {session_id}")
            return {
                "response": "Session not found. Let's start over.",
                "error": True,
            }
        
        # Add user message to history
        session.add_message("user", message)
        
        # Process through intake engine
        result = self.intake_engine.process_message(session, message)
        
        # Add assistant response to history
        session.add_message("assistant", result["response"])
        
        return result
    
    def close_session(self, session_id: str):
        """Close and cleanup session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed Creative Director session: {session_id}")
    
    def get_active_sessions(self) -> List[CreativeDirectorSession]:
        """Get all active sessions"""
        return [s for s in self.sessions.values() if s.is_active]
