"""
================================================================================
NEXUS - Creative Intent Detector
================================================================================
Detects when users want to create video content and routes to Creative Director.

Intent Categories:
- VIDEO_CREATION: User wants to make a video ad/commercial
- VIDEO_INQUIRY: User asking about video services
- INTAKE_CONTINUATION: User in active Creative Director session
- GENERAL: Standard NEXUS assistant query

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("nexus.creative_director.intent")


class CreativeIntent(Enum):
    """Creative Director intent categories"""
    VIDEO_CREATION = "video_creation"       # User wants to create a video
    VIDEO_INQUIRY = "video_inquiry"         # User asking about video services
    INTAKE_ACTIVE = "intake_active"         # User in intake conversation
    PRODUCTION_ACTIVE = "production_active" # User has active production
    GENERAL = "general"                     # Route to standard NEXUS


@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: CreativeIntent
    confidence: float
    triggers: List[str]
    suggested_response: Optional[str] = None
    should_handoff: bool = False


class CreativeIntentDetector:
    """
    Detects Creative Director intents from user messages.
    
    Uses pattern matching + keyword analysis to identify when users
    want to create video content vs general queries.
    """
    
    # High-confidence video creation triggers
    VIDEO_CREATION_PATTERNS = [
        r"(?:create|make|build|generate|produce)\s+(?:a\s+)?(?:video|commercial|ad|advertisement)",
        r"(?:i\s+)?(?:want|need|looking\s+for)\s+(?:a\s+)?(?:video|commercial|ad)",
        r"video\s+(?:ad|commercial|content)\s+for\s+my",
        r"(?:start|begin|let's\s+do)\s+(?:the\s+)?(?:video|commercial|ad)",
        r"neural\s+ad\s+forge",
        r"make\s+me\s+a\s+(?:video|ad|commercial)",
        r"help\s+me\s+(?:create|make)\s+(?:a\s+)?(?:video|ad)",
    ]
    
    # Video inquiry patterns (asking about services)
    VIDEO_INQUIRY_PATTERNS = [
        r"how\s+(?:does|do)\s+(?:video|ad|commercial)\s+(?:creation|generation|production)",
        r"what\s+(?:is|are)\s+(?:your|the)\s+video",
        r"tell\s+me\s+(?:about|more\s+about)\s+(?:video|ad|commercial)",
        r"pricing\s+(?:for|on)\s+(?:video|ad|commercial)",
        r"how\s+(?:long|much)\s+(?:does|do)\s+(?:video|ad)",
        r"(?:ragnarok|video\s+pipeline|ad\s+generator)",
    ]
    
    # Keywords that boost video intent
    VIDEO_KEYWORDS = {
        "video": 0.3,
        "commercial": 0.4,
        "ad": 0.2,
        "advertisement": 0.3,
        "tiktok": 0.2,
        "youtube": 0.2,
        "instagram": 0.2,
        "reels": 0.2,
        "shorts": 0.2,
        "veo": 0.4,
        "sora": 0.4,
        "script": 0.2,
        "storyboard": 0.3,
        "footage": 0.3,
        "b-roll": 0.3,
        "voiceover": 0.2,
        "30 second": 0.3,
        "60 second": 0.3,
        "product demo": 0.3,
        "explainer": 0.3,
    }
    
    # Offer-specific triggers
    OFFER_TRIGGERS = {
        "neural ad forge": CreativeIntent.VIDEO_CREATION,
        "marketing overlord": CreativeIntent.VIDEO_INQUIRY,
        "cinesite autopilot": CreativeIntent.VIDEO_INQUIRY,
        "total command": CreativeIntent.VIDEO_INQUIRY,
    }
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self._compiled_creation = [re.compile(p, re.IGNORECASE) for p in self.VIDEO_CREATION_PATTERNS]
        self._compiled_inquiry = [re.compile(p, re.IGNORECASE) for p in self.VIDEO_INQUIRY_PATTERNS]
    
    def detect(
        self,
        message: str,
        session_context: Optional[Dict] = None,
    ) -> IntentResult:
        """
        Detect creative intent from user message.
        
        Args:
            message: User's message
            session_context: Optional context with session state
            
        Returns:
            IntentResult with detected intent and confidence
        """
        message_lower = message.lower().strip()
        triggers = []
        confidence = 0.0
        
        # Check if user is in active Creative Director session
        if session_context:
            cd_mode = session_context.get("creative_director_mode")
            cd_phase = session_context.get("creative_director_phase")
            
            if cd_mode == "intake":
                return IntentResult(
                    intent=CreativeIntent.INTAKE_ACTIVE,
                    confidence=1.0,
                    triggers=["active_intake_session"],
                    should_handoff=True,
                )
            elif cd_mode == "production":
                return IntentResult(
                    intent=CreativeIntent.PRODUCTION_ACTIVE,
                    confidence=1.0,
                    triggers=["active_production"],
                    should_handoff=True,
                )
        
        # Check offer-specific triggers first
        for offer, intent in self.OFFER_TRIGGERS.items():
            if offer in message_lower:
                triggers.append(f"offer:{offer}")
                confidence = 0.9
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    triggers=triggers,
                    should_handoff=(intent == CreativeIntent.VIDEO_CREATION),
                )
        
        # Check video creation patterns (high confidence)
        for pattern in self._compiled_creation:
            match = pattern.search(message_lower)
            if match:
                triggers.append(f"pattern:{match.group()}")
                confidence = max(confidence, 0.85)
        
        # Check video inquiry patterns (medium confidence)
        for pattern in self._compiled_inquiry:
            match = pattern.search(message_lower)
            if match:
                triggers.append(f"inquiry:{match.group()}")
                confidence = max(confidence, 0.7)
        
        # Keyword scoring
        keyword_score = 0.0
        for keyword, weight in self.VIDEO_KEYWORDS.items():
            if keyword in message_lower:
                keyword_score += weight
                triggers.append(f"keyword:{keyword}")
        
        confidence = min(1.0, confidence + (keyword_score * 0.5))
        
        # Determine final intent
        if confidence >= self.confidence_threshold:
            # Check if it's creation vs inquiry
            has_creation_trigger = any(t.startswith("pattern:") for t in triggers)
            
            if has_creation_trigger or confidence >= 0.85:
                intent = CreativeIntent.VIDEO_CREATION
                should_handoff = True
                suggested_response = self._get_handoff_response()
            else:
                intent = CreativeIntent.VIDEO_INQUIRY
                should_handoff = False
                suggested_response = None
        else:
            intent = CreativeIntent.GENERAL
            should_handoff = False
            suggested_response = None
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            triggers=triggers,
            suggested_response=suggested_response,
            should_handoff=should_handoff,
        )
    
    def _get_handoff_response(self) -> str:
        """Get response for handoff to Creative Director"""
        return (
            "I'd be happy to help you create a video! 🎬\n\n"
            "I'm switching you to our AI Creative Director mode. "
            "I'll guide you through a quick intake to understand your needs, "
            "then our 6-agent pipeline will research your market, generate concepts, "
            "write the script, and produce your video.\n\n"
            "Let's start with your business. **What's your company name?**"
        )
    
    def should_activate_creative_director(
        self,
        message: str,
        session_context: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Simple check if Creative Director should be activated.
        
        Returns:
            Tuple of (should_activate, handoff_message)
        """
        result = self.detect(message, session_context)
        
        if result.should_handoff:
            return True, result.suggested_response
        
        return False, None


# =============================================================================
# QUICK USAGE
# =============================================================================

if __name__ == "__main__":
    detector = CreativeIntentDetector()
    
    test_messages = [
        "I want to create a video ad for my fitness app",
        "How much does video creation cost?",
        "Tell me about your services",
        "Make me a 30 second commercial for TikTok",
        "What is the Neural Ad Forge?",
        "Help me with my marketing",
        "I need a video for my product launch",
    ]
    
    print("\n🎬 Creative Intent Detection Test\n" + "="*50)
    
    for msg in test_messages:
        result = detector.detect(msg)
        icon = "🎬" if result.should_handoff else "📋" if result.intent != CreativeIntent.GENERAL else "💬"
        print(f"\n{icon} \"{msg}\"")
        print(f"   Intent: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Handoff: {result.should_handoff}")
        if result.triggers:
            print(f"   Triggers: {result.triggers[:3]}")
