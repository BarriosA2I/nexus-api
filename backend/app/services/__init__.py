# Nexus Assistant Unified - Services
from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from .rag_local import LocalRAGService
from .job_store import JobStore
from .ragnarok_bridge import RagnarokBridge

# Voiceover service (lazy import to handle missing dependencies)
try:
    from .voiceover_service import (
        get_voiceover_agent,
        generate_voiceover_from_intake,
        get_service_health as get_voiceover_health,
    )
    VOICEOVER_AVAILABLE = True
except ImportError:
    VOICEOVER_AVAILABLE = False

# Knowledge Base (200+ stats for data-backed responses)
try:
    from .nexus_knowledge_base import (
        get_contextual_knowledge,
        get_objection_response,
        get_relevant_case_study,
        get_random_stat,
        Industry,
        ObjectionType,
        QUICK_STATS,
        INDUSTRY_USE_CASES,
        ROI_STATISTICS,
        CASE_STUDIES,
    )
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "LocalRAGService",
    "JobStore",
    "RagnarokBridge",
    "VOICEOVER_AVAILABLE",
    "KNOWLEDGE_BASE_AVAILABLE",
]
