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

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "LocalRAGService",
    "JobStore",
    "RagnarokBridge",
    "VOICEOVER_AVAILABLE",
]
