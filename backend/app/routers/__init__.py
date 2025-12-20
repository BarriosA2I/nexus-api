# Nexus Assistant Unified - Routers
import logging

logger = logging.getLogger(__name__)

from .nexus import router as nexus_router
from .ragnarok import router as ragnarok_router

# Intake router (optional - depends on voiceover module)
try:
    from .intake import router as intake_router
    INTAKE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Intake router unavailable: {e}")
    intake_router = None
    INTAKE_AVAILABLE = False

__all__ = ["nexus_router", "ragnarok_router", "intake_router", "INTAKE_AVAILABLE"]
