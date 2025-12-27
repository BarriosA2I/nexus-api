"""
NEXUS BRAIN v5.0 APEX - LangGraph Orchestration Module
=======================================================
Pipeline: Classifier -> Router -> RAG -> Agent -> Publisher

Features:
- LangGraph StateGraph for node-based orchestration
- Thompson Sampling with warm priors for model selection
- Complexity Classifier (System 1/2) for routing
- PostgreSQL checkpointing for state persistence
- Dual-scope RAG (barrios_a2i + detected industry)
"""

from .graph import NexusBrainOrchestrator, create_orchestrator
from .state import ConversationState
from .thompson_router import ThompsonRouter, WARM_PRIORS
from .complexity_classifier import ComplexityClassifier, ComplexityLevel

__all__ = [
    "NexusBrainOrchestrator",
    "create_orchestrator",
    "ConversationState",
    "ThompsonRouter",
    "WARM_PRIORS",
    "ComplexityClassifier",
    "ComplexityLevel",
]
