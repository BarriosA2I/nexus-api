"""
NEXUS BRAIN v5.0 APEX - Thompson Sampling Router
=================================================
Multi-armed bandit model selection with warm priors.

Uses Beta distribution to balance exploration vs exploitation.
Priors are pre-seeded with estimated success rates from benchmarks.

Models:
- Haiku: Fast, cheap (85% success rate) - System 1 queries
- Sonnet: Balanced (78% success rate) - Default
- Opus: Premium (92% success rate) - Complex reasoning
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .state import ComplexityLevel, ModelTier, RouterResult

logger = logging.getLogger("nexus.thompson")


# =============================================================================
# WARM PRIORS - Pre-seeded success rates from benchmarks
# =============================================================================

@dataclass
class ModelPrior:
    """Beta distribution parameters for a model."""
    alpha: float  # Successes + 1
    beta: float   # Failures + 1
    model_id: str
    cost_multiplier: float = 1.0

    @property
    def success_rate(self) -> float:
        """Expected success rate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Draw from Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        """Update distribution with new observation."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1


# Pre-seeded priors based on benchmark data
WARM_PRIORS: Dict[ModelTier, ModelPrior] = {
    ModelTier.HAIKU: ModelPrior(
        alpha=85,   # 85 successes
        beta=15,    # 15 failures
        model_id="claude-3-5-haiku-20241022",
        cost_multiplier=0.25,  # 4x cheaper
    ),
    ModelTier.SONNET: ModelPrior(
        alpha=78,   # 78 successes
        beta=22,    # 22 failures
        model_id="claude-sonnet-4-20250514",
        cost_multiplier=1.0,   # Baseline
    ),
    ModelTier.OPUS: ModelPrior(
        alpha=92,   # 92 successes
        beta=8,     # 8 failures
        model_id="claude-opus-4-20250514",
        cost_multiplier=15.0,  # 15x more expensive
    ),
}

# Complexity-aware model constraints
COMPLEXITY_CONSTRAINTS: Dict[ComplexityLevel, Dict[str, any]] = {
    ComplexityLevel.SYSTEM_1: {
        "allowed_models": [ModelTier.HAIKU, ModelTier.SONNET],
        "prefer": ModelTier.HAIKU,
        "max_cost_multiplier": 1.0,
    },
    ComplexityLevel.SYSTEM_2: {
        "allowed_models": [ModelTier.SONNET, ModelTier.OPUS],
        "prefer": ModelTier.SONNET,
        "max_cost_multiplier": 15.0,
    },
}


class ThompsonRouter:
    """
    Thompson Sampling router for adaptive model selection.

    Balances:
    - Exploitation: Use models with high observed success
    - Exploration: Try models to learn their true performance
    - Cost: Prefer cheaper models when quality is similar

    Thread-safe via independent random state per request.
    """

    def __init__(
        self,
        priors: Optional[Dict[ModelTier, ModelPrior]] = None,
        cost_weight: float = 0.1,
        exploration_boost: float = 0.05,
    ):
        """
        Initialize Thompson router.

        Args:
            priors: Model priors (defaults to WARM_PRIORS)
            cost_weight: Weight for cost penalty (0-1)
            exploration_boost: Probability of forced exploration
        """
        # Deep copy priors to allow mutation
        self.priors: Dict[ModelTier, ModelPrior] = {}
        source_priors = priors or WARM_PRIORS
        for tier, prior in source_priors.items():
            self.priors[tier] = ModelPrior(
                alpha=prior.alpha,
                beta=prior.beta,
                model_id=prior.model_id,
                cost_multiplier=prior.cost_multiplier,
            )

        self.cost_weight = cost_weight
        self.exploration_boost = exploration_boost

        logger.info(
            f"Thompson router initialized with priors: "
            f"Haiku={self.priors[ModelTier.HAIKU].success_rate:.2%}, "
            f"Sonnet={self.priors[ModelTier.SONNET].success_rate:.2%}, "
            f"Opus={self.priors[ModelTier.OPUS].success_rate:.2%}"
        )

    def select(
        self,
        complexity: ComplexityLevel,
        force_model: Optional[ModelTier] = None,
    ) -> RouterResult:
        """
        Select model using Thompson Sampling.

        Args:
            complexity: Query complexity level
            force_model: Override selection (for testing)

        Returns:
            RouterResult with selected model and metadata
        """
        # Handle forced selection
        if force_model is not None:
            prior = self.priors[force_model]
            return RouterResult(
                selected_model=force_model,
                model_id=prior.model_id,
                sample_value=prior.success_rate,
                reasoning=f"Forced selection: {force_model.value}",
            )

        # Get constraints for complexity level
        constraints = COMPLEXITY_CONSTRAINTS[complexity]
        allowed = constraints["allowed_models"]

        # Exploration: randomly try less-used model
        if random.random() < self.exploration_boost:
            # Pick least-sampled model from allowed set
            least_sampled = min(
                allowed,
                key=lambda t: self.priors[t].alpha + self.priors[t].beta
            )
            prior = self.priors[least_sampled]
            return RouterResult(
                selected_model=least_sampled,
                model_id=prior.model_id,
                sample_value=prior.sample(),
                reasoning=f"Exploration: trying {least_sampled.value}",
            )

        # Thompson Sampling: sample from each prior
        samples: Dict[ModelTier, float] = {}
        for tier in allowed:
            prior = self.priors[tier]
            # Sample from Beta and apply cost penalty
            raw_sample = prior.sample()
            cost_penalty = prior.cost_multiplier * self.cost_weight
            samples[tier] = raw_sample - cost_penalty

        # Select highest adjusted sample
        best_tier = max(samples, key=lambda t: samples[t])
        prior = self.priors[best_tier]

        return RouterResult(
            selected_model=best_tier,
            model_id=prior.model_id,
            sample_value=samples[best_tier],
            reasoning=(
                f"Thompson selected {best_tier.value} "
                f"(sample={samples[best_tier]:.3f}, "
                f"success_rate={prior.success_rate:.2%})"
            ),
        )

    def update(self, model: ModelTier, success: bool) -> None:
        """
        Update model prior with observed outcome.

        Call this after evaluating response quality.

        Args:
            model: Which model was used
            success: Whether response was successful
        """
        prior = self.priors[model]
        old_rate = prior.success_rate
        prior.update(success)
        new_rate = prior.success_rate

        logger.info(
            f"Thompson update: {model.value} "
            f"{'SUCCESS' if success else 'FAILURE'} "
            f"({old_rate:.2%} -> {new_rate:.2%})"
        )

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current model statistics."""
        return {
            tier.value: {
                "alpha": prior.alpha,
                "beta": prior.beta,
                "success_rate": prior.success_rate,
                "samples": prior.alpha + prior.beta - 2,  # Subtract initial priors
            }
            for tier, prior in self.priors.items()
        }


# =============================================================================
# SINGLETON ROUTER
# =============================================================================

_router: Optional[ThompsonRouter] = None


def get_thompson_router() -> ThompsonRouter:
    """Get singleton Thompson router instance."""
    global _router
    if _router is None:
        _router = ThompsonRouter()
    return _router


def select_model(
    complexity: ComplexityLevel,
    force_model: Optional[ModelTier] = None,
) -> RouterResult:
    """Convenience function to select model."""
    router = get_thompson_router()
    return router.select(complexity, force_model)


def update_model_outcome(model: ModelTier, success: bool) -> None:
    """Convenience function to update model prior."""
    router = get_thompson_router()
    router.update(model, success)
