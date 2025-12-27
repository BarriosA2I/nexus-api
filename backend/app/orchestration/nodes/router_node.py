"""
NEXUS BRAIN v5.0 APEX - Router Node
====================================
Second node in pipeline: selects LLM model via Thompson Sampling.

Uses complexity classification to constrain model selection:
- System 1: Haiku or Sonnet (fast, cheap)
- System 2: Sonnet or Opus (powerful, expensive)

Thompson Sampling balances exploration/exploitation for optimal selection.
"""

import logging
import time
from typing import Any, Dict

from ..state import ConversationState, ComplexityLevel
from ..thompson_router import select_model, ModelTier

logger = logging.getLogger("nexus.node.router")


async def router_node(state: ConversationState) -> Dict[str, Any]:
    """
    Router node: select LLM model via Thompson Sampling.

    Uses the complexity classification from the previous node to
    constrain model selection. Thompson Sampling balances:
    - Exploitation: Use models with high observed success
    - Exploration: Try models to learn their true performance
    - Cost: Prefer cheaper models when quality is similar

    Args:
        state: Current conversation state with complexity result

    Returns:
        State updates with router result and selected model
    """
    start_time = time.time()

    # Get complexity from previous node
    complexity = state.get("complexity")
    if complexity is None:
        # Fallback to System 2 if classifier didn't run
        logger.warning("No complexity result, defaulting to System 2")
        complexity_level = ComplexityLevel.SYSTEM_2
    else:
        complexity_level = complexity.level

    # Select model using Thompson Sampling
    router_result = select_model(complexity_level)

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        f"Model selected: {router_result.selected_model.value} "
        f"(model_id={router_result.model_id}, "
        f"sample={router_result.sample_value:.3f}) - "
        f"{router_result.reasoning}"
    )

    return {
        "router_result": router_result,
        "selected_model": router_result.model_id,
        "node_timings": {
            **state.get("node_timings", {}),
            "router": elapsed,
        },
    }


def get_model_for_complexity(complexity: ComplexityLevel) -> str:
    """
    Get default model for complexity level (non-Thompson fallback).

    Args:
        complexity: System 1 or System 2

    Returns:
        Model ID string
    """
    if complexity == ComplexityLevel.SYSTEM_1:
        return "claude-3-5-haiku-20241022"
    else:
        return "claude-sonnet-4-20250514"
