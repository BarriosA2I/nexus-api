"""
Adaptive Graph-of-Thoughts (AGoT)
Recursive DAG decomposition with backtracking and contradiction detection.

Based on: Graph-of-Thoughts (arXiv:2308.09687)
Purpose: +15-20% on complex multi-hop reasoning

Parameters:
- Max depth: 3 levels
- Decomposition threshold: complexity > 0.5
- Backtrack trigger: PRM score < 0.6
- Node types: THOUGHT, DECOMPOSITION, RECURSIVE, TERMINAL
"""

import asyncio
import uuid
import logging
import time
from typing import List, Dict, Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

from opentelemetry import trace

logger = logging.getLogger("nexus.graph_of_thoughts")
tracer = trace.get_tracer(__name__)


class LLMClient(Protocol):
    """Protocol for LLM generation."""
    async def generate(self, prompt: str) -> str:
        ...


class PRMClient(Protocol):
    """Protocol for Process Reward Model evaluation."""
    async def evaluate_step(
        self,
        query: str,
        step: str,
        context: Optional[str] = None
    ) -> float:
        ...


class NodeType(Enum):
    """Types of nodes in the Graph-of-Thoughts."""
    THOUGHT = "thought"           # Base reasoning unit
    DECOMPOSITION = "decomposition"  # Splits into sub-thoughts
    RECURSIVE = "recursive"       # Nested graph call
    TERMINAL = "terminal"         # Final answer node
    AGGREGATION = "aggregation"   # Combines child answers


class NodeState(Enum):
    """States of thought nodes."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BACKTRACKED = "backtracked"


@dataclass
class ThoughtNode:
    """A node in the Graph-of-Thoughts."""
    id: str
    thought: str
    node_type: NodeType
    state: NodeState = NodeState.PENDING
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    prm_score: Optional[float] = None
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "thought": self.thought,
            "node_type": self.node_type.value,
            "state": self.state.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "answer": self.answer,
            "prm_score": self.prm_score,
            "depth": self.depth,
        }


@dataclass
class GraphResult:
    """Result from Graph-of-Thoughts solving."""
    answer: str
    confidence: float
    reasoning_trace: List[str]
    nodes_explored: int
    backtracks: int
    max_depth_reached: int
    elapsed_ms: float
    graph_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace,
            "nodes_explored": self.nodes_explored,
            "backtracks": self.backtracks,
            "max_depth_reached": self.max_depth_reached,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class AGoTConfig:
    """Configuration for Adaptive Graph-of-Thoughts."""
    max_depth: int = 3
    decomposition_threshold: float = 0.5  # Complexity above this triggers decomposition
    backtrack_threshold: float = 0.6      # PRM below this triggers backtrack
    max_children: int = 3                 # Max sub-thoughts per decomposition
    max_backtracks: int = 5               # Max backtrack attempts
    contradiction_detection: bool = True
    parallel_exploration: bool = True
    timeout_seconds: float = 30.0


@dataclass
class Checkpoint:
    """Checkpoint for backtracking."""
    node_id: str
    state_snapshot: Dict[str, Dict[str, Any]]
    timestamp: float

    @classmethod
    def create(
        cls,
        node_id: str,
        nodes: Dict[str, ThoughtNode]
    ) -> "Checkpoint":
        """Create a checkpoint from current state."""
        state_snapshot = {
            node_id: node.to_dict()
            for node_id, node in nodes.items()
        }
        return cls(
            node_id=node_id,
            state_snapshot=state_snapshot,
            timestamp=time.time()
        )


class AdaptiveGraphOfThoughts:
    """
    Recursive DAG decomposition with backtracking.

    Algorithm:
    1. Evaluate thought complexity
    2. If simple: generate answer directly (TERMINAL)
    3. If decomposable: split into sub-thoughts (DECOMPOSITION)
    4. If recursive: nested graph call (RECURSIVE)
    5. Verify each node with PRM
    6. Backtrack on PRM < threshold or contradiction
    7. Aggregate terminal nodes into final answer
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prm_client: Optional[PRMClient] = None,
        complexity_classifier: Optional[Callable[[str], float]] = None,
        config: Optional[AGoTConfig] = None
    ):
        """
        Initialize Graph-of-Thoughts solver.

        Args:
            llm_client: LLM for generation
            prm_client: Process Reward Model for verification (optional)
            complexity_classifier: Function returning complexity score 0-1
            config: Configuration options
        """
        self.llm = llm_client
        self.prm = prm_client
        self.complexity_fn = complexity_classifier or self._default_complexity
        self.config = config or AGoTConfig()

        # State (reset per solve)
        self.nodes: Dict[str, ThoughtNode] = {}
        self._graph: Optional[Any] = None
        if NETWORKX_AVAILABLE:
            self._graph = nx.DiGraph()
        self.checkpoints: deque = deque(maxlen=10)
        self.backtrack_count = 0

    @tracer.start_as_current_span("agot.solve")
    async def solve(
        self,
        query: str,
        context: Optional[str] = None,
        depth: int = 0
    ) -> GraphResult:
        """
        Solve a query using Graph-of-Thoughts reasoning.

        Args:
            query: The question/thought to process
            context: Optional context from retrieval
            depth: Current recursion depth

        Returns:
            GraphResult with answer and reasoning trace
        """
        span = trace.get_current_span()
        span.set_attribute("query_length", len(query))
        span.set_attribute("depth", depth)

        start_time = time.time()

        # Reset state for new query (except in recursive calls)
        if depth == 0:
            self.nodes = {}
            if NETWORKX_AVAILABLE:
                self._graph = nx.DiGraph()
            self.checkpoints.clear()
            self.backtrack_count = 0

        # Create root node
        root_id = self._create_node(query, NodeType.THOUGHT, depth=depth)

        try:
            # Process the graph with timeout
            await asyncio.wait_for(
                self._process_node(root_id, context),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Graph-of-Thoughts timeout after {self.config.timeout_seconds}s")
            root = self.nodes[root_id]
            if root.answer is None:
                root.answer = "Processing timed out. Please try a simpler query."
                root.state = NodeState.FAILED

        elapsed_ms = (time.time() - start_time) * 1000

        # Aggregate results
        result = self._aggregate_results(root_id, elapsed_ms)

        span.set_attribute("nodes_explored", result.nodes_explored)
        span.set_attribute("backtracks", result.backtracks)
        span.set_attribute("confidence", result.confidence)

        logger.info(
            f"AGoT completed: {result.nodes_explored} nodes, "
            f"{result.backtracks} backtracks, {elapsed_ms:.1f}ms"
        )

        return result

    async def _process_node(
        self,
        node_id: str,
        context: Optional[str] = None
    ) -> None:
        """Process a single node in the graph."""
        node = self.nodes[node_id]
        node.state = NodeState.PROCESSING

        # Save checkpoint before processing
        self._save_checkpoint(node_id)

        # Evaluate complexity
        complexity = await self._evaluate_complexity(node.thought, context)

        logger.debug(f"Node {node_id}: complexity={complexity}, depth={node.depth}")

        if complexity == 'simple' or node.depth >= self.config.max_depth:
            # Generate answer directly
            await self._process_terminal(node, context)

        elif complexity == 'decomposable':
            # Decompose into sub-thoughts
            await self._process_decomposition(node, context)

        elif complexity == 'recursive' and node.depth < self.config.max_depth:
            # Nested graph call
            await self._process_recursive(node, context)

        else:
            # Default to terminal
            await self._process_terminal(node, context)

        # Verify with PRM if available
        if node.answer and node.prm_score is None and self.prm:
            node.prm_score = await self._verify_with_prm(node)

        # Check for backtrack trigger
        if node.prm_score and node.prm_score < self.config.backtrack_threshold:
            await self._handle_backtrack(node_id)

        # Check for contradictions
        if self.config.contradiction_detection and self._detect_contradiction(node_id):
            await self._handle_backtrack(node_id)

    async def _process_terminal(
        self,
        node: ThoughtNode,
        context: Optional[str]
    ) -> None:
        """Process a terminal (answer-generating) node."""
        node.node_type = NodeType.TERMINAL

        prompt = f"""Answer the following thought directly and concisely.

Thought: {node.thought}
{"Context: " + context[:2000] if context else ""}

Provide a clear, focused answer in 1-3 sentences:"""

        try:
            node.answer = await self.llm.generate(prompt)
            node.state = NodeState.COMPLETED
        except Exception as e:
            logger.error(f"Terminal generation failed: {e}")
            node.answer = "Unable to generate answer."
            node.state = NodeState.FAILED

    async def _process_decomposition(
        self,
        node: ThoughtNode,
        context: Optional[str]
    ) -> None:
        """Decompose thought into sub-thoughts."""
        node.node_type = NodeType.DECOMPOSITION

        prompt = f"""Break down this complex thought into {self.config.max_children} simpler sub-questions that, when answered, will help answer the original question.

Original thought: {node.thought}

Return exactly {self.config.max_children} sub-questions, one per line. Each should be self-contained and answerable independently:"""

        try:
            response = await self.llm.generate(prompt)
            sub_thoughts = [
                s.strip().lstrip('0123456789.-) ')
                for s in response.strip().split('\n')
                if s.strip() and not s.strip().startswith('#')
            ][:self.config.max_children]

            if not sub_thoughts:
                # Fallback to terminal if decomposition fails
                await self._process_terminal(node, context)
                return

            # Create child nodes
            child_tasks = []
            for sub in sub_thoughts:
                child_id = self._create_node(
                    sub,
                    NodeType.THOUGHT,
                    parent_id=node.id,
                    depth=node.depth + 1
                )
                node.children_ids.append(child_id)
                if self._graph is not None:
                    self._graph.add_edge(node.id, child_id)
                child_tasks.append(self._process_node(child_id, context))

            # Process children (parallel or sequential)
            if self.config.parallel_exploration:
                await asyncio.gather(*child_tasks, return_exceptions=True)
            else:
                for task in child_tasks:
                    await task

            # Aggregate children answers
            await self._aggregate_children(node)

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            await self._process_terminal(node, context)

    async def _process_recursive(
        self,
        node: ThoughtNode,
        context: Optional[str]
    ) -> None:
        """Process with nested graph call."""
        node.node_type = NodeType.RECURSIVE

        # Recursive solve
        nested_result = await self.solve(
            node.thought,
            context,
            depth=node.depth + 1
        )

        node.answer = nested_result.answer
        node.prm_score = nested_result.confidence
        node.state = NodeState.COMPLETED

    async def _aggregate_children(self, node: ThoughtNode) -> None:
        """Aggregate answers from child nodes."""
        child_answers = []
        for child_id in node.children_ids:
            child = self.nodes[child_id]
            if child.answer and child.state == NodeState.COMPLETED:
                child_answers.append(f"- {child.thought}: {child.answer}")

        if not child_answers:
            node.answer = "Unable to synthesize answer from sub-questions."
            node.state = NodeState.FAILED
            return

        prompt = f"""Synthesize these sub-answers into a coherent response to the original question.

Original question: {node.thought}

Sub-answers:
{chr(10).join(child_answers)}

Provide a synthesized answer that combines the insights:"""

        try:
            node.answer = await self.llm.generate(prompt)
            node.state = NodeState.COMPLETED
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Use first child answer as fallback
            first_child = self.nodes.get(node.children_ids[0])
            node.answer = first_child.answer if first_child else "Unable to aggregate."
            node.state = NodeState.COMPLETED

    async def _verify_with_prm(self, node: ThoughtNode) -> float:
        """Verify node answer with Process Reward Model."""
        if self.prm is None:
            return 0.7  # Default moderate confidence

        try:
            score = await self.prm.evaluate_step(
                query=node.thought,
                step=node.answer or "",
                context=None
            )
            return score
        except Exception as e:
            logger.error(f"PRM verification failed: {e}")
            return 0.5  # Neutral score on error

    async def _evaluate_complexity(
        self,
        thought: str,
        context: Optional[str]
    ) -> str:
        """Evaluate thought complexity."""
        complexity_score = self.complexity_fn(thought)

        if complexity_score < 0.33:
            return 'simple'
        elif complexity_score < self.config.decomposition_threshold:
            return 'moderate'
        elif complexity_score < 0.8:
            return 'decomposable'
        else:
            return 'recursive'

    def _default_complexity(self, thought: str) -> float:
        """Default complexity classifier."""
        score = 0.0

        # Length factor
        words = len(thought.split())
        score += min(words / 50, 0.3)

        # Reasoning keywords (System 2 triggers)
        reasoning_words = [
            'why', 'how', 'compare', 'analyze', 'explain',
            'evaluate', 'contrast', 'differentiate', 'synthesize',
            'implications', 'consequences', 'relationship'
        ]
        for word in reasoning_words:
            if word in thought.lower():
                score += 0.1

        # Multiple questions
        if thought.count('?') > 1:
            score += 0.2

        # Conjunctions suggesting multi-part
        if any(c in thought.lower() for c in [' and ', ' or ', ' but ', ' then ', ' also ']):
            score += 0.1

        # Conditional keywords
        if any(c in thought.lower() for c in ['if ', 'unless', 'assuming', 'given that']):
            score += 0.15

        return min(score, 1.0)

    def _detect_contradiction(self, node_id: str) -> bool:
        """Detect logical contradictions in the graph."""
        node = self.nodes[node_id]

        if not node.answer:
            return False

        # Check against sibling nodes
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent is None:
                return False

            for sibling_id in parent.children_ids:
                if sibling_id != node_id:
                    sibling = self.nodes.get(sibling_id)
                    if sibling and sibling.answer:
                        # Simple contradiction detection
                        negation_pairs = [
                            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
                            ('possible', 'impossible'), ('can', 'cannot'),
                            ('always', 'never'), ('all', 'none')
                        ]
                        node_lower = node.answer.lower()
                        sibling_lower = sibling.answer.lower()

                        for pos, neg in negation_pairs:
                            if (pos in node_lower and neg in sibling_lower) or \
                               (neg in node_lower and pos in sibling_lower):
                                logger.warning(
                                    f"Contradiction detected between {node_id} and {sibling_id}"
                                )
                                return True

        return False

    def _save_checkpoint(self, node_id: str) -> None:
        """Save checkpoint for potential backtracking."""
        checkpoint = Checkpoint.create(node_id, self.nodes)
        self.checkpoints.append(checkpoint)

    async def _handle_backtrack(self, node_id: str) -> None:
        """Handle backtracking due to low PRM or contradiction."""
        if self.backtrack_count >= self.config.max_backtracks:
            logger.warning(f"Max backtracks ({self.config.max_backtracks}) reached")
            return

        self.backtrack_count += 1
        node = self.nodes[node_id]
        node.state = NodeState.BACKTRACKED

        logger.info(f"Backtracking node {node_id} (attempt {self.backtrack_count})")

        # Mark for regeneration with different approach
        node.answer = None
        node.prm_score = None
        node.metadata['regeneration_attempt'] = node.metadata.get('regeneration_attempt', 0) + 1
        node.metadata['backtrack_reason'] = 'low_prm_score' if node.prm_score else 'contradiction'

    def _create_node(
        self,
        thought: str,
        node_type: NodeType,
        parent_id: Optional[str] = None,
        depth: int = 0
    ) -> str:
        """Create a new thought node."""
        node_id = str(uuid.uuid4())[:8]
        node = ThoughtNode(
            id=node_id,
            thought=thought,
            node_type=node_type,
            parent_id=parent_id,
            depth=depth
        )
        self.nodes[node_id] = node
        if self._graph is not None:
            self._graph.add_node(node_id)
        return node_id

    def _aggregate_results(self, root_id: str, elapsed_ms: float) -> GraphResult:
        """Aggregate final results from the graph."""
        root = self.nodes[root_id]

        # Collect reasoning trace
        trace_list = []
        if NETWORKX_AVAILABLE and self._graph is not None:
            try:
                for node_id in nx.topological_sort(self._graph):
                    node = self.nodes.get(node_id)
                    if node and node.answer:
                        answer_preview = node.answer[:80] + "..." if len(node.answer) > 80 else node.answer
                        trace_list.append(
                            f"[{node.node_type.value}] {node.thought[:50]}... -> {answer_preview}"
                        )
            except nx.NetworkXError:
                # Graph has cycles, use simple iteration
                for node in self.nodes.values():
                    if node.answer:
                        answer_preview = node.answer[:80] + "..." if len(node.answer) > 80 else node.answer
                        trace_list.append(
                            f"[{node.node_type.value}] {node.thought[:50]}... -> {answer_preview}"
                        )
        else:
            for node in self.nodes.values():
                if node.answer:
                    answer_preview = node.answer[:80] + "..." if len(node.answer) > 80 else node.answer
                    trace_list.append(
                        f"[{node.node_type.value}] {node.thought[:50]}... -> {answer_preview}"
                    )

        # Calculate confidence
        completed_nodes = [
            n for n in self.nodes.values()
            if n.state == NodeState.COMPLETED and n.prm_score
        ]
        avg_confidence = (
            sum(n.prm_score for n in completed_nodes) / len(completed_nodes)
            if completed_nodes else 0.5
        )

        # Get max depth
        max_depth = max((n.depth for n in self.nodes.values()), default=0)

        # Build graph data for visualization
        graph_data = None
        if NETWORKX_AVAILABLE and self._graph is not None:
            graph_data = {
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": list(self._graph.edges()),
            }

        return GraphResult(
            answer=root.answer or "Unable to generate answer",
            confidence=avg_confidence,
            reasoning_trace=trace_list,
            nodes_explored=len(self.nodes),
            backtracks=self.backtrack_count,
            max_depth_reached=max_depth,
            elapsed_ms=elapsed_ms,
            graph_data=graph_data
        )

    def get_reasoning_graph(self) -> Optional[Dict[str, Any]]:
        """Get the reasoning graph for visualization."""
        if NETWORKX_AVAILABLE and self._graph is not None:
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "thought": node.thought,
                        "type": node.node_type.value,
                        "state": node.state.value,
                        "answer": node.answer,
                        "prm_score": node.prm_score,
                        "depth": node.depth,
                    }
                    for node in self.nodes.values()
                ],
                "edges": [
                    {"source": u, "target": v}
                    for u, v in self._graph.edges()
                ],
            }
        return None
