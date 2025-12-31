"""
Spreading Activation Retriever
Cognitive-inspired knowledge graph traversal with activation decay.

Based on: Collins & Loftus (1975) spreading activation theory
Integration: Neo4j knowledge graph via async driver

Parameters:
- Initial activation: 1.0 for seed entities
- Decay factor: 0.7^hop (spreading from source)
- Max hops: 3
- Activation threshold: 0.1 (stop spreading below)
- Semantic relevance gating on edges
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    AsyncGraphDatabase = None

from opentelemetry import trace

logger = logging.getLogger("nexus.spreading_activation")
tracer = trace.get_tracer(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    async def embed(self, text: str) -> np.ndarray:
        ...


@dataclass
class ActivatedFact:
    """A fact retrieved via spreading activation."""
    head: str
    relation: str
    tail: str
    activation_score: float
    hop_distance: int
    path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "activation_score": self.activation_score,
            "hop_distance": self.hop_distance,
            "path": self.path,
        }


@dataclass
class SpreadingActivationConfig:
    """Configuration for spreading activation retrieval."""
    max_hops: int = 3
    decay_factor: float = 0.7
    activation_threshold: float = 0.1
    max_activated_nodes: int = 100
    semantic_relevance_weight: float = 0.5
    top_k_facts: int = 20
    use_neo4j: bool = True  # False for in-memory networkx graph


class SpreadingActivationRetriever:
    """
    Cognitive-inspired graph traversal with activation decay.

    Algorithm:
    1. Seed initial entities with activation 1.0
    2. For each hop:
       - Get neighbors of activated nodes
       - Compute propagated activation: parent_activation * decay^hop * semantic_relevance
       - Filter by activation threshold
       - Track paths for explainability
    3. Return top-K activated facts sorted by score

    Supports both Neo4j (persistent) and networkx (in-memory) backends.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[Tuple[str, str]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[SpreadingActivationConfig] = None
    ):
        """
        Initialize the spreading activation retriever.

        Args:
            neo4j_uri: Neo4j connection URI (bolt://...)
            neo4j_auth: Tuple of (username, password)
            embedding_model: Model for computing semantic relevance
            config: Configuration options
        """
        self.config = config or SpreadingActivationConfig()
        self.embedder = embedding_model
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

        # Neo4j setup
        self._driver = None
        if neo4j_uri and AsyncGraphDatabase and self.config.use_neo4j:
            self._driver = AsyncGraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            logger.info(f"Neo4j driver initialized: {neo4j_uri}")

        # In-memory graph fallback
        self._memory_graph: Optional[Any] = None
        if nx is not None:
            self._memory_graph = nx.DiGraph()

    async def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j driver closed")

    @tracer.start_as_current_span("spreading_activation.retrieve")
    async def retrieve(
        self,
        query: str,
        seed_entities: List[str],
        max_hops: Optional[int] = None
    ) -> List[ActivatedFact]:
        """
        Retrieve facts via spreading activation from seed entities.

        Args:
            query: The original query (for semantic relevance scoring)
            seed_entities: Starting entities extracted from query
            max_hops: Override default max hops

        Returns:
            List of ActivatedFact sorted by activation score
        """
        span = trace.get_current_span()
        span.set_attribute("seed_entities_count", len(seed_entities))

        max_hops = max_hops or self.config.max_hops

        if not seed_entities:
            logger.warning("No seed entities provided for spreading activation")
            return []

        # Get query embedding for semantic relevance
        query_emb = await self._get_query_embedding(query)

        # Initialize activation map
        activation: Dict[str, float] = {e: 1.0 for e in seed_entities}
        paths: Dict[str, List[str]] = {e: [e] for e in seed_entities}
        hop_distances: Dict[str, int] = {e: 0 for e in seed_entities}

        # Collect all facts
        all_facts: List[ActivatedFact] = []

        for hop in range(max_hops):
            # Get nodes to spread from (above threshold)
            active_nodes = [
                n for n, a in activation.items()
                if a >= self.config.activation_threshold
            ]

            if not active_nodes:
                logger.debug(f"No active nodes at hop {hop}, stopping")
                break

            # Batch fetch neighbors
            neighbors_batch = await self._get_neighbors_batch(active_nodes)

            new_activations: Dict[str, float] = {}

            for entity in active_nodes:
                parent_activation = activation[entity]
                neighbors = neighbors_batch.get(entity, [])

                for neighbor, relation, properties in neighbors:
                    # Compute decay
                    decay = self.config.decay_factor ** (hop + 1)

                    # Compute semantic relevance to query
                    relevance = await self._compute_semantic_relevance(
                        query_emb, neighbor, relation
                    )

                    # Final propagated activation
                    propagated = parent_activation * decay * (
                        (1 - self.config.semantic_relevance_weight) +
                        self.config.semantic_relevance_weight * relevance
                    )

                    # Accumulate activation (take max from multiple paths)
                    if neighbor in new_activations:
                        new_activations[neighbor] = max(
                            new_activations[neighbor], propagated
                        )
                    else:
                        new_activations[neighbor] = propagated

                    # Track path (keep highest activation path)
                    if neighbor not in paths or propagated > activation.get(neighbor, 0):
                        paths[neighbor] = paths[entity] + [neighbor]
                        hop_distances[neighbor] = hop + 1

                    # Create fact
                    fact = ActivatedFact(
                        head=entity,
                        relation=relation,
                        tail=neighbor,
                        activation_score=propagated,
                        hop_distance=hop + 1,
                        path=paths[entity] + [neighbor]
                    )
                    all_facts.append(fact)

            # Update activation map
            for node, act in new_activations.items():
                if node not in activation or act > activation[node]:
                    activation[node] = act

            # Early termination if no significant new activations
            significant_new = [a for a in new_activations.values()
                            if a >= self.config.activation_threshold]
            if not significant_new:
                logger.debug(f"No significant activations at hop {hop + 1}, stopping")
                break

        # Sort and return top-K facts
        all_facts.sort(key=lambda f: f.activation_score, reverse=True)
        result = all_facts[:self.config.top_k_facts]

        span.set_attribute("facts_retrieved", len(result))
        span.set_attribute("total_hops", max(hop_distances.values()) if hop_distances else 0)

        logger.info(f"Spreading activation retrieved {len(result)} facts from {len(seed_entities)} seeds")
        return result

    async def _get_neighbors_batch(
        self, entities: List[str]
    ) -> Dict[str, List[Tuple[str, str, dict]]]:
        """
        Batch fetch neighbors from graph backend.

        Args:
            entities: List of entity names to fetch neighbors for

        Returns:
            Dict mapping entity -> list of (neighbor, relation, properties)
        """
        if self._driver and self.config.use_neo4j:
            return await self._get_neighbors_neo4j(entities)
        elif self._memory_graph is not None:
            return self._get_neighbors_networkx(entities)
        else:
            logger.warning("No graph backend available")
            return {}

    async def _get_neighbors_neo4j(
        self, entities: List[str]
    ) -> Dict[str, List[Tuple[str, str, dict]]]:
        """Fetch neighbors from Neo4j."""
        query = """
        UNWIND $entities AS entity
        MATCH (n {name: entity})-[r]->(m)
        RETURN entity, m.name AS neighbor, type(r) AS relation, properties(r) AS props
        UNION
        UNWIND $entities AS entity
        MATCH (n {name: entity})<-[r]-(m)
        RETURN entity, m.name AS neighbor, type(r) AS relation, properties(r) AS props
        """

        neighbors: Dict[str, List[Tuple[str, str, dict]]] = defaultdict(list)

        try:
            async with self._driver.session() as session:
                result = await session.run(query, entities=entities)
                records = await result.data()

            for record in records:
                if record.get('neighbor'):
                    neighbors[record['entity']].append(
                        (record['neighbor'], record['relation'], record.get('props') or {})
                    )
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")

        return dict(neighbors)

    def _get_neighbors_networkx(
        self, entities: List[str]
    ) -> Dict[str, List[Tuple[str, str, dict]]]:
        """Fetch neighbors from in-memory networkx graph."""
        neighbors: Dict[str, List[Tuple[str, str, dict]]] = defaultdict(list)

        for entity in entities:
            if entity not in self._memory_graph:
                continue

            # Outgoing edges
            for neighbor in self._memory_graph.successors(entity):
                edge_data = self._memory_graph.edges[entity, neighbor]
                relation = edge_data.get('relation', 'RELATED_TO')
                neighbors[entity].append((neighbor, relation, edge_data))

            # Incoming edges
            for predecessor in self._memory_graph.predecessors(entity):
                edge_data = self._memory_graph.edges[predecessor, entity]
                relation = edge_data.get('relation', 'RELATED_TO')
                neighbors[entity].append((predecessor, relation, edge_data))

        return dict(neighbors)

    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get or cache query embedding."""
        if query not in self._query_embedding_cache:
            if self.embedder:
                self._query_embedding_cache[query] = await self.embedder.embed(query)
            else:
                # Fallback: return zero vector
                self._query_embedding_cache[query] = np.zeros(1536)
        return self._query_embedding_cache[query]

    async def _compute_semantic_relevance(
        self, query_emb: np.ndarray, entity: str, relation: str
    ) -> float:
        """
        Compute semantic relevance of entity+relation to query.

        Args:
            query_emb: Query embedding vector
            entity: Entity name
            relation: Relation type

        Returns:
            Relevance score between 0 and 1
        """
        if self.embedder is None:
            return 0.5  # Neutral relevance without embedder

        # Embed the entity-relation pair
        entity_text = f"{entity} {relation}"
        entity_emb = await self.embedder.embed(entity_text)

        # Cosine similarity
        dot_product = np.dot(query_emb, entity_emb)
        norm_product = np.linalg.norm(query_emb) * np.linalg.norm(entity_emb) + 1e-8
        similarity = dot_product / norm_product

        return float(max(0, min(1, similarity)))  # Clamp to [0, 1]

    def add_fact_to_memory(
        self,
        head: str,
        relation: str,
        tail: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a fact to the in-memory graph (for networkx backend).

        Args:
            head: Source entity
            relation: Relationship type
            tail: Target entity
            properties: Additional edge properties
        """
        if self._memory_graph is None:
            logger.warning("No in-memory graph available")
            return

        edge_data = properties or {}
        edge_data['relation'] = relation

        self._memory_graph.add_edge(head, tail, **edge_data)
        logger.debug(f"Added fact: {head} --[{relation}]--> {tail}")

    def extract_seed_entities(
        self,
        query: str,
        ner_results: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract seed entities from query.

        This method accepts pre-extracted NER results or can use basic heuristics.
        For production, integrate with a proper NER service.

        Args:
            query: The original query
            ner_results: Pre-extracted named entities (optional)

        Returns:
            List of seed entity names
        """
        if ner_results:
            return [e.strip() for e in ner_results if e.strip()]

        # Basic heuristic: extract capitalized words/phrases
        # In production, use Claude or spaCy for NER
        words = query.split()
        entities = []

        for word in words:
            # Skip common words
            if word.lower() in {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                               'what', 'who', 'how', 'why', 'when', 'where'}:
                continue
            # Check if capitalized (potential proper noun)
            if word and word[0].isupper() and len(word) > 1:
                entities.append(word)

        return entities

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._query_embedding_cache.clear()
        logger.debug("Embedding cache cleared")
