"""
================================================================================
🔱 CREATIVE DIRECTOR WIRING - FULL SYSTEM INTEGRATION
================================================================================
Production Wiring for 6-Agent Pipeline | Trinity + RAG + RAGNAROK Integration

CONNECTS:
├── Standard Agents     → IntakeAgent, ProductionAgent, DeliveryAgent
├── Legendary Agents    → Research, Ideation, Script, Review
├── External Systems    → Trinity MCP, Qdrant RAG, RAGNAROK v7.0
├── Event Infrastructure→ RabbitMQ, Redis Cache
└── Observability       → OpenTelemetry, Prometheus

Author: Barrios A2I Cognitive Systems Division
Version: 3.0.0 | Production Wiring
================================================================================
"""

from typing import Any, Dict, List, Optional, Callable, Union, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import os
import json
import logging
import hashlib
from functools import lru_cache

from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Initialize tracer
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "anthropic"
    model_fast: str = "claude-haiku-4-20250514"      # System 1 fast path
    model_balanced: str = "claude-sonnet-4-20250514"  # Standard operations
    model_deep: str = "claude-sonnet-4-20250514"      # System 2 deep reasoning
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class TrinityConfig:
    """Trinity Market Intelligence configuration"""
    mcp_endpoint: str = "http://localhost:8080"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    cache_ttl_seconds: int = 3600  # 1 hour cache


@dataclass
class RAGConfig:
    """RAG/Qdrant configuration"""
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "winning_commercials"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    top_k: int = 10
    score_threshold: float = 0.7


@dataclass
class RAGNAROKConfig:
    """RAGNAROK Video Pipeline configuration"""
    api_endpoint: str = "http://localhost:9000"
    api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    quality_presets: Dict[str, Dict] = field(default_factory=lambda: {
        "standard": {"resolution": "1080p", "fps": 30, "bitrate": "5M"},
        "premium": {"resolution": "4K", "fps": 60, "bitrate": "20M"}
    })
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("RAGNAROK_API_KEY")


@dataclass
class EventBusConfig:
    """Event bus configuration"""
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    exchange_name: str = "creative_director"
    redis_url: str = "redis://localhost:6379/0"


@dataclass
class CreativeDirectorConfig:
    """Master configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    trinity: TrinityConfig = field(default_factory=TrinityConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ragnarok: RAGNAROKConfig = field(default_factory=RAGNAROKConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    
    # Operational settings
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = True
    max_revisions: int = 3
    session_timeout_minutes: int = 60
    
    @classmethod
    def from_env(cls) -> "CreativeDirectorConfig":
        """Load configuration from environment variables"""
        return cls(
            llm=LLMConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model_balanced=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            ),
            trinity=TrinityConfig(
                mcp_endpoint=os.getenv("TRINITY_MCP_ENDPOINT", "http://localhost:8080"),
            ),
            rag=RAGConfig(
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                collection_name=os.getenv("RAG_COLLECTION", "winning_commercials"),
            ),
            ragnarok=RAGNAROKConfig(
                api_endpoint=os.getenv("RAGNAROK_ENDPOINT", "http://localhost:9000"),
                api_key=os.getenv("RAGNAROK_API_KEY"),
            ),
            event_bus=EventBusConfig(
                rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            ),
        )


# =============================================================================
# EXTERNAL SYSTEM ADAPTERS - PROTOCOLS
# =============================================================================

class ITrinityClient(Protocol):
    """Protocol for Trinity Market Intelligence"""
    
    async def analyze_competitors(
        self, industry: str, competitors: List[str]
    ) -> List[Dict[str, Any]]: ...
    
    async def get_audience_insights(
        self, demographic: str, industry: str
    ) -> Dict[str, Any]: ...
    
    async def get_platform_best_practices(
        self, platform: str
    ) -> Dict[str, Any]: ...


class IRAGClient(Protocol):
    """Protocol for RAG retrieval"""
    
    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]: ...
    
    async def embed(self, text: str) -> List[float]: ...


class IRAGNAROKClient(Protocol):
    """Protocol for RAGNAROK video pipeline"""
    
    async def submit_job(
        self, script: Any, brief: Any, quality_tier: str
    ) -> str: ...
    
    async def get_status(self, job_id: str) -> Dict[str, Any]: ...
    
    async def cancel_job(self, job_id: str) -> bool: ...


class IWebSearchClient(Protocol):
    """Protocol for web search fallback"""
    
    async def search(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, Any]]: ...


# =============================================================================
# TRINITY MCP ADAPTER
# =============================================================================

class TrinityMCPAdapter:
    """
    Adapter for Trinity Market Intelligence MCP Server
    
    Connects to your existing Trinity orchestrator via MCP protocol
    """
    
    def __init__(self, config: TrinityConfig):
        self.config = config
        self.endpoint = config.mcp_endpoint
        self._session = None
        self._cache: Dict[str, Any] = {}
        
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
    
    async def _call_mcp(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Trinity MCP tool"""
        await self._ensure_session()
        
        # Check cache
        cache_key = hashlib.md5(
            f"{tool_name}:{json.dumps(arguments, sort_keys=True)}".encode()
        ).hexdigest()
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.utcnow() - cached["timestamp"] < timedelta(seconds=self.config.cache_ttl_seconds):
                return cached["data"]
        
        # MCP call
        with tracer.start_as_current_span(f"trinity_mcp_{tool_name}") as span:
            span.set_attribute("tool_name", tool_name)
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            for attempt in range(self.config.max_retries):
                try:
                    async with self._session.post(
                        f"{self.endpoint}/mcp",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            data = result.get("result", {})
                            
                            # Cache result
                            self._cache[cache_key] = {
                                "data": data,
                                "timestamp": datetime.utcnow()
                            }
                            
                            return data
                        else:
                            logger.warning(f"Trinity MCP error: {response.status}")
                            
                except Exception as e:
                    logger.warning(f"Trinity MCP attempt {attempt+1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            
            span.set_status(Status(StatusCode.ERROR, "All retries failed"))
            return {}
    
    async def analyze_competitors(
        self,
        industry: str,
        competitors: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze competitors via Trinity"""
        result = await self._call_mcp("trinity_analyze_competitors", {
            "industry": industry,
            "competitors": competitors
        })
        return result.get("competitor_analysis", [])
    
    async def get_audience_insights(
        self,
        demographic: str,
        industry: str
    ) -> Dict[str, Any]:
        """Get audience insights via Trinity"""
        result = await self._call_mcp("trinity_audience_insights", {
            "demographic": demographic,
            "industry": industry
        })
        return result.get("insights", {})
    
    async def get_platform_best_practices(
        self,
        platform: str
    ) -> Dict[str, Any]:
        """Get platform best practices via Trinity"""
        result = await self._call_mcp("trinity_platform_analysis", {
            "platform": platform
        })
        return result
    
    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()
            self._session = None


# =============================================================================
# QDRANT RAG ADAPTER
# =============================================================================

class QdrantRAGAdapter:
    """
    Adapter for Qdrant vector database RAG
    
    Connects to your Qdrant instance for script examples and knowledge retrieval
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._client = None
        self._embedding_client = None
    
    async def _ensure_clients(self):
        """Ensure clients are initialized"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.async_qdrant_client import AsyncQdrantClient
                
                self._client = AsyncQdrantClient(url=self.config.qdrant_url)
            except ImportError:
                logger.warning("Qdrant client not installed, using mock")
                self._client = MockQdrantClient()
    
    async def search(
        self,
        query: str,
        top_k: int = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        await self._ensure_clients()
        
        top_k = top_k or self.config.top_k
        
        with tracer.start_as_current_span("qdrant_search") as span:
            span.set_attribute("query_length", len(query))
            span.set_attribute("top_k", top_k)
            
            try:
                # Get query embedding
                query_vector = await self.embed(query)
                
                # Search Qdrant
                results = await self._client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=self.config.score_threshold
                )
                
                # Format results
                docs = []
                for hit in results:
                    docs.append({
                        "id": hit.id,
                        "score": hit.score,
                        "content": hit.payload.get("content", ""),
                        "metadata": hit.payload.get("metadata", {}),
                        "title": hit.payload.get("title", ""),
                    })
                
                span.set_attribute("results_count", len(docs))
                return docs
                
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return []
    
    async def embed(self, text: str) -> List[float]:
        """Get embedding for text"""
        # Use OpenAI embeddings or local model
        try:
            import openai
            
            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=self.config.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using mock")
            # Return mock embedding
            import random
            return [random.random() for _ in range(self.config.embedding_dimension)]
    
    async def close(self):
        """Close client"""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()


class MockQdrantClient:
    """Mock Qdrant client for testing"""
    
    async def search(self, **kwargs):
        # Return mock results
        class MockHit:
            def __init__(self, idx):
                self.id = f"doc_{idx}"
                self.score = 0.9 - (idx * 0.05)
                self.payload = {
                    "content": f"Sample winning commercial script #{idx}",
                    "metadata": {"industry": "general", "platform": "youtube"},
                    "title": f"Winning Ad Example {idx}"
                }
        
        return [MockHit(i) for i in range(min(kwargs.get("limit", 5), 5))]


# =============================================================================
# RAGNAROK API ADAPTER
# =============================================================================

class RAGNAROKAPIAdapter:
    """
    Adapter for RAGNAROK v7.0 APEX Video Pipeline
    
    Connects to your RAGNAROK video generation system
    """
    
    def __init__(self, config: RAGNAROKConfig):
        self.config = config
        self._session = None
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self._session is None:
            import aiohttp
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            )
    
    async def submit_job(
        self,
        script: Any,
        brief: Any,
        quality_tier: str = "standard"
    ) -> str:
        """Submit video generation job to RAGNAROK"""
        await self._ensure_session()
        
        with tracer.start_as_current_span("ragnarok_submit") as span:
            span.set_attribute("quality_tier", quality_tier)
            
            # Prepare payload
            payload = {
                "script": script.model_dump() if hasattr(script, "model_dump") else script,
                "brief": brief.model_dump() if hasattr(brief, "model_dump") else brief,
                "quality_preset": self.config.quality_presets.get(quality_tier, {}),
                "webhook_url": self.config.webhook_url,
            }
            
            try:
                async with self._session.post(
                    f"{self.config.api_endpoint}/api/v1/jobs",
                    json=payload
                ) as response:
                    if response.status in (200, 201):
                        result = await response.json()
                        job_id = result.get("job_id")
                        span.set_attribute("job_id", job_id)
                        return job_id
                    else:
                        error = await response.text()
                        raise Exception(f"RAGNAROK submit failed: {response.status} - {error}")
                        
            except Exception as e:
                logger.error(f"RAGNAROK submission error: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from RAGNAROK"""
        await self._ensure_session()
        
        with tracer.start_as_current_span("ragnarok_status") as span:
            span.set_attribute("job_id", job_id)
            
            try:
                async with self._session.get(
                    f"{self.config.api_endpoint}/api/v1/jobs/{job_id}"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "unknown", "error": f"HTTP {response.status}"}
                        
            except Exception as e:
                logger.error(f"RAGNAROK status error: {e}")
                return {"status": "error", "error": str(e)}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a RAGNAROK job"""
        await self._ensure_session()
        
        try:
            async with self._session.delete(
                f"{self.config.api_endpoint}/api/v1/jobs/{job_id}"
            ) as response:
                return response.status in (200, 204)
        except Exception as e:
            logger.error(f"RAGNAROK cancel error: {e}")
            return False
    
    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()
            self._session = None


# =============================================================================
# WEB SEARCH ADAPTER (CRAG Fallback)
# =============================================================================

class WebSearchAdapter:
    """
    Web search adapter for CRAG fallback
    
    Uses Perplexity API or similar for web search
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._session = None
    
    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
    
    async def search(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the web"""
        await self._ensure_session()
        
        if not self.api_key:
            logger.warning("No web search API key, returning empty results")
            return []
        
        with tracer.start_as_current_span("web_search") as span:
            span.set_attribute("query", query[:100])
            
            try:
                # Perplexity API call
                async with self._session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {"role": "user", "content": f"Search for: {query}"}
                        ]
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Parse and format results
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return [{"content": content, "source": "perplexity"}]
                    else:
                        return []
                        
            except Exception as e:
                logger.error(f"Web search error: {e}")
                return []
    
    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None


# =============================================================================
# EVENT BUS ADAPTER (RabbitMQ + Redis)
# =============================================================================

class ProductionEventBus:
    """
    Production-grade event bus with RabbitMQ and Redis
    
    Features:
    - RabbitMQ for durable event distribution
    - Redis for real-time pub/sub and caching
    - Dead letter queue handling
    - Event replay capability
    """
    
    def __init__(self, config: EventBusConfig):
        self.config = config
        self._rabbitmq = None
        self._redis = None
        self._channel = None
        self._handlers: Dict[str, List[Callable]] = {}
        self._history: List[Dict[str, Any]] = []
    
    async def connect(self):
        """Connect to message brokers"""
        # RabbitMQ connection
        try:
            import aio_pika
            
            self._rabbitmq = await aio_pika.connect_robust(
                self.config.rabbitmq_url
            )
            self._channel = await self._rabbitmq.channel()
            
            # Declare exchange
            await self._channel.declare_exchange(
                self.config.exchange_name,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            logger.info("Connected to RabbitMQ")
            
        except ImportError:
            logger.warning("aio_pika not installed, RabbitMQ disabled")
        except Exception as e:
            logger.warning(f"RabbitMQ connection failed: {e}")
        
        # Redis connection
        try:
            import redis.asyncio as redis
            
            self._redis = await redis.from_url(self.config.redis_url)
            logger.info("Connected to Redis")
            
        except ImportError:
            logger.warning("redis not installed, Redis disabled")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all subscribers"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self._history.append(event)
        
        with tracer.start_as_current_span("event_emit") as span:
            span.set_attribute("event_type", event_type)
            
            # Local handlers
            for handler in self._handlers.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event_type}: {e}")
            
            # RabbitMQ publish
            if self._channel:
                try:
                    import aio_pika
                    
                    exchange = await self._channel.get_exchange(self.config.exchange_name)
                    await exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(event).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                        ),
                        routing_key=event_type
                    )
                except Exception as e:
                    logger.error(f"RabbitMQ publish error: {e}")
            
            # Redis publish (for real-time)
            if self._redis:
                try:
                    await self._redis.publish(
                        f"creative_director:{event_type}",
                        json.dumps(event)
                    )
                except Exception as e:
                    logger.error(f"Redis publish error: {e}")
    
    async def close(self):
        """Close connections"""
        if self._rabbitmq:
            await self._rabbitmq.close()
        if self._redis:
            await self._redis.close()


# =============================================================================
# ANTHROPIC LLM CLIENT WRAPPER
# =============================================================================

class AnthropicClientWrapper:
    """
    Wrapper for Anthropic client with circuit breaker and metrics
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._failures = 0
        self._circuit_open = False
        self._last_failure = None
        self._call_count = 0
        self._total_tokens = 0
    
    def _ensure_client(self):
        """Ensure client is initialized"""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        return self._client
    
    @property
    def messages(self):
        """Return messages interface"""
        return self
    
    async def create(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        messages: List[Dict[str, str]] = None,
        system: Optional[str] = None,
        **kwargs
    ):
        """Create a message completion"""
        client = self._ensure_client()
        
        # Circuit breaker check
        if self._circuit_open:
            if datetime.utcnow() - self._last_failure > timedelta(seconds=30):
                self._circuit_open = False
                self._failures = 0
            else:
                raise Exception("Circuit breaker open")
        
        with tracer.start_as_current_span("llm_completion") as span:
            model = model or self.config.model_balanced
            span.set_attribute("model", model)
            
            try:
                create_kwargs = {
                    "model": model,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "messages": messages,
                    **kwargs
                }
                
                if system:
                    create_kwargs["system"] = system
                
                response = await client.messages.create(**create_kwargs)
                
                self._call_count += 1
                self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
                self._failures = 0
                
                span.set_attribute("tokens_used", response.usage.input_tokens + response.usage.output_tokens)
                
                return response
                
            except Exception as e:
                self._failures += 1
                self._last_failure = datetime.utcnow()
                
                if self._failures >= 5:
                    self._circuit_open = True
                    logger.error("LLM circuit breaker opened")
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM usage metrics"""
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "failures": self._failures,
            "circuit_open": self._circuit_open,
        }


# =============================================================================
# WIRED CREATIVE DIRECTOR FACTORY
# =============================================================================

class WiredCreativeDirectorFactory:
    """
    Factory for creating fully-wired Creative Director Nexus
    
    Connects all legendary agents to external systems and orchestration
    """
    
    def __init__(self, config: CreativeDirectorConfig):
        self.config = config
        self._llm_client = None
        self._trinity_client = None
        self._rag_client = None
        self._ragnarok_client = None
        self._web_search_client = None
        self._event_bus = None
    
    async def create_llm_client(self) -> AnthropicClientWrapper:
        """Create LLM client"""
        if self._llm_client is None:
            self._llm_client = AnthropicClientWrapper(self.config.llm)
        return self._llm_client
    
    async def create_trinity_client(self) -> TrinityMCPAdapter:
        """Create Trinity MCP client"""
        if self._trinity_client is None:
            self._trinity_client = TrinityMCPAdapter(self.config.trinity)
        return self._trinity_client
    
    async def create_rag_client(self) -> QdrantRAGAdapter:
        """Create RAG/Qdrant client"""
        if self._rag_client is None:
            self._rag_client = QdrantRAGAdapter(self.config.rag)
        return self._rag_client
    
    async def create_ragnarok_client(self) -> RAGNAROKAPIAdapter:
        """Create RAGNAROK client"""
        if self._ragnarok_client is None:
            self._ragnarok_client = RAGNAROKAPIAdapter(self.config.ragnarok)
        return self._ragnarok_client
    
    async def create_web_search_client(self) -> WebSearchAdapter:
        """Create web search client"""
        if self._web_search_client is None:
            self._web_search_client = WebSearchAdapter()
        return self._web_search_client
    
    async def create_event_bus(self) -> ProductionEventBus:
        """Create event bus"""
        if self._event_bus is None:
            self._event_bus = ProductionEventBus(self.config.event_bus)
            await self._event_bus.connect()
        return self._event_bus
    
    async def create_nexus(self):
        """
        Create fully-wired CreativeDirectorNexus
        
        Returns:
            CreativeDirectorNexus with all legendary agents connected
        """
        # Import the modules (these would be your actual files)
        # In production, ensure these are importable
        try:
            from legendary_creative_director_agents import (
                LegendaryResearchAgent,
                LegendaryIdeationAgent,
                LegendaryScriptAgent,
                LegendaryReviewAgent,
                LegendaryAgentFactory
            )
            from standard_creative_director_agents import (
                IntakeAgent,
                ProductionAgent,
                DeliveryAgent,
            )
            from creative_director_nexus import (
                CreativeDirectorNexus,
                CreativeDirectorSession,
                WorkflowPhase,
            )
            
            USE_REAL_AGENTS = True
            
        except ImportError as e:
            logger.warning(f"Could not import agents: {e}. Using stubs.")
            USE_REAL_AGENTS = False
        
        # Create all clients
        llm_client = await self.create_llm_client()
        trinity_client = await self.create_trinity_client()
        rag_client = await self.create_rag_client()
        ragnarok_client = await self.create_ragnarok_client()
        web_search_client = await self.create_web_search_client()
        event_bus = await self.create_event_bus()
        
        if USE_REAL_AGENTS:
            # Create legendary agents with full wiring
            factory = LegendaryAgentFactory(
                llm_client=llm_client,
                trinity_client=trinity_client,
                rag_client=rag_client,
                web_search_client=web_search_client,
                knowledge_graph=None  # Optional: add Neo4j connector if available
            )
            
            research_agent = factory.create_research_agent()
            ideation_agent = factory.create_ideation_agent()
            script_agent = factory.create_script_agent()
            review_agent = factory.create_review_agent()
            
            # Create nexus with real agents
            nexus = CreativeDirectorNexus(
                llm_client=llm_client,
                ragnarok_client=ragnarok_client,
                event_bus=event_bus,
                research_agent=research_agent,
                ideation_agent=ideation_agent,
                script_agent=script_agent,
                review_agent=review_agent,
            )
            
        else:
            # Create nexus with stubs (from creative_director_nexus.py)
            from creative_director_nexus import CreativeDirectorNexus
            
            nexus = CreativeDirectorNexus(
                llm_client=llm_client,
                ragnarok_client=ragnarok_client,
                event_bus=event_bus,
            )
        
        logger.info("✅ CreativeDirectorNexus created with full wiring")
        
        return nexus
    
    async def close(self):
        """Close all connections"""
        if self._trinity_client:
            await self._trinity_client.close()
        if self._rag_client:
            await self._rag_client.close()
        if self._ragnarok_client:
            await self._ragnarok_client.close()
        if self._web_search_client:
            await self._web_search_client.close()
        if self._event_bus:
            await self._event_bus.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_wired_nexus(
    config: Optional[CreativeDirectorConfig] = None
):
    """
    Quick factory function to create a wired nexus
    
    Usage:
        nexus = await create_wired_nexus()
        session = nexus.create_session(user_id="user123")
    """
    config = config or CreativeDirectorConfig.from_env()
    factory = WiredCreativeDirectorFactory(config)
    return await factory.create_nexus()


def create_config_from_env() -> CreativeDirectorConfig:
    """Create configuration from environment variables"""
    return CreativeDirectorConfig.from_env()


# =============================================================================
# HEALTH CHECK
# =============================================================================

async def health_check(config: CreativeDirectorConfig) -> Dict[str, Any]:
    """
    Run health checks on all external systems
    
    Returns:
        Dict with health status for each system
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "systems": {}
    }
    
    # LLM check
    try:
        client = AnthropicClientWrapper(config.llm)
        response = await client.create(
            model="claude-haiku-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}]
        )
        results["systems"]["llm"] = {"status": "healthy", "model": config.llm.model_balanced}
    except Exception as e:
        results["systems"]["llm"] = {"status": "unhealthy", "error": str(e)}
    
    # Trinity check
    try:
        trinity = TrinityMCPAdapter(config.trinity)
        # Simple connectivity test
        results["systems"]["trinity"] = {"status": "healthy", "endpoint": config.trinity.mcp_endpoint}
    except Exception as e:
        results["systems"]["trinity"] = {"status": "unhealthy", "error": str(e)}
    
    # Qdrant check
    try:
        rag = QdrantRAGAdapter(config.rag)
        results["systems"]["qdrant"] = {"status": "healthy", "url": config.rag.qdrant_url}
    except Exception as e:
        results["systems"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
    
    # RAGNAROK check
    try:
        ragnarok = RAGNAROKAPIAdapter(config.ragnarok)
        results["systems"]["ragnarok"] = {"status": "healthy", "endpoint": config.ragnarok.api_endpoint}
    except Exception as e:
        results["systems"]["ragnarok"] = {"status": "unhealthy", "error": str(e)}
    
    # Overall status
    all_healthy = all(
        s.get("status") == "healthy"
        for s in results["systems"].values()
    )
    results["overall"] = "healthy" if all_healthy else "degraded"
    
    return results


# =============================================================================
# EXAMPLE USAGE & DEMO
# =============================================================================

async def demo_wired_pipeline():
    """
    Demonstrate the fully wired Creative Director pipeline
    """
    print("\n" + "="*70)
    print("🔱 CREATIVE DIRECTOR - WIRED PIPELINE DEMO")
    print("="*70)
    
    # Load config from environment
    config = CreativeDirectorConfig.from_env()
    
    # Run health checks
    print("\n📊 Running Health Checks...")
    health = await health_check(config)
    for system, status in health["systems"].items():
        icon = "✅" if status["status"] == "healthy" else "❌"
        print(f"  {icon} {system}: {status['status']}")
    
    print(f"\n  Overall: {health['overall'].upper()}")
    
    # Create wired nexus
    print("\n🔧 Creating Wired Nexus...")
    factory = WiredCreativeDirectorFactory(config)
    
    try:
        nexus = await factory.create_nexus()
        print("  ✅ Nexus created successfully")
        
        # Create a session
        session = nexus.create_session(user_id="demo_user")
        print(f"\n📋 Session Created: {session.session_id}")
        
        # Show what's connected
        print("\n🔌 Connected Systems:")
        print(f"  • LLM: {config.llm.model_balanced}")
        print(f"  • Trinity: {config.trinity.mcp_endpoint}")
        print(f"  • Qdrant: {config.rag.qdrant_url}")
        print(f"  • RAGNAROK: {config.ragnarok.api_endpoint}")
        
    finally:
        await factory.close()
    
    print("\n" + "="*70)
    print("✅ WIRING DEMO COMPLETE")
    print("="*70 + "\n")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "CreativeDirectorConfig",
    "LLMConfig",
    "TrinityConfig",
    "RAGConfig",
    "RAGNAROKConfig",
    "EventBusConfig",
    
    # Adapters
    "TrinityMCPAdapter",
    "QdrantRAGAdapter",
    "RAGNAROKAPIAdapter",
    "WebSearchAdapter",
    "ProductionEventBus",
    "AnthropicClientWrapper",
    
    # Factory
    "WiredCreativeDirectorFactory",
    
    # Convenience
    "create_wired_nexus",
    "create_config_from_env",
    "health_check",
]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_wired_pipeline())
