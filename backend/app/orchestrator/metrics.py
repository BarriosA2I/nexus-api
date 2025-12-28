"""
Prometheus Metrics for NEXUS SuperGraph
Production-grade observability with counters, histograms, and gauges
"""
import logging
import time
from functools import wraps
from typing import Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger("nexus.metrics")

# Check if Prometheus client is available
PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, REGISTRY
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed. Metrics disabled.")


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # ─────────────────────────────────────────────────────────────────────────
    # COUNTERS - Monotonically increasing values
    # ─────────────────────────────────────────────────────────────────────────

    INTENT_COUNTER = Counter(
        "nexus_intent_total",
        "Total intents classified by type and detection method",
        ["intent", "method"]  # method = system1_regex | system2_llm
    )

    REQUEST_COUNTER = Counter(
        "nexus_requests_total",
        "Total requests processed",
        ["subgraph", "status"]  # status = success | error | circuit_open
    )

    CIRCUIT_TRIPS = Counter(
        "nexus_circuit_trips_total",
        "Total circuit breaker trips by circuit name",
        ["circuit_name"]
    )

    LLM_CALLS = Counter(
        "nexus_llm_calls_total",
        "Total LLM API calls by model and status",
        ["model", "status"]  # status = success | error | rate_limited
    )

    RAG_RETRIEVALS = Counter(
        "nexus_rag_retrievals_total",
        "Total RAG retrieval operations",
        ["source", "status"]  # source = qdrant | cache | fallback
    )

    TOKEN_USAGE = Counter(
        "nexus_tokens_total",
        "Total tokens used by type and model",
        ["model", "token_type"]  # token_type = input | output
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HISTOGRAMS - Distribution of values
    # ─────────────────────────────────────────────────────────────────────────

    LATENCY_HISTOGRAM = Histogram(
        "nexus_request_latency_seconds",
        "Request latency distribution by subgraph",
        ["subgraph"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )

    NODE_LATENCY = Histogram(
        "nexus_node_latency_seconds",
        "Individual node latency distribution",
        ["node"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )

    COST_HISTOGRAM = Histogram(
        "nexus_request_cost_usd",
        "Cost per request distribution by model",
        ["model"],
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )

    INTENT_CONFIDENCE = Histogram(
        "nexus_intent_confidence",
        "Intent classification confidence distribution",
        ["intent"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )

    RAG_CONFIDENCE = Histogram(
        "nexus_rag_confidence",
        "RAG answer confidence distribution",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    )

    CONTEXT_LENGTH = Histogram(
        "nexus_context_length_chars",
        "Retrieved context length distribution",
        buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GAUGES - Point-in-time values
    # ─────────────────────────────────────────────────────────────────────────

    ACTIVE_SESSIONS = Gauge(
        "nexus_active_sessions",
        "Currently active user sessions"
    )

    CIRCUIT_STATE = Gauge(
        "nexus_circuit_state",
        "Circuit breaker state (0=closed, 1=half_open, 2=open)",
        ["circuit_name"]
    )

    PENDING_REQUESTS = Gauge(
        "nexus_pending_requests",
        "Requests currently in processing"
    )

    CACHE_SIZE = Gauge(
        "nexus_cache_size",
        "Current cache size by cache type",
        ["cache_type"]  # semantic | session | response
    )

    # ─────────────────────────────────────────────────────────────────────────
    # INFO - Static metadata
    # ─────────────────────────────────────────────────────────────────────────

    SYSTEM_INFO = Info(
        "nexus_system",
        "NEXUS system information"
    )

    # Set system info on startup
    SYSTEM_INFO.info({
        "version": "1.0.0",
        "environment": "production",
        "langgraph_enabled": "true",
    })

else:
    # No-op implementations when Prometheus not available
    class NoOpMetric:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def time(self): return self._timer()
        @contextmanager
        def _timer(self):
            yield

    INTENT_COUNTER = NoOpMetric()
    REQUEST_COUNTER = NoOpMetric()
    CIRCUIT_TRIPS = NoOpMetric()
    LLM_CALLS = NoOpMetric()
    RAG_RETRIEVALS = NoOpMetric()
    TOKEN_USAGE = NoOpMetric()
    LATENCY_HISTOGRAM = NoOpMetric()
    NODE_LATENCY = NoOpMetric()
    COST_HISTOGRAM = NoOpMetric()
    INTENT_CONFIDENCE = NoOpMetric()
    RAG_CONFIDENCE = NoOpMetric()
    CONTEXT_LENGTH = NoOpMetric()
    ACTIVE_SESSIONS = NoOpMetric()
    CIRCUIT_STATE = NoOpMetric()
    PENDING_REQUESTS = NoOpMetric()
    CACHE_SIZE = NoOpMetric()
    SYSTEM_INFO = NoOpMetric()


# ============================================================================
# METRIC HELPERS
# ============================================================================

def record_intent(intent: str, method: str, confidence: float):
    """Record intent classification metrics"""
    INTENT_COUNTER.labels(intent=intent, method=method).inc()
    INTENT_CONFIDENCE.labels(intent=intent).observe(confidence)


def record_request(subgraph: str, status: str, latency_seconds: float, cost_usd: float = 0):
    """Record request-level metrics"""
    REQUEST_COUNTER.labels(subgraph=subgraph, status=status).inc()
    LATENCY_HISTOGRAM.labels(subgraph=subgraph).observe(latency_seconds)
    if cost_usd > 0:
        COST_HISTOGRAM.labels(model="aggregate").observe(cost_usd)


def record_node_latency(node: str, latency_seconds: float):
    """Record individual node latency"""
    NODE_LATENCY.labels(node=node).observe(latency_seconds)


def record_llm_call(model: str, status: str, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0):
    """Record LLM API call metrics"""
    LLM_CALLS.labels(model=model, status=status).inc()
    if input_tokens > 0:
        TOKEN_USAGE.labels(model=model, token_type="input").inc(input_tokens)
    if output_tokens > 0:
        TOKEN_USAGE.labels(model=model, token_type="output").inc(output_tokens)
    if cost_usd > 0:
        COST_HISTOGRAM.labels(model=model).observe(cost_usd)


def record_circuit_trip(circuit_name: str):
    """Record circuit breaker trip"""
    CIRCUIT_TRIPS.labels(circuit_name=circuit_name).inc()


def update_circuit_state(circuit_name: str, state: str):
    """
    Update circuit breaker state gauge

    Args:
        state: "closed", "half_open", or "open"
    """
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    CIRCUIT_STATE.labels(circuit_name=circuit_name).set(state_map.get(state, 0))


def record_rag_retrieval(source: str, status: str, context_length: int = 0, confidence: float = 0):
    """Record RAG retrieval metrics"""
    RAG_RETRIEVALS.labels(source=source, status=status).inc()
    if context_length > 0:
        CONTEXT_LENGTH.observe(context_length)
    if confidence > 0:
        RAG_CONFIDENCE.observe(confidence)


# ============================================================================
# DECORATORS FOR AUTOMATIC METRICS
# ============================================================================

def metered_node(node_name: str):
    """
    Decorator to automatically record node metrics.

    Records:
    - Node latency
    - Success/error counts
    - Cost if available in result

    Usage:
        @metered_node("retrieve")
        async def retrieve_node(state: NexusState) -> dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)

                # Record latency
                latency = time.perf_counter() - start_time
                record_node_latency(node_name, latency)

                # Extract cost if present
                if isinstance(result, dict) and "model_calls" in result:
                    for call in result.get("model_calls", []):
                        record_llm_call(
                            model=call.get("model", "unknown"),
                            status="success",
                            input_tokens=call.get("input_tokens", 0),
                            output_tokens=call.get("output_tokens", 0),
                            cost_usd=call.get("cost_usd", 0),
                        )

                return result

            except Exception as e:
                latency = time.perf_counter() - start_time
                record_node_latency(node_name, latency)
                raise

        return wrapper
    return decorator


def metered_subgraph(subgraph_name: str):
    """
    Decorator to record subgraph-level metrics.

    Usage:
        @metered_subgraph("rag")
        async def run_rag(state: NexusState) -> dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            PENDING_REQUESTS.inc()
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                latency = time.perf_counter() - start_time

                cost = 0
                if isinstance(result, dict):
                    cost = result.get("total_cost_usd", 0)

                record_request(subgraph_name, "success", latency, cost)
                return result

            except Exception as e:
                latency = time.perf_counter() - start_time
                record_request(subgraph_name, "error", latency)
                raise

            finally:
                PENDING_REQUESTS.dec()

        return wrapper
    return decorator


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

def get_metrics() -> str:
    """
    Generate Prometheus metrics in text format.
    Use in a /metrics endpoint.

    Returns:
        Prometheus text format metrics
    """
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus client not installed\n"

    return generate_latest(REGISTRY).decode('utf-8')


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics"""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def setup_metrics_endpoint(app):
    """
    Add /metrics endpoint to FastAPI app.

    Usage:
        from app.orchestrator.metrics import setup_metrics_endpoint
        setup_metrics_endpoint(app)
    """
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        return Response(
            content=get_metrics(),
            media_type=get_metrics_content_type()
        )

    logger.info("Prometheus /metrics endpoint configured")
