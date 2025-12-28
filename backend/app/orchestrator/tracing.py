"""
OpenTelemetry Tracing for NEXUS SuperGraph
100% trace coverage across all nodes and subgraphs
"""
import os
import logging
from functools import wraps
from typing import Optional, Dict, Any
from contextvars import ContextVar

logger = logging.getLogger("nexus.tracing")

# Context variable for correlation ID propagation
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

# Check if OpenTelemetry is available
OTEL_AVAILABLE = False
tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not installed. Tracing disabled.")


def init_tracing(service_name: str = "nexus-supergraph") -> Optional[Any]:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Supports:
    - Jaeger (via OTLP)
    - Grafana Tempo
    - Any OTLP-compatible backend

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., http://localhost:4317)
        OTEL_SERVICE_NAME: Override service name
        OTEL_TRACES_SAMPLER: Sampling strategy (always_on, always_off, traceidratio)
        OTEL_TRACES_SAMPLER_ARG: Sampler argument (e.g., 0.1 for 10% sampling)
    """
    global tracer

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - using no-op tracer")
        return None

    # Create resource with service info
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name),
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
    })

    # Initialize provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint configured
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"OTLP exporter configured: {otlp_endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not installed. Install opentelemetry-exporter-otlp-proto-grpc")

    # Add console exporter for development
    if os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true":
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("Console span exporter enabled")
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("nexus.supergraph", "1.0.0")

    logger.info("OpenTelemetry tracing initialized")
    return tracer


def get_tracer():
    """Get the global tracer instance"""
    global tracer
    if tracer is None:
        tracer = init_tracing()
    return tracer


# ============================================================================
# DECORATORS FOR NODE TRACING
# ============================================================================

def traced_node(node_name: str):
    """
    Decorator to add OpenTelemetry tracing to graph nodes.

    Captures:
    - Session ID
    - Current intent
    - Phase (for multi-step flows)
    - Cost and latency metrics
    - Errors with full exception info

    Usage:
        @traced_node("retrieve")
        async def retrieve_node(state: NexusState) -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(state, *args, **kwargs):
            _tracer = get_tracer()

            # If no tracer, just run the function
            if _tracer is None:
                return await func(state, *args, **kwargs)

            # Extract trace context
            session_id = state.get("session_id", "unknown")
            current_intent = state.get("current_intent", "unknown")
            cd_phase = state.get("cd_phase", "none")

            with _tracer.start_as_current_span(
                f"node.{node_name}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "nexus.node": node_name,
                    "nexus.session_id": session_id,
                    "nexus.intent": current_intent,
                    "nexus.phase": cd_phase,
                    "nexus.message_count": len(state.get("messages", [])),
                }
            ) as span:
                try:
                    result = await func(state, *args, **kwargs)
                    span.set_status(Status(StatusCode.OK))

                    # Add result metrics as span attributes
                    if isinstance(result, dict):
                        if "total_cost_usd" in result:
                            span.set_attribute("nexus.cost_usd", result["total_cost_usd"])
                        if "total_latency_ms" in result:
                            span.set_attribute("nexus.latency_ms", result["total_latency_ms"])
                        if "intent_confidence" in result:
                            span.set_attribute("nexus.intent_confidence", result["intent_confidence"])
                        if "rag_confidence" in result:
                            span.set_attribute("nexus.rag_confidence", result["rag_confidence"])
                        if "errors" in result and result["errors"]:
                            span.set_attribute("nexus.errors", len(result["errors"]))
                            span.add_event("node_errors", {"errors": str(result["errors"])})

                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_subgraph(subgraph_name: str):
    """
    Decorator for tracing entire subgraph invocations.
    Creates a parent span for all nodes within the subgraph.

    Usage:
        @traced_subgraph("rag")
        async def invoke_rag_subgraph(state: NexusState) -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(state, *args, **kwargs):
            _tracer = get_tracer()

            if _tracer is None:
                return await func(state, *args, **kwargs)

            session_id = state.get("session_id", "unknown")

            with _tracer.start_as_current_span(
                f"subgraph.{subgraph_name}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "nexus.subgraph": subgraph_name,
                    "nexus.session_id": session_id,
                    "nexus.intent": state.get("current_intent", "unknown"),
                }
            ) as span:
                try:
                    result = await func(state, *args, **kwargs)
                    span.set_status(Status(StatusCode.OK))

                    # Capture subgraph-level metrics
                    if isinstance(result, dict):
                        span.set_attribute("nexus.total_cost_usd", result.get("total_cost_usd", 0))
                        span.set_attribute("nexus.total_latency_ms", result.get("total_latency_ms", 0))

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_llm_call(model: str, purpose: str = "completion"):
    """
    Decorator for tracing LLM API calls (Anthropic, OpenAI, etc.)

    Captures:
    - Model name
    - Token usage
    - Cost
    - Latency

    Usage:
        @traced_llm_call("claude-sonnet-4-20250514", purpose="generation")
        async def call_claude(messages: list) -> str:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _tracer = get_tracer()

            if _tracer is None:
                return await func(*args, **kwargs)

            with _tracer.start_as_current_span(
                f"llm.{model}",
                kind=SpanKind.CLIENT,
                attributes={
                    "llm.model": model,
                    "llm.purpose": purpose,
                    "llm.provider": "anthropic" if "claude" in model else "openai",
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))

                    # Try to extract token usage from result
                    if hasattr(result, 'usage'):
                        span.set_attribute("llm.input_tokens", result.usage.input_tokens)
                        span.set_attribute("llm.output_tokens", result.usage.output_tokens)

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_external_call(service: str):
    """
    Decorator for tracing external service calls (Qdrant, Redis, etc.)

    Usage:
        @traced_external_call("qdrant")
        async def search_vectors(query: str) -> list:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _tracer = get_tracer()

            if _tracer is None:
                return await func(*args, **kwargs)

            with _tracer.start_as_current_span(
                f"external.{service}",
                kind=SpanKind.CLIENT,
                attributes={
                    "external.service": service,
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


# ============================================================================
# CONTEXT PROPAGATION HELPERS
# ============================================================================

def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into HTTP headers for distributed tracing.
    Use when making HTTP calls to other services.
    """
    if not OTEL_AVAILABLE:
        return headers

    try:
        from opentelemetry.propagate import inject
        inject(headers)
    except Exception as e:
        logger.warning(f"Failed to inject trace context: {e}")

    return headers


def extract_trace_context(headers: Dict[str, str]):
    """
    Extract trace context from incoming HTTP headers.
    Use in middleware to continue traces from upstream services.
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        from opentelemetry.propagate import extract
        return extract(headers)
    except Exception as e:
        logger.warning(f"Failed to extract trace context: {e}")
        return None


# ============================================================================
# STRUCTURED LOGGING WITH TRACE CONTEXT
# ============================================================================

def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context for structured logging.

    Returns:
        dict with trace_id, span_id, correlation_id
    """
    context = {
        "correlation_id": correlation_id.get() or "unknown",
    }

    if OTEL_AVAILABLE:
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                ctx = current_span.get_span_context()
                context["trace_id"] = format(ctx.trace_id, '032x')
                context["span_id"] = format(ctx.span_id, '016x')
        except Exception:
            pass

    return context


# Initialize on import if configured
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    init_tracing()
