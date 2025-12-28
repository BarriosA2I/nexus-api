"""
Circuit Breaker Pattern for NEXUS Workers
Prevents cascading failures by failing fast when downstream services are unhealthy
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum
from functools import wraps

logger = logging.getLogger("nexus.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, not calling downstream
    HALF_OPEN = "half_open"  # Testing if downstream recovered


@dataclass
class CircuitBreaker:
    """
    Per-worker circuit breaker with exponential backoff

    States:
        CLOSED: Normal operation, tracking failures
        OPEN: Downstream unhealthy, fail immediately
        HALF_OPEN: Allow one test request through

    Usage:
        breaker = CircuitBreaker(name="anthropic", threshold=5, reset_timeout=30)

        if await breaker.can_execute():
            try:
                result = await call_api()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise CircuitOpenError("anthropic")
    """
    name: str
    threshold: int = 5          # Failures before opening
    reset_timeout: float = 30.0  # Seconds before trying half-open
    half_open_max: int = 3      # Max concurrent half-open requests

    # Internal state
    state: CircuitState = field(default=CircuitState.CLOSED)
    failures: int = field(default=0)
    successes: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_count: int = field(default=0)

    # Metrics
    total_calls: int = field(default=0)
    total_failures: int = field(default=0)
    total_rejections: int = field(default=0)

    async def can_execute(self) -> bool:
        """Check if request should proceed"""
        self.total_calls += 1

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if enough time passed to try half-open
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
                logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN (testing recovery)")
                return True

            self.total_rejections += 1
            logger.warning(f"Circuit {self.name}: OPEN - rejecting request")
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests through
            if self.half_open_count < self.half_open_max:
                self.half_open_count += 1
                return True

            self.total_rejections += 1
            return False

        return True

    def record_success(self) -> None:
        """Record successful call"""
        self.successes += 1

        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open → close circuit
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.half_open_count = 0
            logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (recovered)")

        # Reset failure count on success in closed state
        if self.state == CircuitState.CLOSED:
            self.failures = 0

    def record_failure(self) -> None:
        """Record failed call"""
        self.failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open → back to open
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN (still failing)")

        elif self.state == CircuitState.CLOSED and self.failures >= self.threshold:
            # Too many failures → open circuit
            self.state = CircuitState.OPEN
            logger.error(f"Circuit {self.name}: CLOSED → OPEN (threshold {self.threshold} reached)")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "threshold": self.threshold,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "success_rate": (self.total_calls - self.total_failures) / max(1, self.total_calls),
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected"""
    def __init__(self, circuit_name: str):
        self.circuit_name = circuit_name
        super().__init__(f"Circuit breaker '{circuit_name}' is OPEN - failing fast")


# ============================================================================
# GLOBAL CIRCUIT BREAKER REGISTRY
# ============================================================================
class CircuitBreakerRegistry:
    """Centralized registry for all circuit breakers"""

    _instance: Optional["CircuitBreakerRegistry"] = None

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, threshold: int = 5, reset_timeout: float = 30.0) -> CircuitBreaker:
        """Register or get existing circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                name=name,
                threshold=threshold,
                reset_timeout=reset_timeout,
            )
            logger.info(f"Registered circuit breaker: {name}")
        return self.breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self.breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict]:
        return {name: cb.get_stats() for name, cb in self.breakers.items()}


# ============================================================================
# DECORATOR FOR AUTOMATIC CIRCUIT BREAKING
# ============================================================================
def with_circuit_breaker(
    circuit_name: str,
    threshold: int = 5,
    reset_timeout: float = 30.0,
    fallback: Optional[Callable] = None,
):
    """
    Decorator to wrap async functions with circuit breaker logic

    Usage:
        @with_circuit_breaker("anthropic", threshold=5, reset_timeout=30)
        async def call_claude(prompt: str) -> str:
            return await client.messages.create(...)
    """
    registry = CircuitBreakerRegistry.get_instance()
    breaker = registry.register(circuit_name, threshold, reset_timeout)

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not await breaker.can_execute():
                if fallback:
                    logger.warning(f"Circuit {circuit_name} open - using fallback")
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(circuit_name)

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()

                # Check if we should use fallback
                if fallback and breaker.state == CircuitState.OPEN:
                    logger.warning(f"Circuit {circuit_name} opened - using fallback")
                    return await fallback(*args, **kwargs)

                raise

        return wrapper
    return decorator


# ============================================================================
# PRE-CONFIGURED BREAKERS FOR NEXUS WORKERS
# ============================================================================
def get_nexus_breakers() -> Dict[str, CircuitBreaker]:
    """Get pre-configured circuit breakers for all NEXUS workers"""
    registry = CircuitBreakerRegistry.get_instance()

    return {
        "anthropic": registry.register("anthropic", threshold=5, reset_timeout=30),
        "anthropic_haiku": registry.register("anthropic_haiku", threshold=10, reset_timeout=15),
        "qdrant": registry.register("qdrant", threshold=3, reset_timeout=60),
        "redis": registry.register("redis", threshold=3, reset_timeout=30),
        "postgres": registry.register("postgres", threshold=3, reset_timeout=60),
        "ragnarok": registry.register("ragnarok", threshold=2, reset_timeout=120),
        "trinity": registry.register("trinity", threshold=3, reset_timeout=60),
        "openai_tts": registry.register("openai_tts", threshold=3, reset_timeout=30),
        "kie_veo": registry.register("kie_veo", threshold=2, reset_timeout=120),
    }


# Initialize on import
_breakers = get_nexus_breakers()
