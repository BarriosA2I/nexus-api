"""
Nexus Assistant Unified - Circuit Breaker
Netflix Hystrix-style circuit breaker for resilience
"""
import asyncio
import time
import logging
from typing import Callable, Any, Optional, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

from ..config import settings

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Rejecting requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, state: CircuitState):
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(f"Circuit '{circuit_name}' is {state.value}")


# Alias for compatibility with external services
CircuitBreakerOpen = CircuitBreakerError


class CircuitBreaker:
    """
    Circuit breaker implementation following Netflix Hystrix pattern.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: After failure_threshold failures, reject all requests for recovery_timeout
    - HALF_OPEN: After recovery_timeout, allow limited requests to test recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: int = None,
        half_open_max_calls: int = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold or settings.CB_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or settings.CB_RECOVERY_TIMEOUT
        self.half_open_max_calls = half_open_max_calls or settings.CB_HALF_OPEN_MAX_CALLS

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_state_change >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics"""
        return self._stats

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def failure_count(self) -> int:
        """Get current consecutive failure count"""
        return self._stats.consecutive_failures

    @property
    def last_failure_time(self) -> Optional[datetime]:
        """Get last failure time as datetime"""
        if self._stats.last_failure_time:
            return datetime.fromtimestamp(self._stats.last_failure_time)
        return None

    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        return self.state != CircuitState.OPEN

    def record_success(self):
        """Public method to record a successful call"""
        self._record_success()

    def record_failure(self):
        """Public method to record a failed call"""
        self._record_failure()

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value}")

    def _record_success(self):
        """Record successful call"""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self):
        """Record failed call"""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _record_rejection(self):
        """Record rejected call"""
        self._stats.total_calls += 1
        self._stats.rejected_calls += 1

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Raises CircuitBreakerError if circuit is open.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                self._record_rejection()
                raise CircuitBreakerError(self.name, current_state)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._record_success()

            return result

        except Exception as e:
            async with self._lock:
                self._record_failure()
            raise

    def reset(self):
        """Manually reset circuit to closed state"""
        self._transition_to(CircuitState.CLOSED)
        self._stats = CircuitStats()
        logger.info(f"Circuit '{self.name}' manually reset")

    def to_dict(self) -> Dict[str, Any]:
        """Export circuit state as dictionary"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._stats.consecutive_failures,
            "last_failure": datetime.fromtimestamp(self._stats.last_failure_time).isoformat()
                if self._stats.last_failure_time else None,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
            }
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _circuits: Dict[str, CircuitBreaker] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._circuits = {}
        return cls._instance

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: int = None,
    ) -> CircuitBreaker:
        """Get existing circuit or create new one"""
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            logger.info(f"Created circuit breaker: {name}")
        return self._circuits[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit by name"""
        return self._circuits.get(name)

    def all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuits"""
        return self._circuits.copy()

    def reset_all(self):
        """Reset all circuits"""
        for circuit in self._circuits.values():
            circuit.reset()

    def health_summary(self) -> list:
        """Get health summary for all circuits"""
        return [cb.to_dict() for cb in self._circuits.values()]


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str):
    """
    Decorator to wrap function with circuit breaker.

    Usage:
        @circuit_breaker("my_service")
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        cb = circuit_registry.get_or_create(name)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)

        return wrapper
    return decorator


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Convenience function that wraps circuit_registry.get_or_create().
    """
    return circuit_registry.get_or_create(name)
