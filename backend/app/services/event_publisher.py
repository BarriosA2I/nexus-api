"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEXUS EVENT PUBLISHER v1.0                                ║
║              RabbitMQ Publisher for Feedback Loop                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Publishes conversation events to trigger research when gaps detected        ║
║  Barrios A2I Cognitive Systems Division | December 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("event_publisher")

# Try to import aio_pika (optional dependency)
try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    logger.warning("⚠️ aio-pika not installed - event publishing disabled")

# Import schemas with fallback
try:
    from .schemas import ConversationEvent, RoutingKeys
except ImportError:
    try:
        from schemas import ConversationEvent, RoutingKeys
    except ImportError:
        # Define minimal fallback if schemas not available
        class RoutingKeys:
            CONVERSATION_COMPLETED = "nexus.conversation.completed"
            CONFIDENCE_LOW = "nexus.confidence.low"
            INDUSTRY_UNKNOWN = "nexus.industry.unknown"
            EXCHANGE = "nexus.events"

        ConversationEvent = None
        logger.warning("⚠️ schemas not available - using minimal fallback")


class NexusEventPublisher:
    """
    Publishes conversation events to RabbitMQ for Trigger System.

    Gracefully degrades when RabbitMQ is not available:
    - No aio-pika installed → logs info, continues
    - No RABBITMQ_URL env → logs info, continues
    - Connection fails → logs warning, continues
    """

    def __init__(self):
        self._connection = None
        self._channel = None
        self._exchange = None
        self._enabled = False

    async def connect(self):
        """Connect to RabbitMQ. Gracefully degrades if unavailable."""
        if not RABBITMQ_AVAILABLE:
            logger.info("RabbitMQ publishing disabled (aio-pika not installed)")
            return

        rabbitmq_url = os.getenv("RABBITMQ_URL")
        if not rabbitmq_url:
            logger.info("RabbitMQ publishing disabled (RABBITMQ_URL not set)")
            return

        try:
            self._connection = await aio_pika.connect_robust(rabbitmq_url)
            self._channel = await self._connection.channel()

            # Declare topic exchange for routing
            self._exchange = await self._channel.declare_exchange(
                RoutingKeys.EXCHANGE,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )

            self._enabled = True
            logger.info(f"✅ RabbitMQ connected: exchange={RoutingKeys.EXCHANGE}")

        except Exception as e:
            logger.warning(f"⚠️ RabbitMQ connection failed: {e}")
            self._enabled = False

    async def publish_conversation_event(
        self,
        conversation_id: str,
        user_message: str,
        nexus_response: str,
        detected_industry: Optional[str] = None,
        confidence_score: float = 1.0,
        rag_chunks_used: int = 0,
        response_latency_ms: float = 0.0,
    ):
        """
        Publish a conversation event to RabbitMQ.

        Routing:
        - Low confidence (<0.7) → nexus.confidence.low
        - Unknown industry → nexus.industry.unknown
        - Normal → nexus.conversation.completed
        """
        if not self._enabled:
            logger.debug(f"Event publish skipped (not enabled): session={conversation_id[:12]}")
            return

        # Build event payload
        event_data = {
            "conversation_id": conversation_id,
            "user_message": user_message[:500],  # Truncate for size
            "nexus_response": nexus_response[:500],
            "detected_industry": detected_industry,
            "confidence_score": confidence_score,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rag_chunks_used": rag_chunks_used,
            "response_latency_ms": response_latency_ms,
        }

        # Determine routing key based on confidence/industry
        if confidence_score < 0.7:
            routing_key = RoutingKeys.CONFIDENCE_LOW
        elif detected_industry is None:
            routing_key = RoutingKeys.INDUSTRY_UNKNOWN
        else:
            routing_key = RoutingKeys.CONVERSATION_COMPLETED

        try:
            message = aio_pika.Message(
                body=json.dumps(event_data).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            await self._exchange.publish(message, routing_key=routing_key)

            logger.debug(
                f"📤 Event published: {routing_key} | "
                f"industry={detected_industry} | conf={confidence_score:.2f}"
            )

        except Exception as e:
            logger.warning(f"⚠️ Event publish failed: {e}")

    async def close(self):
        """Close RabbitMQ connection."""
        if self._connection:
            await self._connection.close()
            logger.info("RabbitMQ connection closed")

    @property
    def is_enabled(self) -> bool:
        """Check if event publishing is enabled."""
        return self._enabled


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_event_publisher: Optional[NexusEventPublisher] = None


async def get_event_publisher() -> NexusEventPublisher:
    """Get or create the event publisher singleton."""
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = NexusEventPublisher()
        await _event_publisher.connect()
    return _event_publisher


async def close_event_publisher():
    """Close the event publisher (call on shutdown)."""
    global _event_publisher
    if _event_publisher:
        await _event_publisher.close()
        _event_publisher = None
