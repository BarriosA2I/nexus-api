"""
EVENT CONSTANTS
===============
Centralized definition of Exchange and Routing keys.
Prevents "magic string" drift between producer and consumer.

CRITICAL: Import this everywhere instead of hardcoding strings!
"""


class EventConstants:
    """Shared constants for RabbitMQ event system."""

    # Exchange
    EXCHANGE_NAME = "nexus_events"

    # Routing Keys
    ROUTING_KEY_CHAT_COMPLETED = "nexus.conversation.completed"
    ROUTING_KEY_LOW_CONFIDENCE = "nexus.confidence.low"
    ROUTING_KEY_INDUSTRY_UNKNOWN = "nexus.industry.unknown"

    # Queues
    QUEUE_NAME_RESEARCH = "nexus_research_queue"
    QUEUE_NAME_ANALYTICS = "nexus_analytics_queue"

    # Content Types
    CONTENT_TYPE_JSON = "application/json"

    # Dead Letter
    DLX_EXCHANGE = "nexus_dlx"
    DLQ_NAME = "nexus_dead_letter_queue"
