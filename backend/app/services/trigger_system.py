"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     NEXUS TRIGGER SYSTEM v1.0                                ║
║              "Zero Human Intervention Research Automation"                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  APScheduler Cron Jobs | RabbitMQ Event Consumers | Gap Detection            ║
║  Barrios A2I Cognitive Systems Division | December 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Trigger Types:
1. SCHEDULED - Cron-based weekly/biweekly/monthly research
2. GAP_DETECTED - Real-time gap detection from conversations
3. NEWS_UPDATE - News monitoring for regulatory/market changes
4. LOW_CONFIDENCE - Triggered when Nexus responds with low confidence
5. REPEATED_QUESTION - Same topic asked 3+ times
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import aio_pika
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from opentelemetry import trace
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from nexus_research_oracle import (
    INDUSTRY_SCHEDULE,
    NexusResearchOracle,
    ResearchPriority,
    ResearchTask,
    ResearchTrigger,
    slog,
)

logger = logging.getLogger("nexus_triggers")
tracer = trace.get_tracer("nexus_triggers", "1.0.0")


# =============================================================================
# DATA MODELS
# =============================================================================


class ConversationEvent(BaseModel):
    """Event from Nexus Brain conversation."""

    conversation_id: str
    user_message: str
    nexus_response: str
    detected_industry: Optional[str] = None
    confidence_score: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GapDetectionResult(BaseModel):
    """Result of gap detection analysis."""

    gap_detected: bool
    industry: Optional[str] = None
    topic: Optional[str] = None
    reason: str
    priority: ResearchPriority = ResearchPriority.MEDIUM


class NewsAlert(BaseModel):
    """News/regulatory change alert."""

    source: str
    title: str
    summary: str
    affected_industries: List[str]
    urgency: ResearchPriority
    detected_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# CONVERSATION TRACKER (for Repeated Questions)
# =============================================================================


class ConversationTracker:
    """
    Tracks conversation patterns to detect repeated questions.

    Uses Redis for persistence across restarts.
    """

    def __init__(
        self,
        redis_client: Redis,
        threshold: int = 3,
        window_hours: int = 24,
    ):
        self.redis = redis_client
        self.threshold = threshold
        self.window = timedelta(hours=window_hours)

    async def track_question(
        self,
        topic: str,
        industry: Optional[str] = None,
    ) -> bool:
        """
        Track a question topic.

        Returns True if threshold exceeded (research needed).
        """
        key = f"nexus:questions:{topic.lower()}"
        now = datetime.utcnow()
        cutoff = (now - self.window).timestamp()

        # Add current timestamp
        await self.redis.zadd(key, {str(now.timestamp()): now.timestamp()})

        # Remove old entries
        await self.redis.zremrangebyscore(key, "-inf", cutoff)

        # Check count
        count = await self.redis.zcard(key)

        # Set TTL to clean up old keys
        await self.redis.expire(key, int(self.window.total_seconds() * 2))

        if count >= self.threshold:
            slog.log(
                "INFO",
                "repeated_question_detected",
                topic=topic,
                count=count,
                industry=industry,
            )
            return True

        return False

    async def get_hot_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently asked topics."""
        # This would scan Redis keys - simplified for demo
        # In production, maintain a sorted set of topic counts
        return []


# =============================================================================
# GAP DETECTOR
# =============================================================================


class GapDetector:
    """
    Detects knowledge gaps from conversations.

    Triggers:
    - Unknown industry mentioned
    - Low confidence response
    - Stale knowledge (>30 days old)
    """

    # Industries we have full coverage for
    KNOWN_INDUSTRIES = {
        "law_firms",
        "dental_practices",
        "marketing_agencies",
        "real_estate",
        "ecommerce",
        "saas",
        "accounting",
        "construction",
        "restaurants",
        "insurance",
        "manufacturing",
        "healthcare",
    }

    # Pattern matching for industry detection
    INDUSTRY_PATTERNS = {
        "plumbing": ["plumber", "plumbing", "pipe", "drain", "leak"],
        "hvac": ["hvac", "heating", "cooling", "air conditioning", "furnace"],
        "landscaping": ["landscaping", "lawn", "gardening", "yard", "tree service"],
        "photography": ["photographer", "photography", "photos", "wedding photos"],
        "fitness": ["gym", "fitness", "personal trainer", "workout", "coaching"],
        "veterinary": ["vet", "veterinary", "animal", "pet clinic"],
        "pharmacy": ["pharmacy", "pharmacist", "prescription", "drugstore"],
        "automotive": ["auto shop", "mechanic", "car repair", "dealership", "body shop"],
        "flooring": ["flooring", "carpet", "tile", "hardwood", "flooring company"],
        "roofing": ["roofing", "roof", "roofer", "shingles"],
        "cleaning": ["cleaning service", "maid", "janitorial", "commercial cleaning"],
        "event_planning": ["event planner", "wedding planner", "catering", "party planning"],
        "tutoring": ["tutoring", "tutor", "learning center", "test prep"],
        "daycare": ["daycare", "childcare", "preschool", "child care center"],
        "funeral": ["funeral home", "mortuary", "cremation"],
    }

    def __init__(self, oracle: NexusResearchOracle, redis_client: Redis):
        self.oracle = oracle
        self.redis = redis_client

    async def analyze(self, event: ConversationEvent) -> GapDetectionResult:
        """
        Analyze conversation for knowledge gaps.

        Returns gap detection result with recommended action.
        """
        message_lower = event.user_message.lower()

        # Check 1: Unknown industry mentioned
        for industry, patterns in self.INDUSTRY_PATTERNS.items():
            if any(p in message_lower for p in patterns):
                if industry not in self.KNOWN_INDUSTRIES:
                    # Check freshness
                    freshness = await self._get_industry_freshness(industry)

                    if freshness is None:
                        return GapDetectionResult(
                            gap_detected=True,
                            industry=industry,
                            reason=f"Unknown industry: {industry}",
                            priority=ResearchPriority.HIGH,
                        )
                    elif (datetime.utcnow() - freshness).days > 30:
                        return GapDetectionResult(
                            gap_detected=True,
                            industry=industry,
                            reason=f"Stale knowledge for {industry} ({(datetime.utcnow() - freshness).days} days old)",
                            priority=ResearchPriority.MEDIUM,
                        )

        # Check 2: Low confidence response
        if event.confidence_score < 0.7:
            return GapDetectionResult(
                gap_detected=True,
                industry=event.detected_industry,
                topic=self._extract_topic(event.user_message),
                reason=f"Low confidence response ({event.confidence_score:.2f})",
                priority=ResearchPriority.MEDIUM,
            )

        # Check 3: Detected industry needs refresh
        if event.detected_industry:
            freshness = await self._get_industry_freshness(event.detected_industry)
            if freshness and (datetime.utcnow() - freshness).days > 30:
                return GapDetectionResult(
                    gap_detected=True,
                    industry=event.detected_industry,
                    reason=f"Knowledge refresh needed for {event.detected_industry}",
                    priority=ResearchPriority.LOW,
                )

        return GapDetectionResult(
            gap_detected=False,
            reason="No gap detected",
        )

    async def _get_industry_freshness(self, industry: str) -> Optional[datetime]:
        """Get last update time for industry from Redis cache."""
        key = f"nexus:freshness:{industry}"
        timestamp = await self.redis.get(key)

        if timestamp:
            return datetime.fromisoformat(timestamp.decode())
        return None

    async def _set_industry_freshness(self, industry: str, timestamp: datetime):
        """Update freshness cache."""
        key = f"nexus:freshness:{industry}"
        await self.redis.set(key, timestamp.isoformat())

    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract main topic from message (simplified)."""
        # In production, would use NER or keyword extraction
        words = message.lower().split()
        # Return first noun-like word (simplified)
        stopwords = {"i", "we", "you", "the", "a", "an", "is", "are", "have", "do", "what", "how"}
        for word in words:
            if word not in stopwords and len(word) > 3:
                return word
        return None


# =============================================================================
# NEWS MONITOR
# =============================================================================


class NewsMonitor:
    """
    Monitors news for regulatory/market changes.

    Uses periodic web searches to detect changes that
    should trigger knowledge refresh.
    """

    # Keywords that trigger industry research
    TRIGGER_KEYWORDS = {
        "healthcare": ["hipaa", "cms", "healthcare regulation", "medical billing"],
        "legal": ["legal regulation", "bar association", "court rules"],
        "financial": ["sec", "finra", "financial regulation", "banking law"],
        "real_estate": ["real estate law", "housing regulation", "fair housing"],
        "construction": ["building code", "osha", "construction regulation"],
    }

    def __init__(
        self,
        oracle: NexusResearchOracle,
        check_interval_hours: int = 6,
    ):
        self.oracle = oracle
        self.interval = check_interval_hours
        self.last_check: Dict[str, datetime] = {}

    async def check_for_updates(self) -> List[NewsAlert]:
        """Check all industries for news updates."""
        alerts = []

        for category, keywords in self.TRIGGER_KEYWORDS.items():
            # Rate limit: don't check same category more than once per interval
            if category in self.last_check:
                elapsed = datetime.utcnow() - self.last_check[category]
                if elapsed.total_seconds() < self.interval * 3600:
                    continue

            # Search for news
            query = f"{' '.join(keywords)} news changes 2025"

            try:
                result = await self.oracle.perplexity.search(query, recency_filter="week")

                # Analyze for significance (simplified)
                if self._is_significant(result["content"]):
                    alerts.append(
                        NewsAlert(
                            source="perplexity",
                            title=f"{category.title()} regulatory/market update detected",
                            summary=result["content"][:500],
                            affected_industries=self._map_category_to_industries(category),
                            urgency=ResearchPriority.MEDIUM,
                        )
                    )

                self.last_check[category] = datetime.utcnow()

            except Exception as e:
                slog.log(
                    "WARN",
                    "news_check_failed",
                    category=category,
                    error=str(e),
                )

        return alerts

    def _is_significant(self, content: str) -> bool:
        """Check if news content is significant enough to trigger research."""
        # Simplified - look for action words
        triggers = ["new regulation", "law change", "effective immediately", "major update"]
        content_lower = content.lower()
        return any(t in content_lower for t in triggers)

    def _map_category_to_industries(self, category: str) -> List[str]:
        """Map news category to affected industries."""
        mapping = {
            "healthcare": ["healthcare", "dental_practices"],
            "legal": ["law_firms"],
            "financial": ["accounting", "insurance"],
            "real_estate": ["real_estate"],
            "construction": ["construction"],
        }
        return mapping.get(category, [category])


# =============================================================================
# RABBITMQ EVENT CONSUMER
# =============================================================================


class EventConsumer:
    """
    RabbitMQ consumer for conversation events.

    Queues:
    - nexus.conversation.completed - All finished conversations
    - nexus.confidence.low - Low confidence responses
    - nexus.industry.unknown - Unknown industry detected
    """

    def __init__(
        self,
        rabbitmq_url: str,
        oracle: NexusResearchOracle,
        gap_detector: GapDetector,
        conversation_tracker: ConversationTracker,
    ):
        self.rabbitmq_url = rabbitmq_url
        self.oracle = oracle
        self.gap_detector = gap_detector
        self.tracker = conversation_tracker
        self._connection: Optional[aio_pika.Connection] = None
        self._channel: Optional[aio_pika.Channel] = None

    async def connect(self):
        """Establish RabbitMQ connection."""
        self._connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self._channel = await self._connection.channel()

        # Declare queues
        await self._channel.declare_queue("nexus.conversation.completed", durable=True)
        await self._channel.declare_queue("nexus.confidence.low", durable=True)
        await self._channel.declare_queue("nexus.industry.unknown", durable=True)

        slog.log("INFO", "rabbitmq_connected")

    async def start_consuming(self):
        """Start consuming from all queues."""
        if not self._channel:
            await self.connect()

        # Set up consumers
        conversation_queue = await self._channel.get_queue("nexus.conversation.completed")
        await conversation_queue.consume(self._handle_conversation)

        low_confidence_queue = await self._channel.get_queue("nexus.confidence.low")
        await low_confidence_queue.consume(self._handle_low_confidence)

        unknown_industry_queue = await self._channel.get_queue("nexus.industry.unknown")
        await unknown_industry_queue.consume(self._handle_unknown_industry)

        slog.log("INFO", "consumers_started", queues=3)

    @tracer.start_as_current_span("handle_conversation")
    async def _handle_conversation(self, message: aio_pika.IncomingMessage):
        """Handle completed conversation event."""
        async with message.process():
            try:
                data = json.loads(message.body.decode())
                event = ConversationEvent(**data)

                # Track for repeated questions
                topic = self.gap_detector._extract_topic(event.user_message)
                if topic:
                    needs_research = await self.tracker.track_question(
                        topic=topic,
                        industry=event.detected_industry,
                    )

                    if needs_research:
                        await self.oracle.queue_research(
                            industry=event.detected_industry or "general",
                            trigger=ResearchTrigger.REPEATED_QUESTION,
                            priority=ResearchPriority.MEDIUM,
                            topic=topic,
                        )

                # Run gap detection
                gap_result = await self.gap_detector.analyze(event)

                if gap_result.gap_detected and gap_result.industry:
                    await self.oracle.queue_research(
                        industry=gap_result.industry,
                        trigger=ResearchTrigger.GAP_DETECTED,
                        priority=gap_result.priority,
                        topic=gap_result.topic,
                    )

            except Exception as e:
                slog.log("ERROR", "conversation_handler_failed", error=str(e))

    @tracer.start_as_current_span("handle_low_confidence")
    async def _handle_low_confidence(self, message: aio_pika.IncomingMessage):
        """Handle low confidence response event."""
        async with message.process():
            try:
                data = json.loads(message.body.decode())
                event = ConversationEvent(**data)

                if event.detected_industry:
                    await self.oracle.queue_research(
                        industry=event.detected_industry,
                        trigger=ResearchTrigger.LOW_CONFIDENCE,
                        priority=ResearchPriority.HIGH,
                        topic=self.gap_detector._extract_topic(event.user_message),
                    )

            except Exception as e:
                slog.log("ERROR", "low_confidence_handler_failed", error=str(e))

    @tracer.start_as_current_span("handle_unknown_industry")
    async def _handle_unknown_industry(self, message: aio_pika.IncomingMessage):
        """Handle unknown industry detection event."""
        async with message.process():
            try:
                data = json.loads(message.body.decode())
                industry = data.get("industry")

                if industry:
                    await self.oracle.queue_research(
                        industry=industry,
                        trigger=ResearchTrigger.GAP_DETECTED,
                        priority=ResearchPriority.HIGH,
                    )

            except Exception as e:
                slog.log("ERROR", "unknown_industry_handler_failed", error=str(e))

    async def close(self):
        """Close RabbitMQ connection."""
        if self._connection:
            await self._connection.close()


# =============================================================================
# SCHEDULED JOB MANAGER (APScheduler)
# =============================================================================


class ScheduledJobManager:
    """
    Manages scheduled research jobs using APScheduler.

    Cron schedules from INDUSTRY_SCHEDULE:
    - tier_1_weekly: Every Monday 3 AM
    - tier_2_biweekly: 1st and 15th of month 3 AM
    - tier_3_monthly: 1st of month 3 AM
    """

    def __init__(self, oracle: NexusResearchOracle):
        self.oracle = oracle
        self.scheduler = AsyncIOScheduler()
        self._setup_jobs()

    def _setup_jobs(self):
        """Configure all scheduled jobs."""
        # Tier 1: Weekly (Monday 3 AM)
        self.scheduler.add_job(
            self._run_tier1,
            CronTrigger.from_crontab("0 3 * * MON"),
            id="tier_1_weekly",
            name="Weekly Research (Tier 1)",
        )

        # Tier 2: Biweekly (1st and 15th, 3 AM)
        self.scheduler.add_job(
            self._run_tier2,
            CronTrigger.from_crontab("0 3 1,15 * *"),
            id="tier_2_biweekly",
            name="Biweekly Research (Tier 2)",
        )

        # Tier 3: Monthly (1st of month, 3 AM)
        self.scheduler.add_job(
            self._run_tier3,
            CronTrigger.from_crontab("0 3 1 * *"),
            id="tier_3_monthly",
            name="Monthly Research (Tier 3)",
        )

        # Task processor: Every 5 minutes
        self.scheduler.add_job(
            self._process_queued_tasks,
            "interval",
            minutes=5,
            id="task_processor",
            name="Process Queued Tasks",
        )

        slog.log("INFO", "scheduled_jobs_configured", job_count=4)

    @tracer.start_as_current_span("tier1_research")
    async def _run_tier1(self):
        """Execute Tier 1 (weekly) research."""
        slog.log("INFO", "scheduled_job_started", tier="tier_1_weekly")

        tier_config = INDUSTRY_SCHEDULE["tier_1_weekly"]
        for industry in tier_config["industries"]:
            await self.oracle.queue_research(
                industry=industry,
                trigger=ResearchTrigger.SCHEDULED,
                priority=tier_config["priority"],
            )

    @tracer.start_as_current_span("tier2_research")
    async def _run_tier2(self):
        """Execute Tier 2 (biweekly) research."""
        slog.log("INFO", "scheduled_job_started", tier="tier_2_biweekly")

        tier_config = INDUSTRY_SCHEDULE["tier_2_biweekly"]
        for industry in tier_config["industries"]:
            await self.oracle.queue_research(
                industry=industry,
                trigger=ResearchTrigger.SCHEDULED,
                priority=tier_config["priority"],
            )

    @tracer.start_as_current_span("tier3_research")
    async def _run_tier3(self):
        """Execute Tier 3 (monthly) research."""
        slog.log("INFO", "scheduled_job_started", tier="tier_3_monthly")

        tier_config = INDUSTRY_SCHEDULE["tier_3_monthly"]
        for industry in tier_config["industries"]:
            await self.oracle.queue_research(
                industry=industry,
                trigger=ResearchTrigger.SCHEDULED,
                priority=tier_config["priority"],
            )

    async def _process_queued_tasks(self):
        """Process queued tasks (runs every 5 minutes)."""
        processed = 0

        while self.oracle.task_queue and processed < 5:  # Max 5 per cycle
            try:
                await self.oracle.process_next_task()
                processed += 1
            except Exception as e:
                slog.log("ERROR", "task_processing_failed", error=str(e))
                break

        if processed > 0:
            slog.log("INFO", "tasks_processed", count=processed)

    def start(self):
        """Start the scheduler."""
        self.scheduler.start()
        slog.log("INFO", "scheduler_started")

    def shutdown(self):
        """Shutdown the scheduler."""
        self.scheduler.shutdown()
        slog.log("INFO", "scheduler_stopped")

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time),
            }
            for job in self.scheduler.get_jobs()
        ]


# =============================================================================
# MAIN TRIGGER ORCHESTRATOR
# =============================================================================


class TriggerOrchestrator:
    """
    Master orchestrator for all trigger systems.

    Coordinates:
    - Scheduled jobs (APScheduler)
    - Event consumers (RabbitMQ)
    - Gap detection (real-time)
    - News monitoring (periodic)
    """

    def __init__(
        self,
        perplexity_api_key: str,
        anthropic_api_key: str,
        rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
        redis_url: str = "redis://localhost:6379",
        qdrant_url: str = "http://localhost:6333",
    ):
        # Initialize Oracle
        self.oracle = NexusResearchOracle(
            perplexity_api_key=perplexity_api_key,
            anthropic_api_key=anthropic_api_key,
            qdrant_url=qdrant_url,
        )

        # Redis client
        self.redis = Redis.from_url(redis_url)

        # Initialize components
        self.conversation_tracker = ConversationTracker(self.redis)
        self.gap_detector = GapDetector(self.oracle, self.redis)
        self.news_monitor = NewsMonitor(self.oracle)

        # Event consumer
        self.event_consumer = EventConsumer(
            rabbitmq_url=rabbitmq_url,
            oracle=self.oracle,
            gap_detector=self.gap_detector,
            conversation_tracker=self.conversation_tracker,
        )

        # Scheduler
        self.scheduler = ScheduledJobManager(self.oracle)

        self._running = False

    async def start(self):
        """Start all trigger systems."""
        self._running = True

        # Connect to RabbitMQ
        await self.event_consumer.connect()

        # Start scheduled jobs
        self.scheduler.start()

        # Start event consumers
        await self.event_consumer.start_consuming()

        # Start news monitor loop
        asyncio.create_task(self._news_monitor_loop())

        slog.log("INFO", "trigger_orchestrator_started")

    async def _news_monitor_loop(self):
        """Background loop for news monitoring."""
        while self._running:
            try:
                alerts = await self.news_monitor.check_for_updates()

                for alert in alerts:
                    for industry in alert.affected_industries:
                        await self.oracle.queue_research(
                            industry=industry,
                            trigger=ResearchTrigger.NEWS_UPDATE,
                            priority=alert.urgency,
                        )

            except Exception as e:
                slog.log("ERROR", "news_monitor_error", error=str(e))

            # Check every 6 hours
            await asyncio.sleep(6 * 3600)

    async def stop(self):
        """Stop all trigger systems."""
        self._running = False

        # Shutdown scheduler
        self.scheduler.shutdown()

        # Close RabbitMQ
        await self.event_consumer.close()

        # Close Oracle
        await self.oracle.close()

        # Close Redis
        await self.redis.close()

        slog.log("INFO", "trigger_orchestrator_stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "oracle": self.oracle.get_status(),
            "scheduled_jobs": self.scheduler.get_jobs(),
        }


# =============================================================================
# INTEGRATION WITH NEXUS BRAIN
# =============================================================================


class NexusBrainIntegration:
    """
    Integration layer between Nexus Brain and Research Oracle.

    Publishes events to RabbitMQ for trigger processing.
    Provides knowledge retrieval for response augmentation.
    """

    def __init__(
        self,
        oracle: NexusResearchOracle,
        rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
        redis_url: str = "redis://localhost:6379",
    ):
        self.oracle = oracle
        self.rabbitmq_url = rabbitmq_url
        self.redis = Redis.from_url(redis_url)
        self._connection: Optional[aio_pika.Connection] = None
        self._channel: Optional[aio_pika.Channel] = None

    async def connect(self):
        """Establish connections."""
        self._connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self._channel = await self._connection.channel()

    async def publish_conversation_event(
        self,
        conversation_id: str,
        user_message: str,
        nexus_response: str,
        detected_industry: Optional[str] = None,
        confidence_score: float = 1.0,
    ):
        """Publish conversation event for trigger processing."""
        if not self._channel:
            await self.connect()

        event = ConversationEvent(
            conversation_id=conversation_id,
            user_message=user_message,
            nexus_response=nexus_response,
            detected_industry=detected_industry,
            confidence_score=confidence_score,
        )

        # Route to appropriate queue
        if confidence_score < 0.7:
            queue_name = "nexus.confidence.low"
        else:
            queue_name = "nexus.conversation.completed"

        await self._channel.default_exchange.publish(
            aio_pika.Message(body=event.model_dump_json().encode()),
            routing_key=queue_name,
        )

    async def get_augmented_knowledge(
        self,
        message: str,
        industry: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get relevant knowledge to augment Nexus Brain response.

        Returns:
        {
            "chunks": List of relevant knowledge chunks,
            "scripts": Conversation scripts for industry,
            "terminology": Industry terminology,
            "freshness": Knowledge freshness info
        }
        """
        # Search vector store
        chunks = await self.oracle.search_knowledge(
            query=message,
            industry=industry,
            limit=5,
        )

        # Get cached scripts
        scripts = None
        if industry:
            scripts_key = f"nexus:scripts:{industry}"
            cached = await self.redis.get(scripts_key)
            if cached:
                scripts = json.loads(cached.decode())

        # Get terminology
        terminology = None
        if industry:
            term_key = f"nexus:terminology:{industry}"
            cached = await self.redis.get(term_key)
            if cached:
                terminology = json.loads(cached.decode())

        # Get freshness
        freshness = None
        if industry:
            fresh_key = f"nexus:freshness:{industry}"
            cached = await self.redis.get(fresh_key)
            if cached:
                freshness = cached.decode()

        return {
            "chunks": chunks,
            "scripts": scripts,
            "terminology": terminology,
            "freshness": freshness,
        }

    async def detect_industry(self, message: str) -> Optional[str]:
        """Detect industry from user message."""
        message_lower = message.lower()

        # Industry keywords mapping
        industry_keywords = {
            "dental_practices": ["dental", "dentist", "orthodontist", "teeth"],
            "law_firms": ["law firm", "attorney", "lawyer", "legal"],
            "real_estate": ["realtor", "real estate", "broker", "property"],
            "ecommerce": ["ecommerce", "shopify", "online store", "amazon seller"],
            "healthcare": ["doctor", "clinic", "medical", "healthcare", "physician"],
            "marketing_agencies": ["marketing agency", "ad agency", "digital agency"],
            "saas": ["saas", "software company", "tech startup", "platform"],
            "accounting": ["accountant", "cpa", "bookkeeper", "tax"],
            "construction": ["contractor", "construction", "builder"],
            "restaurants": ["restaurant", "cafe", "bar", "food service"],
        }

        for industry, keywords in industry_keywords.items():
            if any(kw in message_lower for kw in keywords):
                return industry

        return None

    async def close(self):
        """Close connections."""
        if self._connection:
            await self._connection.close()
        await self.redis.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os

    async def main():
        # Initialize orchestrator
        orchestrator = TriggerOrchestrator(
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

        # Start all systems
        await orchestrator.start()

        print("\n" + "=" * 60)
        print("🚀 NEXUS TRIGGER ORCHESTRATOR RUNNING")
        print("=" * 60)
        print("\nStatus:", json.dumps(orchestrator.get_status(), indent=2, default=str))

        # Run until interrupted
        try:
            while True:
                await asyncio.sleep(60)
                print(f"\n[{datetime.utcnow().isoformat()}] Status: {orchestrator.get_status()}")
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            await orchestrator.stop()

    asyncio.run(main())
