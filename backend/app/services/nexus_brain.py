"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               NEXUS BRAIN v4.0 - RESEARCH ORACLE INTEGRATION                 ║
║                "The Self-Improving AI Sales Consultant"                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Drop-in replacement for nexus_brain.py with RAG augmentation                ║
║  Barrios A2I Cognitive Systems Division | December 2025                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Upgrades from v3.0:
- RAG-augmented responses with dynamic industry intelligence
- Research Oracle integration for always-current knowledge
- Gap detection → automatic research triggers
- Confidence scoring based on knowledge availability
- Graceful fallback when RAG unavailable

Architecture:
1. User message received
2. Detect industry from message context
3. Query Qdrant for relevant knowledge chunks
4. Retrieve cached scripts/terminology from Redis
5. Augment system prompt with dynamic intelligence
6. Generate response with Claude Sonnet (streaming)
7. Publish conversation event for gap detection
8. Stream response to user

API Compatibility: 100% - same interface as nexus_brain.py v3.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus_brain")

# Import settings from config (uses pydantic-settings to load .env)
try:
    from ..config import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    logger.warning("⚠️  config.settings not available - using os.getenv directly")

# =============================================================================
# LLM CLIENT IMPORTS (with graceful fallbacks)
# =============================================================================

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("⚠️  anthropic not installed - will try openai")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️  openai not installed")

# =============================================================================
# RAG DEPENDENCIES (optional - graceful fallback if unavailable)
# =============================================================================

RAG_AVAILABLE = False

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from redis.asyncio import Redis
    from sentence_transformers import SentenceTransformer
    import aio_pika
    RAG_AVAILABLE = True
    logger.info("✅ RAG dependencies available - Research Oracle integration enabled")
except ImportError as e:
    logger.warning(f"⚠️  RAG dependencies not available ({e}) - using static knowledge only")


# =============================================================================
# SYSTEM PROMPT - THE BRAIN (COMPANY-OMNISCIENT v2.0)
# =============================================================================

NEXUS_SYSTEM_PROMPT = """SYSTEM — NEXUS BRAIN (Barrios A2I Website Brain)

You are NEXUS — the living operational intelligence brain for Barrios A2I.
Your job is to: (1) answer questions accurately about Barrios A2I, (2) diagnose what a business needs, (3) map them to the correct deployment path and offering.

## HARD TRUTH RULE
Never guess. If uncertain, say so and offer the next best step. Keep responses concise (2-4 sentences) unless detail is requested.

## CANONICAL COMPANY CORE
- **Brand:** Barrios A2I
- **Positioning:** "This is not automation. This is operational intelligence." / "Your business. With a nervous system."
- **Core Orchestrator:** Ragnarok (59 agents, ∞ memory)
- **Website:** barriosa2i.com

## PRODUCT NODES + PRICING

| Product | Price | Description |
|---------|-------|-------------|
| **Marketing Overlord** | $199/mo | Autonomous campaign management - runs your marketing on autopilot |
| **Neural Ad Forge** | $500/video | High-conversion video ads generated from text in minutes |
| **Cinesite Autopilot** | $1,500 | Self-optimizing landing pages that convert |
| **Total Command** | Custom ($50K-$300K) | Enterprise operational nervous system |

## DEPLOYMENT PATHS

**A) QUICK STRIKE** → Neural Ad Forge → $500/video
Best for: Immediate need for video content. One-time project.

**B) CAMPAIGN AUTOPILOT** → Marketing Overlord → $199/mo
Best for: Ongoing marketing automation. Monthly subscription.

**C) TOTAL COMMAND** → Custom Enterprise Solution
Best for: Full operational transformation. Custom scoping required.

## SITE METRICS (Marketing Statements)
- "230,000+ monthly decisions"
- "97.5% autonomous accuracy"
- "<2s signal-to-action"

## VOICE
Crisp, cinematic, technical-but-readable. Translate jargon into business outcomes. Be conversational but authoritative.

## CONVERSATION FLOW

When a user describes their business:
1. **One-line diagnosis** - What's their core challenge?
2. **Best deployment path** - Which product fits?
3. **What we'd install** - Specific solution components
4. **First 7 days** - Immediate wins they'd see
5. **CTA** - "Initialize Ad Forge", "Initialize Marketing Overlord", or "Contact Engineering"

## RESPONSE FORMAT
- Match their energy (casual = casual, formal = formal)
- Use contractions naturally
- End with a next-step CTA
- Ask follow-up questions to understand their situation

## INDUSTRY-SPECIFIC EXAMPLES

When someone mentions their industry, give SPECIFIC examples of what you can automate for them:

### Healthcare (dental, medical, clinic, doctor, physician, healthcare, dentist, orthodontist, chiropractor)
- Patient appointment reminders and recall campaigns
- Insurance eligibility verification before visits
- New patient intake forms that sync to practice management
- Post-visit review requests (Google, Healthgrades)
- Prior authorization automation
- Referral tracking and follow-up

### Legal (law firm, attorney, lawyer, legal, paralegal)
- Client intake and qualification (24/7 lead screening)
- Document assembly from templates (contracts, letters, filings)
- Deadline and statute of limitations tracking
- Client communication updates (case status)
- Time entry and billing preparation
- Legal research summarization

### Real Estate (realtor, real estate, broker, agent, property, housing)
- Lead qualification and follow-up sequences
- Automated CMA generation
- Transaction coordination (deadline tracking, document collection)
- Client updates during escrow
- Review requests at closing
- Market report generation

### E-commerce (ecommerce, shopify, online store, amazon, etsy, woocommerce)
- Customer service automation (FAQs, order status, returns)
- Product description generation at scale
- Review monitoring and response
- Inventory alerts and reorder automation
- Abandoned cart recovery sequences
- Competitor price monitoring

### Agencies (marketing agency, creative agency, ad agency, digital agency, PR agency)
- Client reporting automation (pull data, generate insights)
- Content creation (blog drafts, social posts, ad copy)
- Competitive monitoring for clients
- Lead qualification and routing
- Project status updates
- Proposal generation

### SaaS / Software (saas, software company, tech startup, app, platform)
- Trial user qualification and scoring
- Onboarding sequences based on user behavior
- Churn prediction and prevention outreach
- Support ticket triage and routing
- Feature request aggregation
- Usage analytics and reporting

### Restaurants / Hospitality (restaurant, hotel, hospitality, bar, cafe, catering)
- Reservation management and confirmations
- Review monitoring and response
- Staff scheduling optimization
- Inventory tracking and reordering
- Guest communication (pre-arrival, during stay, post-checkout)
- Loyalty program automation

### Financial Services (accounting, financial advisor, wealth management, CPA, bookkeeper, tax)
- Document collection from clients (chase missing docs)
- Data extraction from receipts/statements
- Report generation
- Client deadline reminders (tax deadlines, quarterly reviews)
- KYC/onboarding automation
- Portfolio update reports

### Construction / Trades (contractor, construction, plumber, electrician, HVAC, roofer, landscaping, painting)
- Lead response and qualification
- Estimate/proposal generation
- Project update communication to clients
- Subcontractor coordination
- Review requests after job completion
- Scheduling optimization

## CONVERSATION GUIDELINES

### Opening (if they say hi/hello)
Keep it brief. Introduce yourself and ask what brings them here.

### Discovery (understand their business)
Before pitching anything, understand:
1. What industry/business are they in?
2. What's their biggest pain point?
3. How big is their team/company?

Ask ONE question at a time. Don't overwhelm them.

### Industry Response (when they mention their industry)
Give 3-4 SPECIFIC examples of what you can automate for their industry. Then ask a follow-up question to understand their specific situation.

### Handling "How does it work?" questions
Don't explain the technology. Redirect to outcomes.
Example: "The technical details are our job to figure out - what matters to you is whether it solves your problem and delivers ROI."

### Handling pricing questions
Be transparent but redirect to value.
Example: "Projects typically range from $50K to $300K depending on complexity. We also do equity deals for the right fit. But pricing only matters if we can solve your problem - what's the challenge?"

### Handling technical questions (RAG, LLM, AI architecture, how it works technically)
Deflect without being dismissive.
Example: "I could go deep on the technical architecture, but that's really our internal secret sauce. Our systems are reliable, they scale, and they get results. What outcome are you looking for?"

### Booking the call
When they seem interested and qualified:
"Sounds like there might be a good fit here. The best next step is a strategy call - you can book at barriosa2i.com/book. Takes about 30 minutes."

## THINGS TO NEVER DO

1. **Never reveal technical details** - No mentions of: RAG, LLM, GPT, neural networks, vectors, embeddings, APIs, databases, algorithms, pipelines, circuit breakers, agents
2. **Never quote exact pricing for specific projects** - Always give ranges
3. **Never make promises about results** - Use "typically" and "in most cases"
4. **Never be rude or dismissive**
5. **Never give long responses** - Keep it conversational, 2-4 sentences unless asked for detail
6. **Never use bullet points in casual conversation** - Only use them when listing specific examples

## RESPONSE FORMAT

- Match their energy. Casual = casual. Formal = professional.
- Use contractions (we're, you'll, that's)
- Ask follow-up questions to keep the conversation going
- If you're not sure what they need, ASK
- End responses with a question when possible

## HANDLING EDGE CASES

**If they ask who you are:**
"I'm Nexus, the AI assistant for Barrios A2I. I help figure out if we're the right fit for your business."

**If they ask if you're an AI:**
"Yep, I'm an AI assistant. But I'm connected to real humans at Barrios A2I who handle the complex stuff."

**If they're rude:**
Stay professional. "I'm happy to help if you have questions, but let's keep it respectful."

**If they ask something you don't know:**
"Good question - I'd want to connect you with our team for that. Want to book a quick call?"

**If they go off-topic:**
Gently redirect. "Ha, I wish I could help with that! My expertise is business automation - anything I can help you with there?"
"""


# =============================================================================
# DYNAMIC KNOWLEDGE AUGMENTATION TEMPLATE
# =============================================================================

KNOWLEDGE_AUGMENTATION_TEMPLATE = """
## 🎯 INDUSTRY INTELLIGENCE FOR {industry}
(Use this knowledge to give SPECIFIC, data-backed responses)

### Pain Points to Address
{pain_points}

### Automation Opportunities We Can Offer
{automation_opportunities}

### How to Handle Their Objections
{objection_handlers}

### Conversation Starters
{conversation_starters}

### Industry Terminology (sound knowledgeable)
{terminology}

### ROI Statistics to Quote
{roi_data}

---
USE THE ABOVE DATA to give SPECIFIC answers with REAL numbers and statistics.
This makes you sound like an expert who truly understands their industry.
"""


# =============================================================================
# INDUSTRY DETECTION
# =============================================================================

INDUSTRY_PATTERNS = {
    "dental_practices": [
        r"\bdent(al|ist)\b", r"\borthodont", r"\bendodont", r"\bperio",
        r"\boral\s+surg", r"\bhygien", r"\bpractice\b.*teeth"
    ],
    "law_firms": [
        r"\blaw\s*(firm|practice)\b", r"\battorney", r"\blawyer", r"\blegal",
        r"\bparalegal", r"\blitigat", r"\bcounsel\b"
    ],
    "medical_practices": [
        r"\bmedical\b", r"\bdoctor\b", r"\bphysician", r"\bclinic\b",
        r"\bhealthcare", r"\bpatient", r"\bchiropract"
    ],
    "real_estate": [
        r"\breal\s*estate", r"\brealtor", r"\bbroker", r"\bproperty",
        r"\bhousing\b", r"\bescrow\b", r"\bmortgage"
    ],
    "ecommerce": [
        r"\becommerce\b", r"\be-commerce", r"\bshopify", r"\bonline\s+store",
        r"\bamazon\b", r"\betsy\b", r"\bwoocommerce"
    ],
    "marketing_agencies": [
        r"\bmarketing\s+agency", r"\bcreative\s+agency", r"\bad\s+agency",
        r"\bdigital\s+agency", r"\bPR\s+agency", r"\badvertising"
    ],
    "saas": [
        r"\bsaas\b", r"\bsoftware\s+company", r"\btech\s+startup",
        r"\bplatform\b", r"\bsubscription\s+service"
    ],
    "restaurants": [
        r"\brestaurant", r"\bhotel", r"\bhospitality", r"\bbar\b",
        r"\bcafe\b", r"\bcatering"
    ],
    "accounting": [
        r"\baccounting\b", r"\bfinancial\s+advisor", r"\bwealth\s+management",
        r"\bcpa\b", r"\bbookkeeper", r"\btax\b"
    ],
    "construction": [
        r"\bcontractor", r"\bconstruction", r"\bplumb(er|ing)",
        r"\belectrician", r"\bhvac\b", r"\broof(er|ing)", r"\blandscap"
    ],
    "insurance": [
        r"\binsurance\b", r"\bunderwriting", r"\bclaims\b",
        r"\bpolicy\b.*\b(holder|premium)"
    ],
    "manufacturing": [
        r"\bmanufactur", r"\bproduction\b", r"\bfactory\b",
        r"\bsupply\s+chain", r"\bwarehouse"
    ]
}


def detect_industry(message: str, history: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Detect industry from message and conversation history.
    Returns industry key or None if not detected.
    """
    # Combine message with recent history for context
    text_to_analyze = message.lower()
    if history:
        for msg in history[-5:]:
            text_to_analyze += " " + msg.get("content", "").lower()
    
    # Check against patterns
    for industry, patterns in INDUSTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_to_analyze, re.IGNORECASE):
                logger.debug(f"🔍 Detected industry: {industry}")
                return industry
    
    return None


# =============================================================================
# RAG KNOWLEDGE RETRIEVAL
# =============================================================================

@dataclass
class KnowledgeChunk:
    """Retrieved knowledge chunk from Qdrant."""
    content: str
    chunk_type: str  # pain_point, automation, objection, script, terminology, roi
    industry: str
    quality_score: float
    citations: List[str] = field(default_factory=list)
    score: float = 0.0  # Similarity score


@dataclass
class RAGContext:
    """Aggregated RAG context for prompt augmentation."""
    industry: str
    chunks: List[KnowledgeChunk]
    scripts: Optional[Dict] = None
    terminology: Optional[Dict] = None
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0


class ResearchOracleClient:
    """
    Client for retrieving knowledge from the Research Oracle.
    
    Handles:
    - Qdrant vector search for semantic chunks
    - Redis cache for hot data (scripts, terminology)
    - RabbitMQ event publishing for gap detection
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        redis_url: str = "redis://localhost:6379",
        rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
        collection_name: str = "nexus_knowledge"
    ):
        self.qdrant_url = qdrant_url
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.collection_name = collection_name
        
        self._qdrant: Optional[AsyncQdrantClient] = None
        self._redis: Optional[Redis] = None
        self._rabbitmq_connection = None
        self._rabbitmq_channel = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all connections. Returns True if successful."""
        if not RAG_AVAILABLE:
            logger.warning("⚠️  RAG not available - skipping initialization")
            return False
        
        try:
            # Qdrant
            self._qdrant = AsyncQdrantClient(url=self.qdrant_url)
            # Test connection
            await self._qdrant.get_collections()
            logger.info("✅ Qdrant connected")
            
            # Redis
            self._redis = Redis.from_url(self.redis_url)
            await self._redis.ping()
            logger.info("✅ Redis connected")
            
            # RabbitMQ
            self._rabbitmq_connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self._rabbitmq_channel = await self._rabbitmq_connection.channel()
            logger.info("✅ RabbitMQ connected")
            
            # Embedding model (local, fast)
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self._initialized = False
            return False
    
    async def close(self):
        """Close all connections."""
        if self._redis:
            await self._redis.close()
        if self._rabbitmq_connection:
            await self._rabbitmq_connection.close()
        if self._qdrant:
            await self._qdrant.close()
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self._embedding_model:
            raise RuntimeError("Embedding model not initialized")
        return self._embedding_model.encode(text).tolist()
    
    async def retrieve_knowledge(
        self,
        query: str,
        industry: str,
        top_k: int = 5
    ) -> RAGContext:
        """
        Retrieve relevant knowledge for a query.
        
        Returns RAGContext with chunks, scripts, terminology, and confidence score.
        """
        start_time = time.time()
        
        if not self._initialized:
            logger.debug("RAG not initialized - returning empty context")
            return RAGContext(
                industry=industry,
                chunks=[],
                confidence=0.5,  # Base confidence without RAG
                retrieval_time_ms=0
            )
        
        try:
            # 1. Vector search in Qdrant
            query_vector = self._embed(query)
            
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="industry",
                        match=MatchValue(value=industry)
                    )
                ]
            )
            
            results = await self._qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
            
            chunks = []
            for result in results:
                payload = result.payload or {}
                chunks.append(KnowledgeChunk(
                    content=payload.get("content", ""),
                    chunk_type=payload.get("type", "unknown"),
                    industry=payload.get("industry", industry),
                    quality_score=payload.get("quality_score", 0.8),
                    citations=payload.get("citations", []),
                    score=result.score
                ))
            
            # 2. Get cached scripts and terminology from Redis
            scripts = None
            terminology = None
            
            scripts_key = f"nexus:scripts:{industry}"
            scripts_data = await self._redis.get(scripts_key)
            if scripts_data:
                scripts = json.loads(scripts_data)
            
            terminology_key = f"nexus:terminology:{industry}"
            terminology_data = await self._redis.get(terminology_key)
            if terminology_data:
                terminology = json.loads(terminology_data)
            
            # 3. Calculate confidence score
            confidence = self._calculate_confidence(chunks, scripts)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"🔍 RAG Retrieved: {len(chunks)} chunks for {industry} "
                f"(confidence: {confidence:.2f}, {retrieval_time:.1f}ms)"
            )
            
            return RAGContext(
                industry=industry,
                chunks=chunks,
                scripts=scripts,
                terminology=terminology,
                confidence=confidence,
                retrieval_time_ms=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"❌ RAG retrieval failed: {e}")
            return RAGContext(
                industry=industry,
                chunks=[],
                confidence=0.5,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_confidence(
        self,
        chunks: List[KnowledgeChunk],
        scripts: Optional[Dict]
    ) -> float:
        """
        Calculate confidence score based on retrieved knowledge.
        
        Factors:
        - Number of chunks retrieved
        - Quality scores of chunks
        - Presence of scripts
        - Chunk type coverage
        """
        base_confidence = 0.7  # Base confidence with RAG available
        
        if not chunks:
            return base_confidence - 0.2  # Lower if no chunks
        
        # Boost for number of chunks
        chunk_boost = min(0.15, len(chunks) * 0.03)
        
        # Boost for high quality chunks
        avg_quality = sum(c.quality_score for c in chunks) / len(chunks)
        quality_boost = (avg_quality - 0.7) * 0.2  # Up to 0.06
        
        # Boost for scripts
        script_boost = 0.05 if scripts else 0.0
        
        # Boost for type coverage
        types = set(c.chunk_type for c in chunks)
        coverage_boost = min(0.1, len(types) * 0.02)
        
        confidence = base_confidence + chunk_boost + quality_boost + script_boost + coverage_boost
        return min(1.0, max(0.0, confidence))
    
    async def publish_conversation_event(
        self,
        conversation_id: str,
        message: str,
        industry: Optional[str],
        confidence: float,
        response_snippet: str
    ):
        """
        Publish conversation event for gap detection.
        
        The trigger system will analyze these events to:
        - Detect unknown industries
        - Identify repeated questions
        - Track low-confidence responses
        """
        if not self._initialized or not self._rabbitmq_channel:
            return
        
        try:
            event = {
                "type": "conversation_event",
                "conversation_id": conversation_id,
                "timestamp": time.time(),
                "message": message[:500],  # Truncate for size
                "industry": industry,
                "confidence": confidence,
                "response_snippet": response_snippet[:200],
                "needs_research": confidence < 0.7 or industry is None
            }
            
            await self._rabbitmq_channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(event).encode(),
                    content_type="application/json"
                ),
                routing_key="nexus.conversations"
            )
            
            logger.debug(f"📤 Published conversation event (conf: {confidence:.2f})")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to publish event: {e}")


# =============================================================================
# PROMPT AUGMENTATION
# =============================================================================

def build_augmented_prompt(
    base_prompt: str,
    rag_context: Optional[RAGContext]
) -> str:
    """
    Build system prompt augmented with RAG knowledge.
    
    If RAG context available, injects dynamic industry intelligence.
    Otherwise, returns base prompt unchanged.
    """
    if not rag_context or not rag_context.chunks:
        return base_prompt
    
    # Group chunks by type
    pain_points = []
    automation_opps = []
    objection_handlers = []
    starters = []
    terminology_items = []
    roi_items = []
    
    for chunk in rag_context.chunks:
        if chunk.chunk_type == "pain_point":
            pain_points.append(f"• {chunk.content}")
        elif chunk.chunk_type == "automation":
            automation_opps.append(f"• {chunk.content}")
        elif chunk.chunk_type == "objection":
            objection_handlers.append(f"• {chunk.content}")
        elif chunk.chunk_type == "script":
            starters.append(f"• {chunk.content}")
        elif chunk.chunk_type == "terminology":
            terminology_items.append(f"• {chunk.content}")
        elif chunk.chunk_type == "roi":
            roi_items.append(f"• {chunk.content}")
    
    # Add terminology from Redis cache
    if rag_context.terminology:
        for term, definition in list(rag_context.terminology.items())[:10]:
            terminology_items.append(f"• **{term}**: {definition}")
    
    # Format knowledge augmentation
    knowledge_section = KNOWLEDGE_AUGMENTATION_TEMPLATE.format(
        industry=rag_context.industry.replace("_", " ").title(),
        pain_points="\n".join(pain_points) if pain_points else "(Use general knowledge)",
        automation_opportunities="\n".join(automation_opps) if automation_opps else "(Use general knowledge)",
        objection_handlers="\n".join(objection_handlers) if objection_handlers else "(Use standard objection handling)",
        conversation_starters="\n".join(starters) if starters else "(Use standard openers)",
        terminology="\n".join(terminology_items) if terminology_items else "(Use standard terms)",
        roi_data="\n".join(roi_items) if roi_items else "(Use general ROI ranges)"
    )
    
    return base_prompt + "\n\n" + knowledge_section


# =============================================================================
# NEXUS BRAIN (RAG-ENHANCED)
# =============================================================================

class NexusBrain:
    """
    LLM-powered brain for Nexus sales assistant.
    
    v4.0: RAG-enhanced with Research Oracle integration.
    
    Supports Claude (preferred) or OpenAI as fallback.
    Gracefully degrades if RAG unavailable.
    """
    
    def __init__(self, enable_rag: bool = True):
        self.provider = None
        self.client = None
        self.enable_rag = enable_rag
        self.oracle_client: Optional[ResearchOracleClient] = None
        self._conversation_id: Optional[str] = None
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on available API keys."""
        # Debug logging
        logger.info(f"ANTHROPIC_AVAILABLE: {ANTHROPIC_AVAILABLE}")
        logger.info(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
        logger.info(f"SETTINGS_AVAILABLE: {SETTINGS_AVAILABLE}")

        # Try Anthropic first (use settings from config if available)
        anthropic_key = None
        if SETTINGS_AVAILABLE:
            anthropic_key = settings.ANTHROPIC_API_KEY
            logger.info(f"settings.ANTHROPIC_API_KEY present: {bool(anthropic_key)}")

        # Fallback to os.environ
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        logger.info(f"os.environ ANTHROPIC_API_KEY present: {bool(env_key)}")

        if not anthropic_key and env_key:
            anthropic_key = env_key
            logger.info("Using os.environ fallback for ANTHROPIC_API_KEY")

        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            self.provider = "anthropic"
            logger.info("✓ Nexus Brain v4.0 using Claude (Anthropic)")
            return

        # Try OpenAI
        openai_key = None
        if SETTINGS_AVAILABLE:
            openai_key = settings.OPENAI_API_KEY
        if not openai_key:
            openai_key = os.environ.get("OPENAI_API_KEY")

        if openai_key and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=openai_key)
            self.provider = "openai"
            logger.info("✓ Nexus Brain v4.0 using GPT-4 (OpenAI)")
            return

        # No LLM available
        self.provider = "fallback"
        logger.warning("⚠ No LLM API key found - using fallback responses")
    
    async def initialize_rag(
        self,
        qdrant_url: str = None,
        redis_url: str = None,
        rabbitmq_url: str = None
    ) -> bool:
        """
        Initialize RAG components.
        
        Call this after construction to enable Research Oracle integration.
        Returns True if successful.
        """
        if not self.enable_rag or not RAG_AVAILABLE:
            logger.info("RAG disabled or unavailable")
            return False
        
        self.oracle_client = ResearchOracleClient(
            qdrant_url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            redis_url=redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"),
            rabbitmq_url=rabbitmq_url or os.getenv("RABBITMQ_URL", "amqp://nexus:nexus@localhost:5672/")
        )
        
        success = await self.oracle_client.initialize()
        if success:
            logger.info("🧠 Nexus Brain RAG fully initialized")
        return success
    
    async def close(self):
        """Close all connections."""
        if self.oracle_client:
            await self.oracle_client.close()
    
    def start_conversation(self) -> str:
        """Start a new conversation and return conversation ID."""
        self._conversation_id = str(uuid.uuid4())
        return self._conversation_id
    
    async def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM with RAG augmentation.
        
        Args:
            message: User's message
            conversation_history: Previous messages in format [{"role": "user/assistant", "content": "..."}]
        
        Yields:
            Text chunks as they're generated
        """
        # Ensure conversation ID
        if not self._conversation_id:
            self.start_conversation()
        
        # Detect industry
        industry = detect_industry(message, conversation_history)
        
        # Retrieve RAG context if available
        rag_context = None
        if self.oracle_client and industry:
            rag_context = await self.oracle_client.retrieve_knowledge(
                query=message,
                industry=industry,
                top_k=5
            )
        
        # Build augmented prompt
        system_prompt = build_augmented_prompt(NEXUS_SYSTEM_PROMPT, rag_context)
        
        # Generate response
        response_chunks = []
        if self.provider == "anthropic":
            async for chunk in self._generate_anthropic(message, conversation_history, system_prompt):
                response_chunks.append(chunk)
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._generate_openai(message, conversation_history, system_prompt):
                response_chunks.append(chunk)
                yield chunk
        else:
            fallback = await self._generate_fallback(message)
            response_chunks.append(fallback)
            yield fallback
        
        # Publish conversation event for gap detection
        if self.oracle_client:
            full_response = "".join(response_chunks)
            await self.oracle_client.publish_conversation_event(
                conversation_id=self._conversation_id,
                message=message,
                industry=industry,
                confidence=rag_context.confidence if rag_context else 0.5,
                response_snippet=full_response[:200]
            )
    
    async def _generate_anthropic(
        self,
        message: str,
        history: Optional[List[Dict]],
        system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Generate response using Claude."""
        messages = []
        
        # Add conversation history
        if history:
            for msg in history[-10:]:  # Keep last 10 messages for context
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        try:
            async with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            yield "I'm having a moment - could you try that again?"
    
    async def _generate_openai(
        self,
        message: str,
        history: Optional[List[Dict]],
        system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Generate response using GPT-4."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if history:
            for msg in history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            yield "I'm having a moment - could you try that again?"
    
    async def _generate_fallback(self, message: str) -> str:
        """Fallback response when no LLM is available."""
        message_lower = message.lower()
        
        # Very basic pattern matching as last resort
        if any(word in message_lower for word in ["hi", "hello", "hey"]):
            return (
                "Hey! I'm Nexus from Barrios A2I. We build automation systems "
                "that help companies scale. What industry are you in?"
            )
        
        if any(word in message_lower for word in ["dental", "dentist"]):
            return (
                "For dental practices, we automate things like patient reminders, "
                "insurance verification, intake forms, and review requests. "
                "What's taking up most of your team's time?"
            )
        
        if any(word in message_lower for word in ["price", "cost", "how much"]):
            return (
                "Projects range from $50K-$300K depending on complexity. "
                "What problem are you trying to solve?"
            )
        
        # Default
        return (
            "Tell me more about your business - what industry are you in "
            "and what's eating up too much of your team's time?"
        )
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.provider in ("anthropic", "openai")
    
    def get_provider(self) -> str:
        """Get current provider name."""
        return self.provider
    
    def is_rag_enabled(self) -> bool:
        """Check if RAG is enabled and initialized."""
        return self.oracle_client is not None and self.oracle_client._initialized


# =============================================================================
# SINGLETON INSTANCE (Backwards Compatible)
# =============================================================================

_brain: Optional[NexusBrain] = None


def get_nexus_brain() -> NexusBrain:
    """Get or create Nexus Brain singleton."""
    global _brain
    if _brain is None:
        _brain = NexusBrain()
    return _brain


async def initialize_nexus_brain_rag(
    qdrant_url: str = None,
    redis_url: str = None,
    rabbitmq_url: str = None
) -> bool:
    """
    Initialize RAG for the singleton brain.
    
    Call this at startup to enable Research Oracle integration.
    """
    brain = get_nexus_brain()
    return await brain.initialize_rag(qdrant_url, redis_url, rabbitmq_url)


async def generate_nexus_response(
    message: str,
    conversation_history: Optional[List[Dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convenience function to generate response.
    
    Usage:
        async for chunk in generate_nexus_response("Hello"):
            print(chunk, end="")
    """
    brain = get_nexus_brain()
    async for chunk in brain.generate_response(message, conversation_history):
        yield chunk


# =============================================================================
# HEALTH CHECK & DIAGNOSTICS
# =============================================================================

async def get_brain_status() -> Dict[str, Any]:
    """
    Get comprehensive status of Nexus Brain.
    
    Returns dict with:
    - provider: LLM provider in use
    - llm_available: Whether LLM is functional
    - rag_enabled: Whether RAG is initialized
    - qdrant_connected: Qdrant connection status
    - redis_connected: Redis connection status
    - rabbitmq_connected: RabbitMQ connection status
    """
    brain = get_nexus_brain()
    
    status = {
        "provider": brain.get_provider(),
        "llm_available": brain.is_available(),
        "rag_enabled": brain.is_rag_enabled(),
        "qdrant_connected": False,
        "redis_connected": False,
        "rabbitmq_connected": False
    }
    
    if brain.oracle_client and brain.oracle_client._initialized:
        try:
            # Test Qdrant
            await brain.oracle_client._qdrant.get_collections()
            status["qdrant_connected"] = True
        except:
            pass
        
        try:
            # Test Redis
            await brain.oracle_client._redis.ping()
            status["redis_connected"] = True
        except:
            pass
        
        try:
            # Test RabbitMQ (connection is maintained)
            status["rabbitmq_connected"] = (
                brain.oracle_client._rabbitmq_connection is not None
                and not brain.oracle_client._rabbitmq_connection.is_closed
            )
        except:
            pass
    
    return status


# =============================================================================
# CLI FOR TESTING
# =============================================================================

async def _test_conversation():
    """Test conversation flow."""
    print("\n🧠 NEXUS BRAIN v4.0 - TEST MODE\n")
    
    brain = get_nexus_brain()
    print(f"Provider: {brain.get_provider()}")
    
    # Try to initialize RAG
    rag_success = await initialize_nexus_brain_rag()
    print(f"RAG Enabled: {rag_success}")
    
    # Get status
    status = await get_brain_status()
    print(f"Status: {json.dumps(status, indent=2)}\n")
    
    # Test messages
    test_messages = [
        "Hey, I run a dental practice",
        "What can you automate for us?",
        "How much does it cost?"
    ]
    
    history = []
    for msg in test_messages:
        print(f"\n👤 You: {msg}")
        print("🤖 Nexus: ", end="", flush=True)
        
        response_text = ""
        async for chunk in generate_nexus_response(msg, history):
            print(chunk, end="", flush=True)
            response_text += chunk
        
        print("\n")
        
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response_text})
    
    # Cleanup
    await brain.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    asyncio.run(_test_conversation())
