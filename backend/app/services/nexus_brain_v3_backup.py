"""
Nexus Brain v3.0 - LLM-Powered Sales Consultant
================================================
Replaces hardcoded pattern matching with actual AI understanding.
Uses Claude API for intelligent, context-aware conversations.
"""

import os
import logging
from typing import AsyncGenerator, Optional, List, Dict

from ..config import settings  # Load .env via pydantic-settings

logger = logging.getLogger("nexus_brain")

# =============================================================================
# SYSTEM PROMPT - THE BRAIN
# =============================================================================

NEXUS_SYSTEM_PROMPT = """You are Nexus, the AI sales consultant for Barrios A2I, a premium automation agency. Your job is to have natural conversations with potential clients, understand their business needs, and guide qualified prospects toward booking a strategy call.

## YOUR PERSONALITY

You are:
- **Friendly but professional** - Like talking to a smart friend who happens to be a business consultant
- **Curious** - You genuinely want to understand their business before pitching
- **Confident** - You know your stuff, but you're not arrogant
- **Concise** - You respect people's time. Keep responses to 2-4 sentences unless more detail is requested.
- **Sales-aware** - Your goal is to qualify leads and book calls, not just chat

You are NOT:
- Robotic or overly formal
- Pushy or aggressive
- Technical (never use jargon)
- Defensive about how things work

## WHAT BARRIOS A2I DOES

Barrios A2I builds custom automation systems for growing companies. Core offerings:

1. **Business Process Automation** - Automate repetitive tasks so teams can focus on high-value work
2. **AI-Powered Assistants** - Custom chatbots and agents for customer service, lead qualification, internal ops
3. **Content Automation** - Generate marketing content, social posts, reports at scale
4. **Video Production** - AI-generated commercials and promotional videos in minutes, not weeks
5. **Data & Research Systems** - Competitive intelligence, market analysis, lead enrichment

**Pricing:** Projects range from $50K-$300K depending on complexity. Also open to equity partnerships for the right fit.

**Ideal Clients:** Growing companies (usually $2M-$50M revenue) that need to scale operations without proportionally scaling headcount.

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
# LLM CLIENT SETUP
# =============================================================================

# Try to import anthropic, fall back to openai
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed, will try openai")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed")


class NexusBrain:
    """
    LLM-powered brain for Nexus sales assistant.

    Supports Claude (preferred) or OpenAI as fallback.
    """

    def __init__(self):
        self.provider = None
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize LLM client based on available API keys."""
        # Debug logging
        logger.info(f"ANTHROPIC_AVAILABLE: {ANTHROPIC_AVAILABLE}")
        logger.info(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")

        # Try Anthropic first (use settings which loads .env)
        anthropic_key = settings.ANTHROPIC_API_KEY
        # Also check os.environ directly as fallback
        import os
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        logger.info(f"settings.ANTHROPIC_API_KEY present: {bool(anthropic_key)}")
        logger.info(f"os.environ ANTHROPIC_API_KEY present: {bool(env_key)}")
        # Use whichever is available
        if not anthropic_key and env_key:
            anthropic_key = env_key
            logger.info("Using os.environ fallback for ANTHROPIC_API_KEY")
        if anthropic_key:
            logger.info(f"ANTHROPIC_API_KEY starts with: {anthropic_key[:20]}...")

        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            self.provider = "anthropic"
            logger.info("Nexus Brain initialized with Claude (Anthropic)")
            return

        # Try OpenAI
        openai_key = settings.OPENAI_API_KEY
        logger.info(f"OPENAI_API_KEY present: {bool(openai_key)}")
        if openai_key and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=openai_key)
            self.provider = "openai"
            logger.info("Nexus Brain initialized with GPT-4 (OpenAI)")
            return

        # No LLM available
        self.provider = "fallback"
        logger.warning("No LLM API key found - using fallback responses")

    async def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.

        Args:
            message: User's message
            conversation_history: Previous messages in format [{"role": "user/assistant", "content": "..."}]

        Yields:
            Text chunks as they're generated
        """
        if self.provider == "anthropic":
            async for chunk in self._generate_anthropic(message, conversation_history):
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._generate_openai(message, conversation_history):
                yield chunk
        else:
            # Fallback - yield a generic response
            yield await self._generate_fallback(message)

    async def _generate_anthropic(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
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
                system=NEXUS_SYSTEM_PROMPT,
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
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response using GPT-4."""
        messages = [{"role": "system", "content": NEXUS_SYSTEM_PROMPT}]

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

        if any(word in message_lower for word in ["law", "attorney", "lawyer"]):
            return (
                "For law firms, we automate client intake, document assembly, "
                "deadline tracking, and case updates. What practice area are you in?"
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


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_brain: Optional[NexusBrain] = None


def get_nexus_brain() -> NexusBrain:
    """Get or create Nexus Brain singleton."""
    global _brain
    if _brain is None:
        _brain = NexusBrain()
    return _brain


async def generate_nexus_response(
    message: str,
    conversation_history: Optional[List[Dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convenience function to generate Nexus response.

    Args:
        message: User's message
        conversation_history: Previous messages

    Yields:
        Text chunks
    """
    brain = get_nexus_brain()
    async for chunk in brain.generate_response(message, conversation_history):
        yield chunk
