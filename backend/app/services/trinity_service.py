"""
TRINITY SERVICE v2.0 - PRODUCTION WRAPPER
==========================================
Wraps Gary's EXISTING production Trinity system (intelligence_trinity_system.py)
instead of using mock data.

Path: C:/Users/gary/nexus_assistant_unified/backend/app/services/trinity_service.py

This version:
1. Imports the REAL IntelligenceTrinityOrchestrator
2. Uses actual Perplexity/NewsAPI integrations
3. Falls back to mock only if real system unavailable
"""

import asyncio
import logging
import time
import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("nexus.trinity")

# =============================================================================
# ATTEMPT TO IMPORT REAL TRINITY SYSTEM
# =============================================================================

# Add the path to the real Trinity system
TRINITY_PATH = os.getenv("TRINITY_SYSTEM_PATH", 
    r"C:\Users\gary\python-commercial-video-agents\ragnarok_v6_legendary")

if TRINITY_PATH not in sys.path:
    sys.path.insert(0, TRINITY_PATH)

try:
    from intelligence_trinity_system import (
        IntelligenceTrinityOrchestrator,
        IntelligenceRequest,
        IntelligenceResponse as TrinityIntelResponse,
        TrendRequest,
        TrendResponse,
        MarketAnalysisRequest,
        MarketAnalysisResponse,
        CompetitorTrackingRequest,
        CompetitorTrackingResponse,
        TimeHorizon,
    )
    TRINITY_AVAILABLE = True
    logger.info("✅ Real Trinity system imported successfully")
except ImportError as e:
    TRINITY_AVAILABLE = False
    logger.warning(f"⚠️ Real Trinity system not available, using fallback: {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================

TRINITY_ENABLED = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
TRINITY_TIMEOUT = float(os.getenv("TRINITY_TIMEOUT", "60.0"))


# =============================================================================
# OUTPUT DATA MODELS (For RAGNAROK consumption)
# =============================================================================

@dataclass
class RagnarokMarketContext:
    """Formatted market intelligence for RAGNAROK"""
    trending_topics: List[str]
    recommended_hashtags: List[str]
    optimal_posting_time: str
    competitor_gaps: List[str]
    target_pain_points: List[str]
    recommended_tone: str
    market_sentiment: str
    target_demographics: Dict[str, Any]
    decision_factors: List[str]
    market_size_usd: Optional[float] = None
    growth_rate: Optional[float] = None
    

@dataclass
class MarketIntelligence:
    """Combined market intelligence output"""
    context: RagnarokMarketContext
    executive_summary: str
    key_insights: List[str]
    recommended_actions: List[Dict[str, str]]
    risk_factors: List[str]
    confidence: float
    cost_usd: float
    latency_ms: float
    source: str  # "trinity_production" | "mock_fallback"
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_ragnarok_context(self) -> Dict[str, Any]:
        """Format for RAGNAROK enrichment"""
        return {
            "market_intelligence": {
                "trending_topics": self.context.trending_topics,
                "recommended_hashtags": self.context.recommended_hashtags,
                "optimal_posting_time": self.context.optimal_posting_time,
                "competitor_gaps": self.context.competitor_gaps,
                "target_pain_points": self.context.target_pain_points,
                "recommended_tone": self.context.recommended_tone,
                "market_sentiment": self.context.market_sentiment,
                "target_demographics": self.context.target_demographics,
                "decision_factors": self.context.decision_factors,
                "market_size_usd": self.context.market_size_usd,
                "growth_rate": self.context.growth_rate,
            },
            "strategic_context": {
                "executive_summary": self.executive_summary,
                "key_insights": self.key_insights,
                "recommended_actions": self.recommended_actions,
                "risk_factors": self.risk_factors,
            },
            "enrichment_metadata": {
                "source": self.source,
                "confidence": self.confidence,
                "cost_usd": self.cost_usd,
                "latency_ms": self.latency_ms,
                "generated_at": self.generated_at.isoformat(),
            }
        }


# =============================================================================
# MOCK DATA GENERATOR (Fallback only)
# =============================================================================

def generate_mock_intelligence(
    industry: str, 
    product: str, 
    audience: str,
    company_name: str = "Barrios A2I"
) -> MarketIntelligence:
    """Generate mock intelligence when real Trinity unavailable"""
    
    # Industry-specific content
    industry_lower = industry.lower()
    
    if "saas" in industry_lower or "b2b" in industry_lower:
        hashtags = ["#SaaS", "#B2B", "#CloudSoftware", "#ProductLed", "#TechStartup"]
        pain_points = ["Manual processes", "Scaling challenges", "Integration complexity"]
        tone = "professional"
    elif "ai" in industry_lower or "automation" in industry_lower:
        hashtags = ["#AIAutomation", "#MachineLearning", "#FutureOfWork", "#Productivity"]
        pain_points = ["Time-consuming tasks", "Inconsistent quality", "High labor costs"]
        tone = "innovative"
    else:
        hashtags = ["#Innovation", "#Growth", "#Technology", "#Success"]
        pain_points = ["Efficiency", "Competition", "Market changes"]
        tone = "balanced"
    
    return MarketIntelligence(
        context=RagnarokMarketContext(
            trending_topics=[f"AI in {industry}", "Digital transformation", "Automation"],
            recommended_hashtags=hashtags,
            optimal_posting_time="morning",
            competitor_gaps=["Personalization", "Speed to value", "Customer stories"],
            target_pain_points=pain_points,
            recommended_tone=tone,
            market_sentiment="positive",
            target_demographics={
                "primary": audience,
                "titles": ["CTO", "VP Engineering", "Director"],
                "company_size": "50-500"
            },
            decision_factors=["ROI", "Ease of use", "Support quality"],
            market_size_usd=50_000_000_000,
            growth_rate=0.08
        ),
        executive_summary=f"Market intelligence for {company_name} in {industry}. Strong growth potential with opportunities in automation and personalization.",
        key_insights=[
            f"Growing AI adoption in {industry}",
            "Competitors focus on enterprise, leaving SMB gap",
            "Speed to value is key differentiator"
        ],
        recommended_actions=[
            {"action": "Focus on automation messaging", "priority": "high", "timeframe": "Q1"},
            {"action": "Build customer case studies", "priority": "medium", "timeframe": "Q1-Q2"}
        ],
        risk_factors=["Economic uncertainty", "New competitors entering market"],
        confidence=0.75,
        cost_usd=0.0,
        latency_ms=100,
        source="mock_fallback"
    )


# =============================================================================
# TRINITY SERVICE - PRODUCTION WRAPPER
# =============================================================================

class TrinityService:
    """
    Production wrapper for the real Intelligence Trinity system.
    
    Uses Gary's existing 1400+ line implementation with:
    - Perplexity API integration
    - NewsAPI, Serper, Twitter APIs
    - Real cost tracking
    - 1.31s latency, 95% accuracy
    
    Falls back to mock only if real system unavailable.
    """
    
    def __init__(self):
        self.enabled = TRINITY_ENABLED
        self.orchestrator: Optional[Any] = None
        self.cache: Dict[str, MarketIntelligence] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize real orchestrator if available
        if TRINITY_AVAILABLE and self.enabled:
            try:
                self.orchestrator = IntelligenceTrinityOrchestrator()
                logger.info("✅ Trinity orchestrator initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Trinity orchestrator: {e}")
                self.orchestrator = None
    
    async def gather_intelligence(
        self,
        industry: str,
        product_description: str,
        target_audience: str,
        company_name: str = "Barrios A2I",
        competitors: Optional[List[str]] = None,
        include_trends: bool = True,
        include_market: bool = True,
        include_competitors: bool = True,
        trace_id: Optional[str] = None,
        use_cache: bool = True
    ) -> MarketIntelligence:
        """
        Gather market intelligence using the REAL Trinity system.
        
        Runs all 3 agents in parallel:
        1. TrendScoutAgent - Emerging trends and market shifts
        2. MarketAnalystAgent - Market conditions and opportunities
        3. CompetitorTrackerAgent - Competitor activities
        
        Returns comprehensive intelligence for RAGNAROK enrichment.
        """
        start_time = time.time()
        cache_key = f"{company_name}:{industry}:{hash(product_description)}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            age = (datetime.now() - cached.generated_at).total_seconds()
            if age < self.cache_ttl:
                logger.info(f"Trinity cache hit | company={company_name}")
                return cached
        
        # If real Trinity not available, use mock
        if not TRINITY_AVAILABLE or not self.orchestrator:
            logger.info("Using mock intelligence (real Trinity not available)")
            mock = generate_mock_intelligence(industry, product_description, target_audience, company_name)
            mock.latency_ms = (time.time() - start_time) * 1000
            return mock
        
        # Use REAL Trinity system
        try:
            request = IntelligenceRequest(
                company_name=company_name,
                industry=industry,
                region="US",
                competitors=competitors or [],
                include_trends=include_trends,
                include_market_analysis=include_market,
                include_competitor_tracking=include_competitors,
                trend_depth=5,
                forecast_horizon=TimeHorizon.SHORT
            )
            
            # Run the real orchestrator
            logger.info(f"🔍 Running REAL Trinity analysis for {company_name}...")
            response = await asyncio.wait_for(
                self.orchestrator.analyze(request),
                timeout=TRINITY_TIMEOUT
            )
            
            # Transform to our output format
            intel = self._transform_trinity_response(response, start_time)
            
            # Cache result
            self.cache[cache_key] = intel
            
            logger.info(
                f"✅ Trinity analysis complete | "
                f"company={company_name} | "
                f"cost=${intel.cost_usd:.2f} | "
                f"latency={intel.latency_ms:.0f}ms"
            )
            
            return intel
            
        except asyncio.TimeoutError:
            logger.error(f"Trinity timeout after {TRINITY_TIMEOUT}s")
            mock = generate_mock_intelligence(industry, product_description, target_audience, company_name)
            mock.latency_ms = (time.time() - start_time) * 1000
            mock.source = "mock_timeout_fallback"
            return mock
            
        except Exception as e:
            logger.error(f"Trinity error: {e}")
            mock = generate_mock_intelligence(industry, product_description, target_audience, company_name)
            mock.latency_ms = (time.time() - start_time) * 1000
            mock.source = "mock_error_fallback"
            return mock
    
    def _transform_trinity_response(
        self, 
        response: 'TrinityIntelResponse',
        start_time: float
    ) -> MarketIntelligence:
        """Transform real Trinity response to our output format"""
        
        # Extract trends data
        trending_topics = []
        recommended_hashtags = []
        optimal_time = "morning"
        
        if response.trends:
            for trend in response.trends.trends[:5]:
                trending_topics.append(trend.topic)
                recommended_hashtags.extend(trend.related_keywords[:3])
            # Get peak time from most viral trend
            if response.trends.trends:
                optimal_time = "morning"  # Could be extracted from trend data
        
        # Extract competitor gaps
        competitor_gaps = []
        if response.competitor_tracking:
            for rec in response.competitor_tracking.recommendations[:5]:
                competitor_gaps.append(rec)
        
        # Extract market data
        market_size = None
        growth_rate = None
        pain_points = []
        decision_factors = []
        sentiment = "neutral"
        demographics = {}
        
        if response.market_analysis:
            market = response.market_analysis
            market_size = market.metrics.market_size_usd
            growth_rate = market.metrics.growth_rate_yoy
            
            # Extract pain points from threats
            for threat in market.threats[:3]:
                pain_points.append(threat.get("title", str(threat)))
            
            # Extract decision factors from opportunities
            for opp in market.opportunities[:3]:
                decision_factors.append(opp.get("title", str(opp)))
            
            sentiment = "positive" if growth_rate > 0.05 else "neutral"
        
        # Determine recommended tone based on competitive position
        tone = "professional"
        if response.competitor_tracking:
            pos = response.competitor_tracking.competitive_position
            if hasattr(pos, 'position_summary'):
                if "leader" in pos.position_summary.lower():
                    tone = "authoritative"
                elif "challenger" in pos.position_summary.lower():
                    tone = "bold"
        
        return MarketIntelligence(
            context=RagnarokMarketContext(
                trending_topics=trending_topics[:5],
                recommended_hashtags=list(set(recommended_hashtags))[:10],
                optimal_posting_time=optimal_time,
                competitor_gaps=competitor_gaps[:5],
                target_pain_points=pain_points[:5],
                recommended_tone=tone,
                market_sentiment=sentiment,
                target_demographics=demographics,
                decision_factors=decision_factors[:5],
                market_size_usd=market_size,
                growth_rate=growth_rate
            ),
            executive_summary=response.executive_summary,
            key_insights=response.key_insights[:5],
            recommended_actions=response.recommended_actions[:5],
            risk_factors=response.risk_factors[:5],
            confidence=response.confidence,
            cost_usd=response.total_cost_usd,
            latency_ms=(time.time() - start_time) * 1000,
            source="trinity_production"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Trinity service health"""
        return {
            "enabled": self.enabled,
            "trinity_available": TRINITY_AVAILABLE,
            "orchestrator_initialized": self.orchestrator is not None,
            "mode": "production" if (TRINITY_AVAILABLE and self.orchestrator) else "mock",
            "cache_size": len(self.cache),
            "cache_ttl_seconds": self.cache_ttl,
        }
    
    def clear_cache(self) -> int:
        """Clear cache and return count"""
        count = len(self.cache)
        self.cache.clear()
        return count


# =============================================================================
# SINGLETON
# =============================================================================

_trinity_service: Optional[TrinityService] = None


def get_trinity_service() -> TrinityService:
    """Get singleton Trinity service instance"""
    global _trinity_service
    if _trinity_service is None:
        _trinity_service = TrinityService()
    return _trinity_service
