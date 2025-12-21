"""
SEMANTIC ROUTER
===============
Zero-shot industry classification using vector similarity.
Replaces brittle regex with semantic intent matching.

How it works:
1. On startup: Embed descriptions of all 16 industries
2. On request: Embed user message, find closest industry
3. If confidence < 0.45: Fall back to regex

Benefits:
- "Oral health provider" → dental_practices (regex would miss this)
- "Tooth doctor" → dental_practices (regex would miss this)
- "I sell software subscriptions" → b2b_saas
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nexus.semantic_router")

# Industry "DNA" - semantic descriptions for classification
INDUSTRY_DESCRIPTIONS = {
    "dental_practices": "Dentist, orthodontist, oral surgery, dental hygiene, teeth cleaning, dental practice management, patient scheduling, dental insurance.",
    "marketing_agencies": "Digital marketing, SEO, PPC, advertising agency, creative studio, lead generation, social media marketing, content marketing.",
    "ecommerce": "Online store, Shopify, Amazon seller, dropshipping, DTC brand, cart abandonment, e-commerce, product listings.",
    "b2b_saas": "Software as a service, B2B tech, startup, platform, recurring revenue, enterprise software, subscription model.",
    "real_estate": "Realtor, property listings, real estate agent, brokerage, housing market, buying homes, commercial real estate.",
    "legal_practices": "Law firm, attorney, lawyer, legal advice, litigation, personal injury, estate planning, contracts.",
    "healthcare_clinics": "Medical clinic, doctor office, patient care, healthcare provider, urgent care, primary care physician.",
    "insurance_agencies": "Insurance broker, underwriting, policy sales, claims processing, risk management, life insurance, auto insurance.",
    "financial_services": "Financial advisor, wealth management, accounting, CPA, investment firm, tax preparation, bookkeeping.",
    "recruiting_agencies": "Staffing agency, headhunter, talent acquisition, recruitment, HR services, executive search.",
    "construction": "General contractor, construction company, building, remodeling, renovation projects, home builder.",
    "home_services": "HVAC, plumbing, electrician, roofing, landscaping, cleaning services, handyman, pest control.",
    "fitness_gyms": "Gym owner, personal training, fitness studio, yoga, crossfit, health club, wellness center.",
    "restaurants": "Restaurant, cafe, food service, bar, hospitality, catering, menu management, food delivery.",
    "auto_dealerships": "Car dealership, automotive sales, vehicle inventory, showroom, test drives, used cars.",
    "property_management": "Landlord, tenant management, rental properties, leasing, facility maintenance, apartment complex.",
}

# Fallback regex patterns (safety net)
REGEX_FALLBACK = {
    "dental_practices": [r"\bdental\b", r"\bdentist\b", r"\bdds\b", r"\borthodont"],
    "marketing_agencies": [r"\bmarketing\b", r"\bagency\b", r"\bseo\b", r"\bppc\b"],
    "ecommerce": [r"\becommerce\b", r"\be-commerce\b", r"\bshopify\b", r"\bamazon\s+sell"],
    "b2b_saas": [r"\bsaas\b", r"\bb2b\b", r"\bsoftware\s+company"],
    "real_estate": [r"\breal\s+estate\b", r"\brealtor\b", r"\bproperty\b"],
    "legal_practices": [r"\blaw\s+firm\b", r"\battorney\b", r"\blawyer\b"],
    "healthcare_clinics": [r"\bclinic\b", r"\bhealthcare\b", r"\bmedical\b"],
    "insurance_agencies": [r"\binsurance\b"],
    "financial_services": [r"\bfinancial\b", r"\baccounting\b", r"\bcpa\b"],
    "recruiting_agencies": [r"\brecruit", r"\bstaffing\b", r"\bheadhunt"],
    "construction": [r"\bconstruction\b", r"\bcontractor\b", r"\bbuilder\b"],
    "home_services": [r"\bhvac\b", r"\bplumb", r"\belectric", r"\broofing\b"],
    "fitness_gyms": [r"\bgym\b", r"\bfitness\b", r"\bpersonal\s+train"],
    "restaurants": [r"\brestaurant\b", r"\bcafe\b", r"\bcatering\b"],
    "auto_dealerships": [r"\bcar\s+dealer", r"\bauto\s+dealer", r"\bautomotive\b"],
    "property_management": [r"\bproperty\s+manag", r"\blandlord\b", r"\btenant\b"],
}


class SemanticRouter:
    """
    Semantic industry classifier using embedding similarity.
    """

    def __init__(self):
        self.industry_vectors: Dict[str, List[float]] = {}
        self.is_ready = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Pre-compute industry embeddings on startup."""
        async with self._lock:
            if self.is_ready:
                return True

            try:
                from .nexus_rag import get_rag_client

                rag = await get_rag_client()
                if not rag.enabled:
                    logger.warning("Semantic Router disabled: RAG client not ready")
                    return False

                logger.info("Semantic Router: Embedding 16 industries...")

                for industry, description in INDUSTRY_DESCRIPTIONS.items():
                    vector = await rag._embed(description)
                    self.industry_vectors[industry] = vector

                self.is_ready = True
                logger.info("Semantic Router: Initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Semantic Router initialization failed: {e}")
                return False

    async def classify(self, message: str) -> Tuple[Optional[str], float]:
        """
        Classify user message into an industry.

        Args:
            message: User's input message

        Returns:
            Tuple of (industry_name, confidence_score)
            confidence_score is cosine similarity (0.0 to 1.0)
        """
        if not self.is_ready or not self.industry_vectors:
            # Fall back to regex if router not ready
            return self._regex_classify(message), 0.0

        try:
            from .nexus_rag import get_rag_client

            rag = await get_rag_client()
            query_vector = await rag._embed(message)

            best_industry = None
            best_score = -1.0

            # Cosine similarity (OpenAI embeddings are normalized)
            q_vec = np.array(query_vector)

            for industry, i_vec in self.industry_vectors.items():
                score = float(np.dot(q_vec, np.array(i_vec)))
                if score > best_score:
                    best_score = score
                    best_industry = industry

            # Confidence threshold
            if best_score < 0.45:
                logger.info(f"Semantic confidence low ({best_score:.2f}), trying regex fallback")
                regex_result = self._regex_classify(message)
                if regex_result:
                    return regex_result, 0.5  # Medium confidence for regex match
                return best_industry, best_score

            logger.info(f"Semantic Router: {best_industry} (confidence: {best_score:.2f})")
            return best_industry, best_score

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._regex_classify(message), 0.0

    def _regex_classify(self, message: str) -> Optional[str]:
        """Fallback regex classification."""
        message_lower = message.lower()

        for industry, patterns in REGEX_FALLBACK.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return industry

        return None


# Singleton
_router: Optional[SemanticRouter] = None


async def get_semantic_router() -> SemanticRouter:
    """Get or initialize the semantic router singleton."""
    global _router
    if _router is None:
        _router = SemanticRouter()
        await _router.initialize()
    return _router


async def detect_industry_semantic(message: str) -> Tuple[Optional[str], float]:
    """
    Convenience function for industry detection.

    Returns:
        Tuple of (industry, confidence)
    """
    router = await get_semantic_router()
    return await router.classify(message)
