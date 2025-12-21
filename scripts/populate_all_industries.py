#!/usr/bin/env python3
"""
BATCH INDUSTRY POPULATION SCRIPT
=================================
Populates Qdrant nexus_knowledge collection with research data for all
Barrios A2I target industries.

Pipeline per industry:
1. Query Perplexity for pain points, costs, automation opportunities
2. Extract structured data with Claude Haiku
3. Embed with OpenAI API (text-embedding-3-small, 1536 dims)
4. Store in Qdrant nexus_knowledge collection

Usage:
    python scripts/populate_all_industries.py                    # All industries
    python scripts/populate_all_industries.py --industry dental  # Single industry
    python scripts/populate_all_industries.py --dry-run          # Preview only
    python scripts/populate_all_industries.py --skip-existing    # Skip if data exists

Environment Variables Required:
    PERPLEXITY_API_KEY  - Perplexity AI API key
    ANTHROPIC_API_KEY   - Anthropic API key (for Haiku)
    OPENAI_API_KEY      - OpenAI API key (for embeddings)
    QDRANT_URL          - Qdrant Cloud URL
    QDRANT_API_KEY      - Qdrant Cloud API key

Estimated Time: ~2-3 minutes per industry
Estimated Cost: ~$0.05-0.10 per industry (Perplexity + Haiku + OpenAI)

Author: Barrios A2I
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# TARGET INDUSTRIES (Barrios A2I verticals)
# =============================================================================

TARGET_INDUSTRIES = {
    "marketing_agencies": {
        "display_name": "Marketing Agencies",
        "keywords": ["marketing agency", "ad agency", "digital agency", "creative agency"],
        "research_focus": "client retention, reporting automation, lead generation, project management"
    },
    "ecommerce_managers": {
        "display_name": "E-commerce Managers",
        "keywords": ["ecommerce", "online store", "shopify", "amazon seller"],
        "research_focus": "inventory management, customer support, order processing, returns"
    },
    "b2b_saas": {
        "display_name": "B2B SaaS Companies",
        "keywords": ["saas", "software company", "tech startup", "subscription business"],
        "research_focus": "customer onboarding, churn reduction, support tickets, sales demos"
    },
    "real_estate": {
        "display_name": "Real Estate Agencies",
        "keywords": ["realtor", "real estate", "property management", "broker"],
        "research_focus": "lead follow-up, showing scheduling, document management, market analysis"
    },
    "legal_practices": {
        "display_name": "Law Firms",
        "keywords": ["law firm", "attorney", "lawyer", "legal practice"],
        "research_focus": "client intake, billing, document review, case management"
    },
    "healthcare_clinics": {
        "display_name": "Healthcare Clinics",
        "keywords": ["medical practice", "clinic", "doctor office", "healthcare"],
        "research_focus": "patient scheduling, insurance verification, follow-ups, records"
    },
    "insurance_agencies": {
        "display_name": "Insurance Agencies",
        "keywords": ["insurance agency", "insurance broker", "insurance sales"],
        "research_focus": "quote generation, policy renewals, claims processing, lead management"
    },
    "financial_services": {
        "display_name": "Financial Services",
        "keywords": ["financial advisor", "wealth management", "accounting firm", "CPA"],
        "research_focus": "client reporting, compliance, tax preparation, document collection"
    },
    "recruiting_agencies": {
        "display_name": "Recruiting Agencies",
        "keywords": ["staffing agency", "recruiting firm", "headhunter", "talent acquisition"],
        "research_focus": "candidate sourcing, interview scheduling, client communication, ATS"
    },
    "construction_companies": {
        "display_name": "Construction Companies",
        "keywords": ["contractor", "construction company", "builder", "general contractor"],
        "research_focus": "project bidding, subcontractor management, permitting, scheduling"
    },
    "home_services": {
        "display_name": "Home Services (HVAC, Plumbing, Electrical)",
        "keywords": ["hvac", "plumber", "electrician", "home services", "handyman"],
        "research_focus": "dispatch scheduling, customer communication, invoicing, reviews"
    },
    "fitness_gyms": {
        "display_name": "Fitness & Gyms",
        "keywords": ["gym", "fitness center", "personal trainer", "yoga studio"],
        "research_focus": "member retention, class scheduling, billing, lead conversion"
    },
    "restaurants": {
        "display_name": "Restaurants & Hospitality",
        "keywords": ["restaurant", "bar", "cafe", "hospitality", "food service"],
        "research_focus": "reservations, inventory, staff scheduling, customer feedback"
    },
    "auto_dealerships": {
        "display_name": "Auto Dealerships",
        "keywords": ["car dealership", "auto dealer", "vehicle sales", "automotive"],
        "research_focus": "lead follow-up, inventory management, service scheduling, financing"
    },
    "property_management": {
        "display_name": "Property Management",
        "keywords": ["property manager", "landlord", "rental management", "HOA"],
        "research_focus": "tenant communication, maintenance requests, rent collection, leasing"
    },
    "dental_practices": {
        "display_name": "Dental Practices",
        "keywords": ["dental", "dentist", "orthodontist", "dental practice"],
        "research_focus": "patient scheduling, insurance claims, recall campaigns, treatment plans"
    },
}


# =============================================================================
# RESEARCH QUERIES (per industry)
# =============================================================================

RESEARCH_QUERY_TEMPLATES = [
    "{industry} business biggest pain points time wasters operational costs 2025",
    "{industry} automation opportunities ROI case studies real examples with numbers",
    "{industry} decision makers software buying budget typical spend",
    "{industry} common objections technology adoption resistance how to overcome",
    "{industry} industry terminology jargon glossary key metrics KPIs",
    "{industry} staffing costs hiring training turnover statistics 2025",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration loaded from environment."""
    perplexity_api_key: str
    anthropic_api_key: str
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "Config":
        """Load from environment variables."""
        missing = []

        perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        if not perplexity_key:
            missing.append("PERPLEXITY_API_KEY")

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not anthropic_key:
            missing.append("ANTHROPIC_API_KEY")

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            missing.append("OPENAI_API_KEY")

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if not qdrant_url:
            missing.append("QDRANT_URL")

        if missing:
            logger.error(f"Missing environment variables: {', '.join(missing)}")
            sys.exit(1)

        return cls(
            perplexity_api_key=perplexity_key,
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        )


# =============================================================================
# API CLIENTS
# =============================================================================

class PerplexityClient:
    """Perplexity AI client for web research."""

    BASE_URL = "https://api.perplexity.ai"
    MODEL = "sonar"

    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def search(self, query: str) -> Dict[str, Any]:
        """Execute search and return content + citations."""
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.MODEL,
                "messages": [{"role": "user", "content": query}],
                "return_citations": True,
                "search_recency_filter": "month",
            },
        )
        response.raise_for_status()
        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "citations": data.get("citations", []),
            "tokens": data.get("usage", {}).get("total_tokens", 0),
        }

    async def close(self):
        await self.client.aclose()


class HaikuProcessor:
    """Claude Haiku for structured extraction."""

    BASE_URL = "https://api.anthropic.com"
    MODEL = "claude-3-5-haiku-20241022"

    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def extract_structured(
        self,
        raw_research: str,
        industry: str,
        display_name: str,
    ) -> Dict[str, Any]:
        """Extract structured knowledge from raw research."""

        prompt = f"""You are extracting structured business intelligence about the {display_name} industry.

Analyze this research and extract:

1. PAIN POINTS (5-8): Specific problems with dollar costs
   - Each must have a cost impact (e.g., "$5,000-$10,000 per incident")

2. AUTOMATION OPPORTUNITIES (4-6): What can be automated
   - Each must have ROI estimate (e.g., "30-50% time savings")

3. INDUSTRY TERMINOLOGY (5-10): Key jargon and metrics
   - Terms professionals use daily

4. ROI DATA (3-5): Statistics with before/after numbers
   - Specific percentages and dollar amounts

5. SALES ANGLES (3-5): Conversation starters for sales calls
   - Questions that uncover pain points

Be SPECIFIC. Use real numbers, percentages, dollar amounts from the research.
Generic insights are worthless - we need actionable specifics.

Research text:
{raw_research}

Respond with valid JSON in this exact format:
{{
    "pain_points": [
        {{"issue": "description", "cost_impact": "$X-Y per month/year/incident"}}
    ],
    "automation_opportunities": [
        {{"opportunity": "what to automate", "roi": "X% savings or $Y value", "difficulty": "easy|medium|hard"}}
    ],
    "terminology": {{"term": "definition"}},
    "roi_data": [
        {{"metric": "what measured", "before": "X", "after": "Y"}}
    ],
    "sales_angles": ["question 1", "question 2"]
}}"""

        response = await self.client.post(
            "/v1/messages",
            json={
                "model": self.MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract text content
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break

        # Parse JSON from response
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Haiku response: {e}")

        # Return empty structure on failure
        return {
            "pain_points": [],
            "automation_opportunities": [],
            "terminology": {},
            "roi_data": [],
            "sales_angles": [],
        }

    async def close(self):
        await self.client.aclose()


class OpenAIEmbedder:
    """OpenAI embeddings client."""

    URL = "https://api.openai.com/v1/embeddings"
    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = await self.client.post(
            self.URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.MODEL,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
            await asyncio.sleep(0.05)  # Rate limit protection
        return embeddings

    async def close(self):
        await self.client.aclose()


class QdrantStore:
    """Qdrant vector store client."""

    COLLECTION = "nexus_knowledge"

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"api-key": api_key} if api_key else {},
        )

    async def count_industry(self, industry: str) -> int:
        """Count points for an industry."""
        try:
            response = await self.client.post(
                f"{self.url}/collections/{self.COLLECTION}/points/count",
                json={
                    "filter": {
                        "must": [{"key": "industry", "match": {"value": industry}}]
                    }
                },
            )
            if response.status_code == 200:
                return response.json().get("result", {}).get("count", 0)
        except Exception:
            pass
        return 0

    async def upsert(self, points: List[Dict[str, Any]]) -> int:
        """Upsert points to collection."""
        if not points:
            return 0

        response = await self.client.put(
            f"{self.url}/collections/{self.COLLECTION}/points",
            json={"points": points},
        )
        response.raise_for_status()
        return len(points)

    async def close(self):
        await self.client.aclose()


# =============================================================================
# CHUNKING
# =============================================================================

def create_chunks(
    industry: str,
    display_name: str,
    structured_data: Dict[str, Any],
    citations: List[str],
) -> List[Dict[str, Any]]:
    """Convert structured data to knowledge chunks."""
    chunks = []
    timestamp = datetime.utcnow().isoformat()

    # Pain point chunks
    for pp in structured_data.get("pain_points", []):
        content = f"{display_name} Pain Point: {pp.get('issue', '')}"
        if pp.get("cost_impact"):
            content += f"\nCost Impact: {pp['cost_impact']}"

        chunks.append({
            "content": content,
            "type": "pain_point",
            "industry": industry,
            "quality_score": 0.9,
            "citations": citations[:3],
            "processed_at": timestamp,
        })

    # Automation opportunity chunks
    for opp in structured_data.get("automation_opportunities", []):
        content = f"{display_name} Automation Opportunity: {opp.get('opportunity', '')}"
        if opp.get("roi"):
            content += f"\nExpected ROI: {opp['roi']}"
        if opp.get("difficulty"):
            content += f"\nDifficulty: {opp['difficulty']}"

        chunks.append({
            "content": content,
            "type": "automation",
            "industry": industry,
            "quality_score": 0.9,
            "citations": citations[:3],
            "processed_at": timestamp,
        })

    # ROI data chunks
    roi_data = structured_data.get("roi_data", [])
    if roi_data:
        content = f"{display_name} ROI Statistics:\n"
        for roi in roi_data:
            content += f"• {roi.get('metric', '')}: {roi.get('before', '')} → {roi.get('after', '')}\n"

        chunks.append({
            "content": content.strip(),
            "type": "roi",
            "industry": industry,
            "quality_score": 0.9,
            "citations": citations[:3],
            "processed_at": timestamp,
        })

    # Terminology chunk
    terminology = structured_data.get("terminology", {})
    if terminology:
        content = f"{display_name} Industry Terminology:\n"
        for term, definition in terminology.items():
            content += f"• {term}: {definition}\n"

        chunks.append({
            "content": content.strip(),
            "type": "terminology",
            "industry": industry,
            "quality_score": 0.9,
            "citations": citations[:3],
            "processed_at": timestamp,
        })

    # Sales angles chunk
    sales_angles = structured_data.get("sales_angles", [])
    if sales_angles:
        content = f"{display_name} Conversation Starters:\n"
        for angle in sales_angles:
            content += f"• {angle}\n"

        chunks.append({
            "content": content.strip(),
            "type": "script",
            "industry": industry,
            "quality_score": 0.9,
            "citations": citations[:3],
            "processed_at": timestamp,
        })

    return chunks


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def process_industry(
    industry_key: str,
    industry_info: Dict[str, Any],
    perplexity: PerplexityClient,
    haiku: HaikuProcessor,
    embedder: OpenAIEmbedder,
    qdrant: QdrantStore,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process a single industry through the full pipeline."""

    display_name = industry_info["display_name"]
    start_time = time.time()

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting {display_name}")

    result = {
        "industry": industry_key,
        "display_name": display_name,
        "success": False,
        "chunks_created": 0,
        "duration_seconds": 0,
        "error": None,
    }

    try:
        # Step 1: Perplexity Research
        logger.info(f"  [1/4] Researching with Perplexity...")

        all_content = []
        all_citations = []

        queries = [
            t.format(industry=display_name)
            for t in RESEARCH_QUERY_TEMPLATES
        ]

        for i, query in enumerate(queries):
            if dry_run:
                logger.info(f"    Query {i+1}: {query[:60]}...")
                continue

            try:
                search_result = await perplexity.search(query)
                all_content.append(search_result["content"])
                all_citations.extend(search_result.get("citations", []))
                logger.info(f"    Query {i+1}/{len(queries)} complete")
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.warning(f"    Query {i+1} failed: {e}")

        if dry_run:
            result["success"] = True
            result["chunks_created"] = 0
            result["duration_seconds"] = time.time() - start_time
            return result

        if not all_content:
            result["error"] = "No research content retrieved"
            return result

        raw_research = "\n\n---\n\n".join(all_content)
        unique_citations = list(set(all_citations))[:10]

        logger.info(f"  [2/4] Extracting structured data with Haiku...")
        structured_data = await haiku.extract_structured(
            raw_research,
            industry_key,
            display_name,
        )

        # Create chunks
        logger.info(f"  [3/4] Creating knowledge chunks...")
        chunks = create_chunks(
            industry_key,
            display_name,
            structured_data,
            unique_citations,
        )

        if not chunks:
            result["error"] = "No chunks created from structured data"
            return result

        logger.info(f"    Created {len(chunks)} chunks")

        # Embed chunks
        logger.info(f"  [4/4] Embedding and storing in Qdrant...")
        texts = [c["content"] for c in chunks]
        embeddings = await embedder.embed_batch(texts)

        # Prepare Qdrant points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append({
                "id": str(uuid4()),
                "vector": embedding,
                "payload": chunk,
            })

        # Upsert to Qdrant
        count = await qdrant.upsert(points)
        logger.info(f"    Stored {count} chunks in Qdrant")

        result["success"] = True
        result["chunks_created"] = count
        result["duration_seconds"] = round(time.time() - start_time, 1)

        logger.info(
            f"  ✅ {display_name} complete: "
            f"{count} chunks in {result['duration_seconds']}s"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ❌ {display_name} failed: {e}")

    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Populate Qdrant with industry research data"
    )
    parser.add_argument(
        "--industry",
        type=str,
        help="Process single industry (e.g., 'dental_practices')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip industries that already have data",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available industries",
    )
    args = parser.parse_args()

    # List industries
    if args.list:
        print("\nAvailable Industries:")
        print("=" * 50)
        for key, info in TARGET_INDUSTRIES.items():
            print(f"  {key:25} - {info['display_name']}")
        print(f"\nTotal: {len(TARGET_INDUSTRIES)} industries")
        return

    print("=" * 60)
    print("BARRIOS A2I - INDUSTRY KNOWLEDGE POPULATION")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Load config
    config = Config.from_env()

    print(f"Qdrant URL: {config.qdrant_url[:50]}...")
    print(f"Target Collection: nexus_knowledge")

    # Initialize clients
    perplexity = PerplexityClient(config.perplexity_api_key)
    haiku = HaikuProcessor(config.anthropic_api_key)
    embedder = OpenAIEmbedder(config.openai_api_key)
    qdrant = QdrantStore(config.qdrant_url, config.qdrant_api_key)

    # Determine industries to process
    if args.industry:
        if args.industry not in TARGET_INDUSTRIES:
            logger.error(f"Unknown industry: {args.industry}")
            logger.info(f"Available: {', '.join(TARGET_INDUSTRIES.keys())}")
            sys.exit(1)
        industries = {args.industry: TARGET_INDUSTRIES[args.industry]}
    else:
        industries = TARGET_INDUSTRIES

    print(f"Industries to process: {len(industries)}")
    print(f"Estimated time: {len(industries) * 2}-{len(industries) * 3} minutes")
    print()

    # Process each industry
    results = []
    total_chunks = 0

    for i, (industry_key, industry_info) in enumerate(industries.items(), 1):
        print(f"\n[{i}/{len(industries)}] {industry_info['display_name']}")
        print("-" * 40)

        # Check existing data
        if args.skip_existing and not args.dry_run:
            existing = await qdrant.count_industry(industry_key)
            if existing > 0:
                logger.info(f"  ⏭️  Skipping - {existing} chunks already exist")
                results.append({
                    "industry": industry_key,
                    "display_name": industry_info["display_name"],
                    "success": True,
                    "chunks_created": 0,
                    "skipped": True,
                })
                continue

        result = await process_industry(
            industry_key,
            industry_info,
            perplexity,
            haiku,
            embedder,
            qdrant,
            args.dry_run,
        )
        results.append(result)
        total_chunks += result.get("chunks_created", 0)

        # Delay between industries
        if i < len(industries) and not args.dry_run:
            await asyncio.sleep(2)

    # Cleanup
    await perplexity.close()
    await haiku.close()
    await embedder.close()
    await qdrant.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if r.get("success"))
    failed = sum(1 for r in results if not r.get("success"))
    skipped = sum(1 for r in results if r.get("skipped"))

    print(f"Processed:     {len(results)} industries")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"Skipped:       {skipped}")
    print(f"Total Chunks:  {total_chunks}")

    if failed > 0:
        print("\nFailed Industries:")
        for r in results:
            if not r.get("success") and not r.get("skipped"):
                print(f"  - {r['display_name']}: {r.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE - No changes made")
    else:
        print("POPULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
