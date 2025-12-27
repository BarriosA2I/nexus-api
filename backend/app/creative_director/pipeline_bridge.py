"""
================================================================================
NEXUS - Creative Director Pipeline Bridge
================================================================================
Bridges NEXUS to the 6-agent Creative Director pipeline.

Connects:
- BriefData → Research Agent
- Research → Ideation Agent
- Concepts → Script Agent
- Script → Review Agent
- Approved Script → RAGNAROK Video Generation

Author: Barrios A2I | Version: 6.0.0
================================================================================
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from opentelemetry import trace
from prometheus_client import Counter, Histogram, Gauge

from app.creative_director.session_manager import (
    CreativeDirectorSession,
    BriefData,
    WorkflowPhase,
    ProductionJob,
)

logger = logging.getLogger("nexus.creative_director.bridge")
tracer = trace.get_tracer("nexus.creative_director.bridge")


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

PIPELINE_RUNS = Counter(
    'nexus_cd_pipeline_runs_total',
    'Creative Director pipeline runs',
    ['status', 'phase']
)

PIPELINE_DURATION = Histogram(
    'nexus_cd_pipeline_duration_seconds',
    'Creative Director pipeline duration by phase',
    ['phase'],
    buckets=[5, 10, 30, 60, 120, 300]
)

ACTIVE_PRODUCTIONS = Gauge(
    'nexus_cd_active_productions',
    'Active Creative Director productions'
)


# =============================================================================
# PIPELINE EVENTS
# =============================================================================

class PipelineEvent(Enum):
    """Events emitted during pipeline execution"""
    PIPELINE_STARTED = "pipeline.started"
    RESEARCH_STARTED = "research.started"
    RESEARCH_COMPLETE = "research.complete"
    IDEATION_STARTED = "ideation.started"
    CONCEPTS_GENERATED = "concepts.generated"
    CONCEPT_SELECTED = "concept.selected"
    SCRIPTING_STARTED = "scripting.started"
    SCRIPT_COMPLETE = "script.complete"
    REVIEW_STARTED = "review.started"
    REVIEW_PASSED = "review.passed"
    REVIEW_FAILED = "review.failed"
    PRODUCTION_STARTED = "production.started"
    PRODUCTION_PROGRESS = "production.progress"
    PRODUCTION_COMPLETE = "production.complete"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_ERROR = "pipeline.error"


@dataclass
class PipelineEventData:
    """Event data structure"""
    event: PipelineEvent
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MOCK AGENTS (For when real agents aren't available)
# =============================================================================

class MockResearchAgent:
    """Mock Research Agent for testing"""
    
    async def research(self, brief: BriefData) -> Dict[str, Any]:
        """Simulate market research"""
        await asyncio.sleep(1)  # Simulate work
        
        return {
            "market_size": "Large and growing",
            "audience_insights": {
                "demographics": brief.target_audience,
                "pain_points": ["Time constraints", "Quality concerns", "Cost awareness"],
                "motivations": ["Efficiency", "Results", "Innovation"],
            },
            "competitor_analysis": {
                "competitors": brief.competitors or ["Generic Competitor 1", "Generic Competitor 2"],
                "gaps": ["Speed", "AI integration", "User experience"],
            },
            "platform_best_practices": {
                "platform": brief.target_platform,
                "optimal_length": brief.video_duration,
                "hooks": ["Question hooks", "Pain point hooks", "Curiosity hooks"],
            },
            "confidence_score": 0.85,
        }


class MockIdeationAgent:
    """Mock Ideation Agent for testing"""
    
    async def generate_concepts(
        self,
        brief: BriefData,
        research: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Simulate concept generation"""
        await asyncio.sleep(1.5)
        
        concepts = [
            {
                "concept_id": str(uuid.uuid4()),
                "title": f"Problem-Solution: {brief.key_message}",
                "hook": f"Tired of {research['audience_insights']['pain_points'][0]}?",
                "narrative_arc": "Problem → Agitation → Solution → Proof → CTA",
                "visual_style": "Dynamic cuts, bold text overlays",
                "estimated_impact": 8.5,
                "prm_score": 0.87,
            },
            {
                "concept_id": str(uuid.uuid4()),
                "title": f"Testimonial: Real Results with {brief.business_name}",
                "hook": "Here's how [Customer] achieved [Result]",
                "narrative_arc": "Before → Discovery → Transformation → Now",
                "visual_style": "Documentary style, authentic feel",
                "estimated_impact": 7.8,
                "prm_score": 0.82,
            },
            {
                "concept_id": str(uuid.uuid4()),
                "title": f"Demo: {brief.business_name} in Action",
                "hook": "Watch this...",
                "narrative_arc": "Hook → Demo → Benefits → CTA",
                "visual_style": "Screen recording + dynamic graphics",
                "estimated_impact": 7.2,
                "prm_score": 0.79,
            },
        ]
        
        return concepts


class MockScriptAgent:
    """Mock Script Agent for testing"""
    
    async def generate_script(
        self,
        brief: BriefData,
        concept: Dict[str, Any],
        research: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate script generation"""
        await asyncio.sleep(2)
        
        duration = brief.video_duration
        
        return {
            "script_id": str(uuid.uuid4()),
            "concept_id": concept["concept_id"],
            "duration_seconds": duration,
            "scenes": [
                {
                    "scene_number": 1,
                    "duration": 5,
                    "description": f"HOOK: {concept['hook']}",
                    "visual": "Dynamic intro with brand colors",
                    "audio": "Upbeat music fades in",
                    "text_overlay": concept["hook"],
                },
                {
                    "scene_number": 2,
                    "duration": duration - 15,
                    "description": f"BODY: {brief.key_message}",
                    "visual": "Product demonstration with features highlighted",
                    "audio": "Voiceover explaining key benefits",
                    "text_overlay": " | ".join(brief.unique_selling_points[:3]),
                },
                {
                    "scene_number": 3,
                    "duration": 5,
                    "description": "SOCIAL PROOF",
                    "visual": "Stats and testimonials",
                    "audio": "Music builds",
                    "text_overlay": "Join thousands of satisfied customers",
                },
                {
                    "scene_number": 4,
                    "duration": 5,
                    "description": "CTA",
                    "visual": f"{brief.business_name} logo with call to action",
                    "audio": "Music peaks then fades",
                    "text_overlay": f"Get started with {brief.business_name} today!",
                },
            ],
            "voiceover_text": f"""
{concept['hook']}

{brief.key_message}

With {brief.business_name}, you get:
{', '.join(brief.unique_selling_points)}

Join thousands who've already transformed their results.

Get started with {brief.business_name} today!
            """.strip(),
            "support_score": 0.88,
            "coherence_score": 0.91,
        }


class MockReviewAgent:
    """Mock Review Agent for testing"""
    
    async def review_script(
        self,
        script: Dict[str, Any],
        brief: BriefData,
    ) -> Dict[str, Any]:
        """Simulate quality review"""
        await asyncio.sleep(1)
        
        return {
            "approved": True,
            "overall_score": 8.7,
            "scores": {
                "message_clarity": 9.0,
                "brand_alignment": 8.5,
                "emotional_impact": 8.8,
                "cta_effectiveness": 8.5,
                "platform_fit": 9.0,
            },
            "feedback": [
                "Strong opening hook",
                "Clear value proposition",
                "Good pacing for platform",
            ],
            "suggestions": [],
            "prm_quality_gate": "PASSED",
        }


class MockRAGNAROKClient:
    """Mock RAGNAROK Video Pipeline client"""
    
    async def submit_job(
        self,
        script: Dict[str, Any],
        brief: BriefData,
        quality_tier: str,
    ) -> str:
        """Submit video generation job"""
        job_id = f"ragnk-{uuid.uuid4().hex[:8]}"
        logger.info(f"Mock RAGNAROK job submitted: {job_id}")
        return job_id
    
    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status (simulates progress)"""
        return {
            "job_id": job_id,
            "status": "complete",
            "progress": 100,
            "video_url": f"https://cdn.barriosa2i.com/videos/{job_id}.mp4",
            "preview_url": f"https://cdn.barriosa2i.com/previews/{job_id}.jpg",
            "duration_seconds": 30,
            "resolution": "1080p",
        }
    
    async def simulate_production(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Simulate production with progress updates"""
        stages = [
            (10, "Analyzing script..."),
            (25, "Generating visuals..."),
            (50, "Compositing scenes..."),
            (75, "Adding audio..."),
            (90, "Final rendering..."),
            (100, "Complete!"),
        ]
        
        for progress, status in stages:
            await asyncio.sleep(1)
            if progress_callback:
                progress_callback(progress)
        
        return await self.get_status(job_id)


# =============================================================================
# PIPELINE BRIDGE
# =============================================================================

class CreativeDirectorBridge:
    """
    Bridge between NEXUS and Creative Director 6-agent pipeline.
    
    Orchestrates the full workflow:
    1. Research (Trinity + RAG)
    2. Ideation (Concept generation with PRM)
    3. Scripting (Self-RAG script generation)
    4. Review (Quality gate)
    5. Production (RAGNAROK)
    
    Usage:
        bridge = CreativeDirectorBridge(config)
        await bridge.initialize()
        
        async for event in bridge.run_pipeline(session):
            handle_event(event)
    """
    
    def __init__(
        self,
        trinity_client=None,
        rag_client=None,
        ragnarok_client=None,
        use_mock: bool = True,
    ):
        self.trinity_client = trinity_client
        self.rag_client = rag_client
        self.ragnarok_client = ragnarok_client or MockRAGNAROKClient()
        self.use_mock = use_mock
        
        # Initialize agents (mock or real)
        if use_mock:
            self.research_agent = MockResearchAgent()
            self.ideation_agent = MockIdeationAgent()
            self.script_agent = MockScriptAgent()
            self.review_agent = MockReviewAgent()
        else:
            # Import real agents when available
            self._init_real_agents()
        
        self._initialized = False
    
    def _init_real_agents(self):
        """Initialize real agents from creative_director module"""
        try:
            # These would import from your actual creative director agents
            # from creative_director_agents import ResearchAgent, IdeationAgent, etc.
            logger.warning("Real agents not available, using mocks")
            self.research_agent = MockResearchAgent()
            self.ideation_agent = MockIdeationAgent()
            self.script_agent = MockScriptAgent()
            self.review_agent = MockReviewAgent()
        except ImportError as e:
            logger.warning(f"Failed to import real agents: {e}")
            self.research_agent = MockResearchAgent()
            self.ideation_agent = MockIdeationAgent()
            self.script_agent = MockScriptAgent()
            self.review_agent = MockReviewAgent()
    
    async def initialize(self):
        """Initialize the bridge"""
        self._initialized = True
        logger.info("Creative Director Bridge initialized")
    
    async def run_pipeline(
        self,
        session: CreativeDirectorSession,
        concept_selector: Optional[Callable[[List[Dict]], str]] = None,
        progress_callback: Optional[Callable[[PipelineEventData], None]] = None,
    ) -> AsyncGenerator[PipelineEventData, None]:
        """
        Run the full Creative Director pipeline.
        
        Args:
            session: Creative Director session with completed brief
            concept_selector: Callback to select concept (auto-selects best if None)
            progress_callback: Optional callback for progress updates
            
        Yields:
            PipelineEventData for each pipeline event
        """
        brief = session.brief
        
        if not brief.is_complete:
            yield PipelineEventData(
                event=PipelineEvent.PIPELINE_ERROR,
                session_id=session.session_id,
                data={"error": "Brief is not complete"},
            )
            return
        
        ACTIVE_PRODUCTIONS.inc()
        
        with tracer.start_as_current_span("creative_director.pipeline") as span:
            span.set_attribute("session_id", session.session_id)
            span.set_attribute("business", brief.business_name)
            
            try:
                # Emit start event
                yield PipelineEventData(
                    event=PipelineEvent.PIPELINE_STARTED,
                    session_id=session.session_id,
                    data={"brief": brief.to_dict()},
                )
                
                # Phase 1: Research
                session.workflow_phase = WorkflowPhase.RESEARCH
                yield PipelineEventData(
                    event=PipelineEvent.RESEARCH_STARTED,
                    session_id=session.session_id,
                )
                
                research_start = datetime.utcnow()
                research = await self.research_agent.research(brief)
                session.market_research = research
                
                PIPELINE_DURATION.labels(phase="research").observe(
                    (datetime.utcnow() - research_start).total_seconds()
                )
                
                yield PipelineEventData(
                    event=PipelineEvent.RESEARCH_COMPLETE,
                    session_id=session.session_id,
                    data={
                        "confidence_score": research.get("confidence_score", 0),
                        "audience_insights": research.get("audience_insights", {}),
                    },
                )
                
                # Phase 2: Ideation
                session.workflow_phase = WorkflowPhase.IDEATION
                yield PipelineEventData(
                    event=PipelineEvent.IDEATION_STARTED,
                    session_id=session.session_id,
                )
                
                ideation_start = datetime.utcnow()
                concepts = await self.ideation_agent.generate_concepts(brief, research)
                session.concepts = concepts
                
                PIPELINE_DURATION.labels(phase="ideation").observe(
                    (datetime.utcnow() - ideation_start).total_seconds()
                )
                
                yield PipelineEventData(
                    event=PipelineEvent.CONCEPTS_GENERATED,
                    session_id=session.session_id,
                    data={
                        "num_concepts": len(concepts),
                        "concepts": concepts,
                    },
                )
                
                # Select concept
                if concept_selector:
                    selected_id = concept_selector(concepts)
                else:
                    # Auto-select highest impact
                    best = max(concepts, key=lambda c: c.get("estimated_impact", 0))
                    selected_id = best["concept_id"]
                
                session.selected_concept_id = selected_id
                selected_concept = next(c for c in concepts if c["concept_id"] == selected_id)
                
                yield PipelineEventData(
                    event=PipelineEvent.CONCEPT_SELECTED,
                    session_id=session.session_id,
                    data={"concept": selected_concept},
                )
                
                # Phase 3: Scripting
                session.workflow_phase = WorkflowPhase.SCRIPTING
                yield PipelineEventData(
                    event=PipelineEvent.SCRIPTING_STARTED,
                    session_id=session.session_id,
                )
                
                scripting_start = datetime.utcnow()
                script = await self.script_agent.generate_script(
                    brief, selected_concept, research
                )
                session.script = script
                
                PIPELINE_DURATION.labels(phase="scripting").observe(
                    (datetime.utcnow() - scripting_start).total_seconds()
                )
                
                yield PipelineEventData(
                    event=PipelineEvent.SCRIPT_COMPLETE,
                    session_id=session.session_id,
                    data={
                        "script_id": script.get("script_id"),
                        "support_score": script.get("support_score"),
                        "scenes": len(script.get("scenes", [])),
                    },
                )
                
                # Phase 4: Review
                session.workflow_phase = WorkflowPhase.REVIEW
                yield PipelineEventData(
                    event=PipelineEvent.REVIEW_STARTED,
                    session_id=session.session_id,
                )
                
                review_start = datetime.utcnow()
                review = await self.review_agent.review_script(script, brief)
                
                PIPELINE_DURATION.labels(phase="review").observe(
                    (datetime.utcnow() - review_start).total_seconds()
                )
                
                if review.get("approved"):
                    yield PipelineEventData(
                        event=PipelineEvent.REVIEW_PASSED,
                        session_id=session.session_id,
                        data={
                            "overall_score": review.get("overall_score"),
                            "scores": review.get("scores"),
                        },
                    )
                else:
                    yield PipelineEventData(
                        event=PipelineEvent.REVIEW_FAILED,
                        session_id=session.session_id,
                        data={
                            "feedback": review.get("feedback"),
                            "suggestions": review.get("suggestions"),
                        },
                    )
                    session.workflow_phase = WorkflowPhase.ERROR
                    return
                
                # Phase 5: Production
                session.workflow_phase = WorkflowPhase.PRODUCTION
                yield PipelineEventData(
                    event=PipelineEvent.PRODUCTION_STARTED,
                    session_id=session.session_id,
                )
                
                production_start = datetime.utcnow()
                job_id = await self.ragnarok_client.submit_job(
                    script, brief, brief.budget_tier
                )
                
                session.production_job = ProductionJob(
                    job_id=job_id,
                    session_id=session.session_id,
                    status="processing",
                )
                
                # Simulate production progress
                def on_progress(progress: int):
                    if progress_callback:
                        progress_callback(PipelineEventData(
                            event=PipelineEvent.PRODUCTION_PROGRESS,
                            session_id=session.session_id,
                            data={"progress": progress, "job_id": job_id},
                        ))
                
                result = await self.ragnarok_client.simulate_production(
                    job_id, on_progress
                )
                
                session.production_job.status = "complete"
                session.production_job.progress = 100
                session.production_job.video_url = result.get("video_url")
                session.production_job.preview_url = result.get("preview_url")
                session.video_url = result.get("video_url")
                
                PIPELINE_DURATION.labels(phase="production").observe(
                    (datetime.utcnow() - production_start).total_seconds()
                )
                
                yield PipelineEventData(
                    event=PipelineEvent.PRODUCTION_COMPLETE,
                    session_id=session.session_id,
                    data={
                        "video_url": result.get("video_url"),
                        "preview_url": result.get("preview_url"),
                        "job_id": job_id,
                    },
                )
                
                # Complete
                session.workflow_phase = WorkflowPhase.COMPLETE
                
                PIPELINE_RUNS.labels(status="success", phase="complete").inc()
                
                yield PipelineEventData(
                    event=PipelineEvent.PIPELINE_COMPLETE,
                    session_id=session.session_id,
                    data={
                        "video_url": session.video_url,
                        "brief": brief.to_dict(),
                        "concept": selected_concept,
                        "script_id": script.get("script_id"),
                        "quality_score": review.get("overall_score"),
                    },
                )
                
            except Exception as e:
                logger.exception(f"Pipeline error: {session.session_id}")
                session.workflow_phase = WorkflowPhase.ERROR
                
                PIPELINE_RUNS.labels(status="error", phase=session.workflow_phase.value).inc()
                
                yield PipelineEventData(
                    event=PipelineEvent.PIPELINE_ERROR,
                    session_id=session.session_id,
                    data={"error": str(e)},
                )
            
            finally:
                ACTIVE_PRODUCTIONS.dec()
    
    async def get_pipeline_status(
        self,
        session: CreativeDirectorSession,
    ) -> Dict[str, Any]:
        """Get current pipeline status for a session"""
        return {
            "session_id": session.session_id,
            "workflow_phase": session.workflow_phase.value,
            "brief_complete": session.brief.is_complete,
            "has_research": session.market_research is not None,
            "num_concepts": len(session.concepts),
            "has_script": session.script is not None,
            "production_job": {
                "job_id": session.production_job.job_id if session.production_job else None,
                "status": session.production_job.status if session.production_job else None,
                "progress": session.production_job.progress if session.production_job else 0,
            } if session.production_job else None,
            "video_url": session.video_url,
        }


# =============================================================================
# QUICK TEST
# =============================================================================

async def _test_pipeline():
    """Test the pipeline bridge"""
    from app.creative_director.session_manager import CreativeDirectorSession, BriefData
    
    print("\n" + "="*60)
    print("🎬 Creative Director Pipeline Test")
    print("="*60)
    
    # Create session with complete brief
    session = CreativeDirectorSession(
        session_id="test-session-123",
        user_id="test-user",
    )
    
    session.brief = BriefData(
        session_id=session.session_id,
        business_name="FitTech Pro",
        industry="fitness technology",
        target_audience="Health-conscious millennials aged 25-40",
        target_platform="tiktok",
        video_duration=30,
        key_message="Transform your fitness with AI-powered workouts",
        unique_selling_points=["AI personalization", "24/7 coaching", "Progress tracking"],
        brand_tone="bold",
        budget_tier="premium",
        competitors=["Peloton", "Nike Training Club"],
        is_complete=True,
    )
    
    # Initialize bridge
    bridge = CreativeDirectorBridge(use_mock=True)
    await bridge.initialize()
    
    # Run pipeline
    print("\n🚀 Running pipeline...\n")
    
    async for event in bridge.run_pipeline(session):
        icon = {
            PipelineEvent.PIPELINE_STARTED: "🎬",
            PipelineEvent.RESEARCH_STARTED: "🔍",
            PipelineEvent.RESEARCH_COMPLETE: "✅",
            PipelineEvent.IDEATION_STARTED: "💡",
            PipelineEvent.CONCEPTS_GENERATED: "✨",
            PipelineEvent.CONCEPT_SELECTED: "🎯",
            PipelineEvent.SCRIPTING_STARTED: "✍️",
            PipelineEvent.SCRIPT_COMPLETE: "📝",
            PipelineEvent.REVIEW_STARTED: "🔎",
            PipelineEvent.REVIEW_PASSED: "✅",
            PipelineEvent.PRODUCTION_STARTED: "🎥",
            PipelineEvent.PRODUCTION_PROGRESS: "⏳",
            PipelineEvent.PRODUCTION_COMPLETE: "🎉",
            PipelineEvent.PIPELINE_COMPLETE: "🏆",
            PipelineEvent.PIPELINE_ERROR: "❌",
        }.get(event.event, "📌")
        
        print(f"{icon} {event.event.value}")
        
        if event.event == PipelineEvent.PIPELINE_COMPLETE:
            print(f"\n   Video URL: {event.data.get('video_url')}")
            print(f"   Quality Score: {event.data.get('quality_score')}")
    
    print("\n" + "="*60)
    print("✅ Pipeline test complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(_test_pipeline())
