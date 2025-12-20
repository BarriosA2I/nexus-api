"""
================================================================================
VOICEOVER MASTER RAG AGENT - LEGENDARY EDITION
================================================================================
Neural RAG Brain Cognitive Architecture for Commercial Audio Generation

Components:
- Test-Time Reasoning with PRM Beam Search (voice selection optimization)
- Dual-Process System 1/2 Routing (cached fast vs deliberate voice matching)
- Self-Reflective RAG with Reflection Tokens [RET][REL][SUP][USE]
- CRAG Corrective Actions (ElevenLabs → Azure → gTTS fallback chain)
- 4-Tier Hierarchical Memory (voice preferences, patterns, client history)
- Circuit Breakers with Provider-Level Isolation
- Google Drive Asset Integration
- Cost-Optimized Multi-Provider Routing

Pipeline:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VOICEOVER MASTER ORCHESTRATOR                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ ComplexityRouter│→│  VoiceRAG      │→│ ScriptOptimizer │                │
│  │ (System 1/2)   │  │(Voice Matching)│  │(PRM Beam Search)│                │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│           │                  │                   │                           │
│           └──────────────────┼───────────────────┘                           │
│                              ↓                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ ElevenLabs TTS │→│ QualityChecker │→│ Google Drive    │                │
│  │ (+ Azure/gTTS) │  │ (Self-Reflect) │  │ Asset Storage  │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘

Performance Targets:
- Voice Match Accuracy: 0.94+ (PRM-verified)
- Audio Quality Score: 0.96+ (MOS-based)
- Fallback Success Rate: 99.5%
- p95 Latency: <8s per scene
- Cost/Scene: <$0.15 (tiered routing)

Author: Barrios A2I | Version: 1.0.0 | December 2025
================================================================================
"""

import asyncio
import time
import uuid
import logging
import hashlib
import json
import base64
import math
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from io import BytesIO

# Observability
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Pydantic for structured data
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceover_master")

# Tracer
tracer = trace.get_tracer("voiceover_master_agent")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class VoiceProvider(Enum):
    """Supported TTS providers with quality/cost tiers"""
    ELEVENLABS = "elevenlabs"      # Tier 1: Premium ($0.30/1K chars)
    AZURE = "azure"                 # Tier 2: High quality ($0.016/1K chars)
    GTTS = "gtts"                   # Tier 3: Free fallback
    OPENAI = "openai"               # Tier 1.5: High quality ($0.015/1K chars)


class VoiceEmotion(Enum):
    """Emotional tone for voice synthesis"""
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    ENERGETIC = "energetic"
    WARM = "warm"
    URGENT = "urgent"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"


class SceneType(Enum):
    """Commercial scene types with voice requirements"""
    HOOK = "hook"           # 1-3s, punchy, attention-grabbing
    PROBLEM = "problem"     # Pain point, empathetic
    SOLUTION = "solution"   # Product/service intro, confident
    BENEFITS = "benefits"   # Value props, warm
    PROOF = "proof"         # Social proof, credible
    CTA = "cta"             # Call to action, urgent


class CRAGAction(Enum):
    """CRAG corrective actions for voice generation"""
    GENERATE = "generate"       # High confidence - proceed
    OPTIMIZE = "optimize"       # Medium confidence - refine script
    FALLBACK = "fallback"       # Low confidence - switch provider


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VoiceProfile:
    """Voice characteristics for matching"""
    id: str
    name: str
    provider: VoiceProvider
    gender: str  # male, female, neutral
    age_range: str  # young, adult, mature
    accent: str  # american, british, australian, etc.
    style: str  # conversational, narrative, commercial
    emotions: List[VoiceEmotion]
    sample_rate: int = 44100
    cost_per_1k_chars: float = 0.30
    quality_score: float = 0.95
    latency_ms: int = 1500
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class SceneVoiceover:
    """Voiceover for a single commercial scene"""
    scene_id: int
    scene_type: SceneType
    text: str
    duration_target_seconds: float
    emotion: VoiceEmotion
    pacing: str = "normal"  # slow, normal, fast
    emphasis_words: List[str] = field(default_factory=list)
    pause_after: float = 0.3


class VoiceoverRequest(BaseModel):
    """Request for voiceover generation"""
    client_id: str
    commercial_id: str
    scenes: List[Dict[str, Any]]  # Scene configs from intake
    target_voice_profile: Optional[Dict[str, Any]] = None
    brand_voice_guidelines: Optional[str] = None
    total_duration_seconds: float = 30.0
    quality_tier: str = "premium"  # premium, standard, budget


class VoiceoverResult(BaseModel):
    """Result from voiceover generation"""
    client_id: str
    commercial_id: str
    scenes: List[Dict[str, Any]]
    total_duration_seconds: float
    total_cost_usd: float
    provider_used: str
    voice_profile: Dict[str, Any]
    drive_assets: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    processing_time_ms: float


@dataclass
class MemoryTrace:
    """Memory trace for hierarchical memory system"""
    id: str
    content: str
    embedding: List[float]
    timestamp: float
    importance_score: float
    access_count: int
    metadata: Dict[str, Any]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VoiceoverConfig:
    """Configuration for Voiceover Master Agent"""
    
    # Provider settings
    elevenlabs_api_key: str = ""
    elevenlabs_default_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"
    
    openai_api_key: str = ""
    
    # Google Drive settings
    gdrive_sa_json_b64: str = ""
    gdrive_root_folder_id: str = ""
    gdrive_clients_folder: str = "Nexus/Clients"
    
    # Cognitive settings
    working_memory_capacity: int = 7  # Miller's 7±2
    episodic_decay_tau: float = 86400.0  # 24-hour half-life
    prm_threshold: float = 0.3  # Step acceptance threshold
    crag_high_threshold: float = 0.7
    crag_low_threshold: float = 0.4
    
    # Quality settings
    min_quality_score: float = 0.90
    max_retries: int = 3
    
    # Circuit breaker settings
    failure_threshold: int = 3
    recovery_timeout: int = 30
    
    # Cost optimization
    cost_tier_routing: bool = True
    max_cost_per_scene: float = 0.50
    
    @classmethod
    def from_env(cls) -> "VoiceoverConfig":
        """Load config from environment variables"""
        return cls(
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_default_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "eastus"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gdrive_sa_json_b64=os.getenv("GDRIVE_SA_JSON_B64", ""),
            gdrive_root_folder_id=os.getenv("GDRIVE_ROOT_FOLDER_ID", ""),
            gdrive_clients_folder=os.getenv("GDRIVE_CLIENTS_FOLDER_NAME", "Nexus/Clients"),
        )


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Provider-level circuit breaker"""
    
    name: str
    failure_threshold: int = 3
    recovery_timeout: int = 30
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    
    def can_execute(self) -> bool:
        """Check if circuit allows execution"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        
        # HALF_OPEN - allow one test request
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Require 2 successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit {self.name} CLOSED after recovery")
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} OPENED after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "time_since_failure": time.time() - self.last_failure_time if self.last_failure_time else None
        }


# =============================================================================
# HIERARCHICAL MEMORY SYSTEM
# =============================================================================

class VoiceMemorySystem:
    """
    4-Tier Hierarchical Memory for Voice Preferences
    
    L0: Working Memory - Current session voices (7±2 capacity)
    L1: Episodic Memory - Voice generation history (temporal decay)
    L2: Semantic Memory - Voice-brand mappings (knowledge graph)
    L3: Procedural Memory - Successful voice patterns (skill library)
    """
    
    def __init__(
        self,
        config: VoiceoverConfig,
        embedding_func: Callable[[str], List[float]] = None
    ):
        self.config = config
        self.embed = embedding_func or self._simple_embed
        
        # L0: Working Memory (in-process LRU)
        self.working_memory: deque[MemoryTrace] = deque(
            maxlen=config.working_memory_capacity
        )
        
        # L1: Episodic Memory (in-memory for now, can be PostgreSQL)
        self.episodic_memory: Dict[str, MemoryTrace] = {}
        
        # L2: Semantic Memory (voice-brand mappings)
        self.semantic_memory: Dict[str, Dict[str, Any]] = {}
        
        # L3: Procedural Memory (successful patterns)
        self.procedural_memory: List[Dict[str, Any]] = []
    
    def _simple_embed(self, text: str) -> List[float]:
        """Simple hash-based embedding for demo (replace with real embeddings)"""
        # In production, use sentence-transformers or OpenAI embeddings
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)
    
    def compute_importance(
        self,
        similarity: float,
        age_seconds: float,
        access_count: int = 1
    ) -> float:
        """
        Ebbinghaus decay formula with access boost
        
        importance = similarity × e^(-age/τ) × (1 + log(access_count))
        """
        tau = self.config.episodic_decay_tau
        decay = math.exp(-age_seconds / tau)
        access_boost = 1 + math.log(max(access_count, 1))
        return similarity * decay * access_boost
    
    # L0: Working Memory Operations
    def working_add(self, content: str, metadata: Dict = None) -> str:
        """Add to working memory"""
        trace_id = str(uuid.uuid4())
        trace = MemoryTrace(
            id=trace_id,
            content=content,
            embedding=self.embed(content),
            timestamp=time.time(),
            importance_score=1.0,
            access_count=1,
            metadata=metadata or {}
        )
        self.working_memory.append(trace)
        return trace_id
    
    def working_query(self, query: str, k: int = 3) -> List[MemoryTrace]:
        """Query working memory by similarity"""
        if not self.working_memory:
            return []
        
        query_emb = self.embed(query)
        scored = []
        for trace in self.working_memory:
            sim = self._cosine_similarity(query_emb, trace.embedding)
            scored.append((sim, trace))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in scored[:k]]
    
    # L1: Episodic Memory Operations
    def episodic_store(
        self,
        client_id: str,
        voice_id: str,
        result: Dict[str, Any]
    ) -> str:
        """Store voice generation episode"""
        trace_id = f"ep_{client_id}_{voice_id}_{int(time.time())}"
        content = f"{client_id}|{voice_id}|{json.dumps(result)}"
        
        trace = MemoryTrace(
            id=trace_id,
            content=content,
            embedding=self.embed(content),
            timestamp=time.time(),
            importance_score=result.get("quality_score", 0.8),
            access_count=1,
            metadata={
                "client_id": client_id,
                "voice_id": voice_id,
                "quality_score": result.get("quality_score", 0.8),
                "provider": result.get("provider", "unknown")
            }
        )
        self.episodic_memory[trace_id] = trace
        return trace_id
    
    def episodic_query(
        self,
        client_id: str,
        k: int = 5
    ) -> List[MemoryTrace]:
        """Query episodic memory for client history"""
        current_time = time.time()
        scored = []
        
        for trace in self.episodic_memory.values():
            if trace.metadata.get("client_id") == client_id:
                age = current_time - trace.timestamp
                importance = self.compute_importance(
                    trace.importance_score,
                    age,
                    trace.access_count
                )
                scored.append((importance, trace))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in scored[:k]]
    
    # L2: Semantic Memory Operations
    def semantic_store(
        self,
        brand_id: str,
        voice_profile: VoiceProfile,
        match_score: float
    ):
        """Store brand-voice semantic mapping"""
        key = f"{brand_id}_{voice_profile.id}"
        self.semantic_memory[key] = {
            "brand_id": brand_id,
            "voice_id": voice_profile.id,
            "voice_profile": {
                "name": voice_profile.name,
                "provider": voice_profile.provider.value,
                "gender": voice_profile.gender,
                "style": voice_profile.style
            },
            "match_score": match_score,
            "timestamp": time.time()
        }
    
    def semantic_query(self, brand_id: str) -> List[Dict[str, Any]]:
        """Query semantic memory for brand voice mappings"""
        results = []
        for key, value in self.semantic_memory.items():
            if value["brand_id"] == brand_id:
                results.append(value)
        return sorted(results, key=lambda x: x["match_score"], reverse=True)
    
    # L3: Procedural Memory Operations
    def procedural_store(
        self,
        pattern_name: str,
        scene_type: SceneType,
        voice_params: Dict[str, Any],
        success_rate: float
    ):
        """Store successful voice pattern"""
        pattern = {
            "id": str(uuid.uuid4()),
            "name": pattern_name,
            "scene_type": scene_type.value,
            "voice_params": voice_params,
            "success_rate": success_rate,
            "usage_count": 1,
            "created_at": time.time()
        }
        self.procedural_memory.append(pattern)
    
    def procedural_query(
        self,
        scene_type: SceneType,
        min_success_rate: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Query procedural memory for proven patterns"""
        results = []
        for pattern in self.procedural_memory:
            if (pattern["scene_type"] == scene_type.value and
                pattern["success_rate"] >= min_success_rate):
                results.append(pattern)
        return sorted(results, key=lambda x: x["success_rate"], reverse=True)
    
    # Unified Retrieval
    async def retrieve_voice_context(
        self,
        client_id: str,
        scene_type: SceneType,
        brand_description: str = ""
    ) -> Dict[str, Any]:
        """
        Unified retrieval across all memory tiers
        
        Returns context for voice selection and generation
        """
        # L0: Check working memory for current session
        working_results = self.working_query(f"{client_id} {scene_type.value}", k=2)
        
        # L1: Get recent episodes for this client
        episodic_results = self.episodic_query(client_id, k=3)
        
        # L2: Get semantic brand-voice mappings
        semantic_results = self.semantic_query(client_id)
        
        # L3: Get successful patterns for this scene type
        procedural_results = self.procedural_query(scene_type, min_success_rate=0.85)
        
        return {
            "working": [{"content": t.content, "score": t.importance_score} 
                       for t in working_results],
            "episodic": [{"id": t.id, "metadata": t.metadata, "score": t.importance_score}
                        for t in episodic_results],
            "semantic": semantic_results[:3],
            "procedural": procedural_results[:3]
        }


# =============================================================================
# PROCESS REWARD MODEL (PRM) FOR VOICE OPTIMIZATION
# =============================================================================

class VoicePRM:
    """
    Process Reward Model for Voice Selection and Script Optimization
    
    Evaluates step-by-step decisions in voice generation pipeline:
    1. Voice selection quality
    2. Script-voice matching
    3. Pacing and emotion alignment
    4. Audio quality prediction
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.step_cache: Dict[str, float] = {}
    
    async def evaluate_voice_selection(
        self,
        scene: SceneVoiceover,
        voice_profile: VoiceProfile,
        brand_guidelines: str = ""
    ) -> float:
        """
        Evaluate voice selection step
        
        Returns: 0.0-1.0 score for step correctness
        """
        # Check cache
        cache_key = f"voice_sel_{scene.scene_type.value}_{voice_profile.id}"
        if cache_key in self.step_cache:
            return self.step_cache[cache_key]
        
        # Rule-based scoring (can be replaced with LLM evaluation)
        score = 0.5  # Base score
        
        # Emotion matching
        if scene.emotion in voice_profile.emotions:
            score += 0.2
        
        # Style matching for scene type
        style_matches = {
            SceneType.HOOK: ["commercial", "energetic", "narrative"],
            SceneType.CTA: ["commercial", "conversational"],
            SceneType.PROBLEM: ["empathetic", "conversational"],
            SceneType.SOLUTION: ["confident", "narrative"],
            SceneType.BENEFITS: ["warm", "conversational"],
            SceneType.PROOF: ["credible", "narrative"]
        }
        
        if scene.scene_type in style_matches:
            if voice_profile.style in style_matches[scene.scene_type]:
                score += 0.15
        
        # Quality tier matching
        if voice_profile.quality_score >= 0.95:
            score += 0.1
        
        # Cost efficiency
        if voice_profile.cost_per_1k_chars <= 0.30:
            score += 0.05
        
        score = min(score, 1.0)
        self.step_cache[cache_key] = score
        return score
    
    async def evaluate_script_optimization(
        self,
        original_text: str,
        optimized_text: str,
        scene_type: SceneType
    ) -> float:
        """
        Evaluate script optimization step
        
        Checks:
        - Length appropriateness for scene type
        - Emotional resonance
        - Clarity and pacing markers
        """
        score = 0.5
        
        # Length targets by scene type
        length_targets = {
            SceneType.HOOK: (10, 30),      # 10-30 words
            SceneType.PROBLEM: (20, 50),   # 20-50 words
            SceneType.SOLUTION: (30, 60),  # 30-60 words
            SceneType.BENEFITS: (30, 60),  # 30-60 words
            SceneType.PROOF: (20, 50),     # 20-50 words
            SceneType.CTA: (10, 25)        # 10-25 words
        }
        
        word_count = len(optimized_text.split())
        target_min, target_max = length_targets.get(scene_type, (20, 50))
        
        if target_min <= word_count <= target_max:
            score += 0.25
        elif word_count < target_min:
            score += 0.1 * (word_count / target_min)
        else:
            score += 0.1 * (target_max / word_count)
        
        # Check for pacing markers (commas, ellipses)
        if "," in optimized_text or "..." in optimized_text:
            score += 0.1
        
        # Improvement over original
        if len(optimized_text) != len(original_text):
            score += 0.05  # Changed something
        
        # No awkward phrases
        awkward_phrases = ["um", "uh", "like", "you know"]
        if not any(phrase in optimized_text.lower() for phrase in awkward_phrases):
            score += 0.1
        
        return min(score, 1.0)
    
    async def evaluate_audio_quality_prediction(
        self,
        text: str,
        voice_profile: VoiceProfile,
        duration_target: float
    ) -> float:
        """
        Predict audio quality before generation
        
        Based on:
        - Text complexity vs voice capability
        - Duration feasibility
        - Provider reliability
        """
        score = 0.6  # Base prediction
        
        # Words per minute feasibility
        words = len(text.split())
        minutes = duration_target / 60.0
        wpm = words / minutes if minutes > 0 else 0
        
        # Normal speaking rate is 120-150 WPM
        if 100 <= wpm <= 180:
            score += 0.2
        elif 80 <= wpm <= 200:
            score += 0.1
        else:
            score -= 0.1  # Unrealistic pacing
        
        # Provider quality history
        provider_quality = {
            VoiceProvider.ELEVENLABS: 0.15,
            VoiceProvider.AZURE: 0.12,
            VoiceProvider.OPENAI: 0.12,
            VoiceProvider.GTTS: 0.05
        }
        score += provider_quality.get(voice_profile.provider, 0.05)
        
        # Voice quality score contribution
        score += voice_profile.quality_score * 0.1
        
        return min(max(score, 0.0), 1.0)


# =============================================================================
# DUAL-PROCESS ROUTER (SYSTEM 1 / SYSTEM 2)
# =============================================================================

class VoiceDualProcessRouter:
    """
    Dual-Process Cognitive Router for Voice Generation
    
    System 1 (Fast): Cached voices, known clients, simple scenes
    System 2 (Slow): New clients, complex matching, optimization needed
    """
    
    def __init__(
        self,
        memory: VoiceMemorySystem,
        prm: VoicePRM
    ):
        self.memory = memory
        self.prm = prm
        self.voice_cache: Dict[str, VoiceProfile] = {}
    
    async def classify_complexity(
        self,
        request: VoiceoverRequest
    ) -> Tuple[int, str]:
        """
        Classify request complexity (1-10)
        
        Returns: (complexity_score, reasoning)
        """
        score = 1
        reasons = []
        
        # Check client history
        history = self.memory.episodic_query(request.client_id, k=1)
        if not history:
            score += 3
            reasons.append("new_client")
        
        # Check scene count
        if len(request.scenes) > 4:
            score += 2
            reasons.append("many_scenes")
        
        # Check for custom voice requirements
        if request.target_voice_profile:
            score += 2
            reasons.append("custom_voice_required")
        
        # Check brand guidelines
        if request.brand_voice_guidelines:
            score += 2
            reasons.append("brand_guidelines_present")
        
        # Duration complexity
        if request.total_duration_seconds > 45:
            score += 1
            reasons.append("long_duration")
        
        return min(score, 10), "+".join(reasons) or "simple"
    
    async def route(
        self,
        request: VoiceoverRequest
    ) -> Dict[str, Any]:
        """
        Route to System 1 or System 2
        
        Returns routing decision with context
        """
        complexity, reason = await self.classify_complexity(request)
        
        if complexity <= 3:
            # SYSTEM 1: Fast path
            return {
                "system": 1,
                "complexity": complexity,
                "reason": reason,
                "strategy": "cached_voice_lookup",
                "expected_latency_ms": 200
            }
        
        elif complexity <= 6:
            # HYBRID: Try System 1, escalate if needed
            return {
                "system": "hybrid",
                "complexity": complexity,
                "reason": reason,
                "strategy": "try_cache_then_optimize",
                "expected_latency_ms": 2000
            }
        
        else:
            # SYSTEM 2: Full deliberate reasoning
            return {
                "system": 2,
                "complexity": complexity,
                "reason": reason,
                "strategy": "full_voice_matching_pipeline",
                "expected_latency_ms": 5000
            }


# =============================================================================
# SELF-REFLECTIVE RAG FOR VOICE QUALITY
# =============================================================================

class VoiceSelfReflectiveRAG:
    """
    Self-Reflective RAG with Reflection Tokens for Voice Generation
    
    Tokens:
    [RET] - Should retrieve voice examples?
    [REL] - Is retrieved voice relevant?
    [SUP] - Does generated audio support requirements?
    [USE] - Is the result useful for the commercial?
    """
    
    def __init__(
        self,
        memory: VoiceMemorySystem,
        prm: VoicePRM
    ):
        self.memory = memory
        self.prm = prm
        self.reflection_log: List[Dict[str, Any]] = []
    
    async def should_retrieve(
        self,
        scene: SceneVoiceover,
        current_context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        [RET] Reflection: Should we retrieve voice examples?
        """
        # Always retrieve for new scene types or empty context
        if not current_context.get("procedural"):
            return True, "no_procedural_patterns"
        
        # Retrieve if scene type is complex
        if scene.scene_type in [SceneType.HOOK, SceneType.CTA]:
            return True, "high_impact_scene"
        
        # Don't retrieve if we have high-confidence cached pattern
        best_pattern = current_context["procedural"][0] if current_context["procedural"] else None
        if best_pattern and best_pattern["success_rate"] >= 0.95:
            return False, "high_confidence_pattern_cached"
        
        return True, "default_retrieve"
    
    async def is_relevant(
        self,
        scene: SceneVoiceover,
        voice_profile: VoiceProfile,
        match_score: float
    ) -> Tuple[bool, float]:
        """
        [REL] Reflection: Is this voice relevant for the scene?
        """
        # PRM-based evaluation
        prm_score = await self.prm.evaluate_voice_selection(
            scene, voice_profile
        )
        
        # Combined relevance
        relevance = (match_score * 0.6) + (prm_score * 0.4)
        
        is_relevant = relevance >= 0.7
        
        self.reflection_log.append({
            "token": "REL",
            "scene_id": scene.scene_id,
            "voice_id": voice_profile.id,
            "match_score": match_score,
            "prm_score": prm_score,
            "relevance": relevance,
            "is_relevant": is_relevant,
            "timestamp": time.time()
        })
        
        return is_relevant, relevance
    
    async def is_supported(
        self,
        audio_bytes: bytes,
        scene: SceneVoiceover,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        [SUP] Reflection: Does the audio meet requirements?
        """
        support_score = 0.0
        
        # Check duration (should be within 15% of target)
        actual_duration = quality_metrics.get("duration_seconds", 0)
        target_duration = scene.duration_target_seconds
        duration_ratio = actual_duration / target_duration if target_duration > 0 else 0
        
        if 0.85 <= duration_ratio <= 1.15:
            support_score += 0.4
        elif 0.7 <= duration_ratio <= 1.3:
            support_score += 0.2
        
        # Check audio quality
        quality = quality_metrics.get("quality_score", 0.8)
        support_score += quality * 0.4
        
        # Check for silence/artifacts
        if quality_metrics.get("has_silence_issues", False):
            support_score -= 0.2
        
        # File size sanity check
        if len(audio_bytes) > 1000:  # More than 1KB
            support_score += 0.2
        
        is_supported = support_score >= 0.7
        
        self.reflection_log.append({
            "token": "SUP",
            "scene_id": scene.scene_id,
            "duration_ratio": duration_ratio,
            "quality": quality,
            "support_score": support_score,
            "is_supported": is_supported,
            "timestamp": time.time()
        })
        
        return is_supported, support_score
    
    async def is_useful(
        self,
        scene: SceneVoiceover,
        audio_result: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """
        [USE] Reflection: Is this audio useful for the commercial?
        
        Returns: (is_useful, usefulness_score 1-5)
        """
        usefulness = 3  # Base score
        
        # Check if audio was generated
        if audio_result.get("audio_bytes"):
            usefulness += 1
        
        # Check quality score
        if audio_result.get("quality_score", 0) >= 0.9:
            usefulness += 1
        
        # Check if Drive upload succeeded
        if audio_result.get("drive_file_id"):
            usefulness += 0  # Already counted in audio_bytes
        
        # Penalize retries
        if audio_result.get("retry_count", 0) > 1:
            usefulness -= 1
        
        is_useful = usefulness >= 4
        
        self.reflection_log.append({
            "token": "USE",
            "scene_id": scene.scene_id,
            "usefulness": usefulness,
            "is_useful": is_useful,
            "timestamp": time.time()
        })
        
        return is_useful, min(max(usefulness, 1), 5)
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of all reflections"""
        by_token = {"RET": [], "REL": [], "SUP": [], "USE": []}
        
        for log in self.reflection_log:
            token = log.get("token")
            if token in by_token:
                by_token[token].append(log)
        
        return {
            "total_reflections": len(self.reflection_log),
            "by_token": {
                k: {
                    "count": len(v),
                    "avg_score": sum(l.get("relevance", l.get("support_score", l.get("usefulness", 0))) 
                                    for l in v) / len(v) if v else 0
                }
                for k, v in by_token.items()
            }
        }


# =============================================================================
# TTS PROVIDER IMPLEMENTATIONS
# =============================================================================

class TTSProvider(ABC):
    """Abstract base class for TTS providers"""
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesize speech from text
        
        Returns: (audio_bytes, metadata)
        """
        pass
    
    @abstractmethod
    def get_voices(self) -> List[VoiceProfile]:
        """Get available voices"""
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str) -> float:
        """Estimate cost for synthesis"""
        pass


class ElevenLabsProvider(TTSProvider):
    """
    ElevenLabs TTS Provider (Tier 1: Premium)
    
    Features:
    - Highest quality voice synthesis
    - Emotion control
    - Voice cloning support
    - Multi-language
    """
    
    def __init__(self, config: VoiceoverConfig):
        self.api_key = config.elevenlabs_api_key
        self.default_voice_id = config.elevenlabs_default_voice_id
        self.model_id = config.elevenlabs_model_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.cost_per_1k_chars = 0.30
        
        # Pre-defined voice profiles
        self.voices = self._load_voices()
    
    def _load_voices(self) -> List[VoiceProfile]:
        """Load available ElevenLabs voices"""
        return [
            VoiceProfile(
                id="21m00Tcm4TlvDq8ikWAM",
                name="Rachel",
                provider=VoiceProvider.ELEVENLABS,
                gender="female",
                age_range="adult",
                accent="american",
                style="conversational",
                emotions=[VoiceEmotion.WARM, VoiceEmotion.PROFESSIONAL],
                cost_per_1k_chars=0.30,
                quality_score=0.98
            ),
            VoiceProfile(
                id="29vD33N1CtxCmqQRPOHJ",
                name="Drew",
                provider=VoiceProvider.ELEVENLABS,
                gender="male",
                age_range="adult",
                accent="american",
                style="commercial",
                emotions=[VoiceEmotion.CONFIDENT, VoiceEmotion.ENERGETIC],
                cost_per_1k_chars=0.30,
                quality_score=0.97
            ),
            VoiceProfile(
                id="2EiwWnXFnvU5JabPnv8n",
                name="Clyde",
                provider=VoiceProvider.ELEVENLABS,
                gender="male",
                age_range="mature",
                accent="american",
                style="narrative",
                emotions=[VoiceEmotion.PROFESSIONAL, VoiceEmotion.WARM],
                cost_per_1k_chars=0.30,
                quality_score=0.96
            ),
            VoiceProfile(
                id="EXAVITQu4vr4xnSDxMaL",
                name="Sarah",
                provider=VoiceProvider.ELEVENLABS,
                gender="female",
                age_range="young",
                accent="american",
                style="energetic",
                emotions=[VoiceEmotion.ENERGETIC, VoiceEmotion.CASUAL],
                cost_per_1k_chars=0.30,
                quality_score=0.97
            ),
            VoiceProfile(
                id="pNInz6obpgDQGcFmaJgB",
                name="Adam",
                provider=VoiceProvider.ELEVENLABS,
                gender="male",
                age_range="adult",
                accent="american",
                style="commercial",
                emotions=[VoiceEmotion.CONFIDENT, VoiceEmotion.URGENT],
                cost_per_1k_chars=0.30,
                quality_score=0.98
            )
        ]
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Synthesize speech using ElevenLabs API"""
        import aiohttp
        
        voice_id = voice_id or self.default_voice_id
        
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Voice settings for commercial quality
        stability = kwargs.get("stability", 0.5)
        similarity_boost = kwargs.get("similarity_boost", 0.75)
        style = kwargs.get("style", 0.5)
        
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                
                audio_bytes = await response.read()
        
        latency_ms = (time.time() - start_time) * 1000
        
        metadata = {
            "provider": "elevenlabs",
            "voice_id": voice_id,
            "model_id": self.model_id,
            "text_length": len(text),
            "audio_size_bytes": len(audio_bytes),
            "latency_ms": latency_ms,
            "cost_usd": self.estimate_cost(text),
            "quality_score": 0.98
        }
        
        return audio_bytes, metadata
    
    def get_voices(self) -> List[VoiceProfile]:
        return self.voices
    
    def estimate_cost(self, text: str) -> float:
        char_count = len(text)
        return (char_count / 1000) * self.cost_per_1k_chars


class AzureTTSProvider(TTSProvider):
    """
    Azure Cognitive Services TTS (Tier 2: High Quality)
    
    Features:
    - Neural voices
    - SSML support
    - Good cost/quality ratio
    """
    
    def __init__(self, config: VoiceoverConfig):
        self.speech_key = config.azure_speech_key
        self.region = config.azure_speech_region
        self.cost_per_1k_chars = 0.016
        self.voices = self._load_voices()
    
    def _load_voices(self) -> List[VoiceProfile]:
        return [
            VoiceProfile(
                id="en-US-JennyNeural",
                name="Jenny",
                provider=VoiceProvider.AZURE,
                gender="female",
                age_range="adult",
                accent="american",
                style="conversational",
                emotions=[VoiceEmotion.WARM, VoiceEmotion.PROFESSIONAL],
                cost_per_1k_chars=0.016,
                quality_score=0.92
            ),
            VoiceProfile(
                id="en-US-GuyNeural",
                name="Guy",
                provider=VoiceProvider.AZURE,
                gender="male",
                age_range="adult",
                accent="american",
                style="commercial",
                emotions=[VoiceEmotion.CONFIDENT, VoiceEmotion.PROFESSIONAL],
                cost_per_1k_chars=0.016,
                quality_score=0.91
            ),
            VoiceProfile(
                id="en-US-AriaNeural",
                name="Aria",
                provider=VoiceProvider.AZURE,
                gender="female",
                age_range="young",
                accent="american",
                style="energetic",
                emotions=[VoiceEmotion.ENERGETIC, VoiceEmotion.CASUAL],
                cost_per_1k_chars=0.016,
                quality_score=0.90
            )
        ]
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Synthesize using Azure Speech Services"""
        import aiohttp
        
        voice_id = voice_id or "en-US-JennyNeural"
        
        # Build SSML
        rate = kwargs.get("rate", "0%")  # -50% to +100%
        pitch = kwargs.get("pitch", "0%")  # -50% to +50%
        
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
               xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_id}">
                <prosody rate="{rate}" pitch="{pitch}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3"
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=ssml.encode()) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Azure TTS error: {response.status} - {error_text}")
                
                audio_bytes = await response.read()
        
        latency_ms = (time.time() - start_time) * 1000
        
        metadata = {
            "provider": "azure",
            "voice_id": voice_id,
            "text_length": len(text),
            "audio_size_bytes": len(audio_bytes),
            "latency_ms": latency_ms,
            "cost_usd": self.estimate_cost(text),
            "quality_score": 0.92
        }
        
        return audio_bytes, metadata
    
    def get_voices(self) -> List[VoiceProfile]:
        return self.voices
    
    def estimate_cost(self, text: str) -> float:
        char_count = len(text)
        return (char_count / 1000) * self.cost_per_1k_chars


class GTTSProvider(TTSProvider):
    """
    Google Text-to-Speech (gTTS) Provider (Tier 3: Free Fallback)
    
    Features:
    - Free
    - Always available
    - Basic quality
    """
    
    def __init__(self, config: VoiceoverConfig):
        self.cost_per_1k_chars = 0.0
        self.voices = self._load_voices()
    
    def _load_voices(self) -> List[VoiceProfile]:
        return [
            VoiceProfile(
                id="en-US-female",
                name="US Female",
                provider=VoiceProvider.GTTS,
                gender="female",
                age_range="adult",
                accent="american",
                style="neutral",
                emotions=[VoiceEmotion.NEUTRAL],
                cost_per_1k_chars=0.0,
                quality_score=0.70
            ),
            VoiceProfile(
                id="en-GB-female",
                name="UK Female",
                provider=VoiceProvider.GTTS,
                gender="female",
                age_range="adult",
                accent="british",
                style="neutral",
                emotions=[VoiceEmotion.NEUTRAL],
                cost_per_1k_chars=0.0,
                quality_score=0.70
            )
        ]
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Synthesize using gTTS (free fallback)"""
        from gtts import gTTS
        
        # Parse voice_id for language
        lang = "en"
        tld = "com"  # US accent
        
        if voice_id and "GB" in voice_id:
            tld = "co.uk"
        
        start_time = time.time()
        
        tts = gTTS(text=text, lang=lang, tld=tld)
        
        # Write to BytesIO
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()
        
        latency_ms = (time.time() - start_time) * 1000
        
        metadata = {
            "provider": "gtts",
            "voice_id": voice_id or "en-US-female",
            "text_length": len(text),
            "audio_size_bytes": len(audio_bytes),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "quality_score": 0.70
        }
        
        return audio_bytes, metadata
    
    def get_voices(self) -> List[VoiceProfile]:
        return self.voices
    
    def estimate_cost(self, text: str) -> float:
        return 0.0


# =============================================================================
# GOOGLE DRIVE ASSET MANAGER
# =============================================================================

class GoogleDriveAssetManager:
    """
    Google Drive integration for audio asset storage
    
    Features:
    - Service account authentication
    - Hierarchical folder structure
    - Direct bytes upload (no local storage)
    - Asset metadata tracking
    """
    
    def __init__(self, config: VoiceoverConfig):
        self.config = config
        self.service = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Google Drive service"""
        if self._initialized:
            return
        
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
            
            # Decode service account JSON from base64
            if not self.config.gdrive_sa_json_b64:
                logger.warning("GDRIVE_SA_JSON_B64 not set - Drive uploads disabled")
                return
            
            sa_json = base64.b64decode(self.config.gdrive_sa_json_b64).decode()
            sa_info = json.loads(sa_json)
            
            credentials = Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            
            self.service = build("drive", "v3", credentials=credentials)
            self._initialized = True
            logger.info("Google Drive service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive: {e}")
    
    async def ensure_folder_path(
        self,
        path_parts: List[str]
    ) -> str:
        """
        Ensure folder path exists, creating folders as needed
        
        Args:
            path_parts: ["Nexus", "Clients", "client_123", "audio"]
        
        Returns:
            folder_id of the final folder
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.service:
            raise Exception("Google Drive not initialized")
        
        parent_id = self.config.gdrive_root_folder_id
        
        for folder_name in path_parts:
            # Search for existing folder
            query = (
                f"name = '{folder_name}' and "
                f"'{parent_id}' in parents and "
                f"mimeType = 'application/vnd.google-apps.folder' and "
                f"trashed = false"
            )
            
            results = self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id, name)"
            ).execute()
            
            files = results.get("files", [])
            
            if files:
                parent_id = files[0]["id"]
            else:
                # Create folder
                file_metadata = {
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id]
                }
                
                folder = self.service.files().create(
                    body=file_metadata,
                    fields="id"
                ).execute()
                
                parent_id = folder["id"]
                logger.info(f"Created folder: {folder_name} ({parent_id})")
        
        return parent_id
    
    async def upload_audio(
        self,
        client_id: str,
        commercial_id: str,
        scene_id: int,
        audio_bytes: bytes,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Upload audio file to Google Drive
        
        Returns:
            {
                "file_id": "...",
                "web_view_link": "https://drive.google.com/file/d/.../view",
                "download_link": "https://drive.google.com/uc?id=...&export=download",
                "path": "BarriosA2I/Nexus/Clients/{client_id}/audio/..."
            }
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.service:
            logger.warning("Drive not available - skipping upload")
            return None
        
        from googleapiclient.http import MediaInMemoryUpload
        
        # Build folder path
        path_parts = [
            self.config.gdrive_clients_folder.split("/")[0],  # "Nexus"
            *self.config.gdrive_clients_folder.split("/")[1:],  # "Clients"
            client_id,
            commercial_id,
            "audio"
        ]
        
        folder_id = await self.ensure_folder_path(path_parts)
        
        # File name
        filename = f"vo_scene_{scene_id:02d}.mp3"
        
        # Upload
        file_metadata = {
            "name": filename,
            "parents": [folder_id],
            "description": json.dumps(metadata or {})
        }
        
        media = MediaInMemoryUpload(
            audio_bytes,
            mimetype="audio/mpeg",
            resumable=True
        )
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink"
        ).execute()
        
        file_id = file["id"]
        web_view_link = file.get("webViewLink", f"https://drive.google.com/file/d/{file_id}/view")
        download_link = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        full_path = f"BarriosA2I/{'/'.join(path_parts)}/{filename}"
        
        logger.info(f"Uploaded: {filename} -> {file_id}")
        
        return {
            "file_id": file_id,
            "web_view_link": web_view_link,
            "download_link": download_link,
            "path": full_path,
            "filename": filename
        }


# =============================================================================
# CRAG FALLBACK CHAIN
# =============================================================================

class VoiceoverCRAG:
    """
    Corrective RAG for Voiceover Generation
    
    Fallback chain:
    1. ElevenLabs (premium) - if available and budget allows
    2. Azure TTS (standard) - if ElevenLabs fails
    3. gTTS (free) - last resort fallback
    
    Decision tree:
    - Confidence > 0.7: GENERATE with primary provider
    - 0.4 < Confidence <= 0.7: OPTIMIZE script, try again
    - Confidence <= 0.4: FALLBACK to next provider
    """
    
    def __init__(
        self,
        config: VoiceoverConfig,
        providers: Dict[VoiceProvider, TTSProvider],
        circuit_breakers: Dict[VoiceProvider, CircuitBreaker]
    ):
        self.config = config
        self.providers = providers
        self.circuit_breakers = circuit_breakers
        
        # Provider priority chain
        self.provider_chain = [
            VoiceProvider.ELEVENLABS,
            VoiceProvider.AZURE,
            VoiceProvider.GTTS
        ]
    
    async def evaluate_generation(
        self,
        audio_bytes: bytes,
        metadata: Dict[str, Any],
        scene: SceneVoiceover
    ) -> Tuple[float, CRAGAction]:
        """
        Evaluate generated audio quality
        
        Returns: (confidence_score, recommended_action)
        """
        confidence = 0.5
        
        # Check audio size (should be reasonable for duration)
        expected_size = scene.duration_target_seconds * 20000  # ~20KB per second
        actual_size = len(audio_bytes)
        size_ratio = actual_size / expected_size if expected_size > 0 else 0
        
        if 0.5 <= size_ratio <= 2.0:
            confidence += 0.2
        
        # Check provider quality
        provider_quality = metadata.get("quality_score", 0.8)
        confidence += provider_quality * 0.2
        
        # Check latency (lower is better for premium providers)
        latency = metadata.get("latency_ms", 5000)
        if latency < 2000:
            confidence += 0.1
        
        # Determine action
        if confidence >= self.config.crag_high_threshold:
            return confidence, CRAGAction.GENERATE
        elif confidence >= self.config.crag_low_threshold:
            return confidence, CRAGAction.OPTIMIZE
        else:
            return confidence, CRAGAction.FALLBACK
    
    async def generate_with_fallback(
        self,
        text: str,
        voice_id: str,
        scene: SceneVoiceover,
        preferred_provider: VoiceProvider = None
    ) -> Tuple[bytes, Dict[str, Any], VoiceProvider]:
        """
        Generate voiceover with CRAG fallback chain
        
        Returns: (audio_bytes, metadata, provider_used)
        """
        # Build provider chain starting with preferred
        chain = self.provider_chain.copy()
        if preferred_provider and preferred_provider in chain:
            chain.remove(preferred_provider)
            chain.insert(0, preferred_provider)
        
        last_error = None
        
        for provider_type in chain:
            provider = self.providers.get(provider_type)
            breaker = self.circuit_breakers.get(provider_type)
            
            if not provider:
                continue
            
            # Check circuit breaker
            if breaker and not breaker.can_execute():
                logger.warning(f"Circuit OPEN for {provider_type.value} - skipping")
                continue
            
            try:
                # Map voice_id to provider-specific voice
                provider_voice_id = self._map_voice_to_provider(voice_id, provider_type)
                
                # Synthesize
                audio_bytes, metadata = await provider.synthesize(
                    text=text,
                    voice_id=provider_voice_id
                )
                
                # Evaluate quality
                confidence, action = await self.evaluate_generation(
                    audio_bytes, metadata, scene
                )
                
                metadata["crag_confidence"] = confidence
                metadata["crag_action"] = action.value
                
                if action == CRAGAction.GENERATE:
                    # Success!
                    if breaker:
                        breaker.record_success()
                    return audio_bytes, metadata, provider_type
                
                elif action == CRAGAction.OPTIMIZE:
                    # Quality acceptable but not ideal - use it but log
                    logger.info(f"CRAG OPTIMIZE: {provider_type.value} quality={confidence:.2f}")
                    if breaker:
                        breaker.record_success()
                    return audio_bytes, metadata, provider_type
                
                else:
                    # FALLBACK - try next provider
                    logger.warning(f"CRAG FALLBACK: {provider_type.value} quality={confidence:.2f}")
                    if breaker:
                        breaker.record_failure()
                    continue
                    
            except Exception as e:
                logger.error(f"Provider {provider_type.value} failed: {e}")
                if breaker:
                    breaker.record_failure()
                last_error = e
                continue
        
        # All providers failed
        raise Exception(f"All TTS providers failed. Last error: {last_error}")
    
    def _map_voice_to_provider(
        self,
        voice_id: str,
        provider_type: VoiceProvider
    ) -> str:
        """Map generic voice ID to provider-specific ID"""
        # Default mappings for common voice types
        voice_mappings = {
            # Generic -> Provider-specific
            "male_confident": {
                VoiceProvider.ELEVENLABS: "pNInz6obpgDQGcFmaJgB",  # Adam
                VoiceProvider.AZURE: "en-US-GuyNeural",
                VoiceProvider.GTTS: "en-US-male"
            },
            "female_warm": {
                VoiceProvider.ELEVENLABS: "21m00Tcm4TlvDq8ikWAM",  # Rachel
                VoiceProvider.AZURE: "en-US-JennyNeural",
                VoiceProvider.GTTS: "en-US-female"
            },
            "female_energetic": {
                VoiceProvider.ELEVENLABS: "EXAVITQu4vr4xnSDxMaL",  # Sarah
                VoiceProvider.AZURE: "en-US-AriaNeural",
                VoiceProvider.GTTS: "en-US-female"
            }
        }
        
        # Check if voice_id is a generic type
        if voice_id in voice_mappings:
            return voice_mappings[voice_id].get(provider_type, voice_id)
        
        # Return as-is (provider-specific ID)
        return voice_id


# =============================================================================
# VOICE MATCHING RAG
# =============================================================================

class VoiceMatchingRAG:
    """
    RAG-based Voice Matching System
    
    Uses memory + semantic matching to find optimal voice for:
    - Brand guidelines
    - Scene type
    - Emotional requirements
    - Client history
    """
    
    def __init__(
        self,
        memory: VoiceMemorySystem,
        providers: Dict[VoiceProvider, TTSProvider],
        prm: VoicePRM
    ):
        self.memory = memory
        self.providers = providers
        self.prm = prm
        
        # Build voice index
        self.voice_index = self._build_voice_index()
    
    def _build_voice_index(self) -> Dict[str, VoiceProfile]:
        """Build searchable index of all voices"""
        index = {}
        for provider in self.providers.values():
            for voice in provider.get_voices():
                index[voice.id] = voice
        return index
    
    async def match_voice(
        self,
        scene: SceneVoiceover,
        brand_guidelines: str = "",
        client_id: str = "",
        quality_tier: str = "premium"
    ) -> Tuple[VoiceProfile, float]:
        """
        Find best matching voice for scene
        
        Returns: (voice_profile, match_score)
        """
        candidates = []
        
        # Get all available voices
        all_voices = list(self.voice_index.values())
        
        # Filter by quality tier
        if quality_tier == "premium":
            all_voices = [v for v in all_voices 
                         if v.provider in [VoiceProvider.ELEVENLABS, VoiceProvider.OPENAI]]
        elif quality_tier == "standard":
            all_voices = [v for v in all_voices 
                         if v.provider in [VoiceProvider.ELEVENLABS, VoiceProvider.AZURE]]
        # budget = all voices
        
        for voice in all_voices:
            score = await self._compute_match_score(voice, scene, brand_guidelines)
            candidates.append((voice, score))
        
        # Check memory for client preferences
        if client_id:
            semantic_results = self.memory.semantic_query(client_id)
            for mapping in semantic_results[:3]:
                voice_id = mapping.get("voice_id")
                if voice_id in self.voice_index:
                    # Boost score for previously used voices
                    for i, (v, s) in enumerate(candidates):
                        if v.id == voice_id:
                            candidates[i] = (v, s + 0.15)
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not candidates:
            # Fallback to default
            default = list(self.voice_index.values())[0]
            return default, 0.5
        
        best_voice, best_score = candidates[0]
        
        # PRM verification
        prm_score = await self.prm.evaluate_voice_selection(scene, best_voice, brand_guidelines)
        
        # Combine scores
        final_score = (best_score * 0.6) + (prm_score * 0.4)
        
        return best_voice, final_score
    
    async def _compute_match_score(
        self,
        voice: VoiceProfile,
        scene: SceneVoiceover,
        brand_guidelines: str
    ) -> float:
        """Compute match score between voice and scene requirements"""
        score = 0.5  # Base score
        
        # Emotion matching (most important)
        if scene.emotion in voice.emotions:
            score += 0.25
        
        # Style matching
        style_scores = {
            SceneType.HOOK: {"commercial": 0.15, "energetic": 0.15, "narrative": 0.1},
            SceneType.CTA: {"commercial": 0.15, "conversational": 0.1},
            SceneType.PROBLEM: {"empathetic": 0.15, "conversational": 0.1},
            SceneType.SOLUTION: {"confident": 0.15, "commercial": 0.1},
            SceneType.BENEFITS: {"warm": 0.15, "conversational": 0.1},
            SceneType.PROOF: {"credible": 0.15, "narrative": 0.1}
        }
        
        if scene.scene_type in style_scores:
            score += style_scores[scene.scene_type].get(voice.style, 0.05)
        
        # Quality contribution
        score += voice.quality_score * 0.1
        
        # Brand guideline matching (if provided)
        if brand_guidelines:
            guidelines_lower = brand_guidelines.lower()
            
            # Gender preference
            if "male" in guidelines_lower and voice.gender == "male":
                score += 0.1
            elif "female" in guidelines_lower and voice.gender == "female":
                score += 0.1
            
            # Accent preference
            if "british" in guidelines_lower and voice.accent == "british":
                score += 0.1
            elif "american" in guidelines_lower and voice.accent == "american":
                score += 0.05
        
        return min(score, 1.0)


# =============================================================================
# SCRIPT OPTIMIZER WITH PRM BEAM SEARCH
# =============================================================================

class ScriptOptimizer:
    """
    PRM-Guided Script Optimization using Beam Search
    
    Optimizes voiceover scripts for:
    - Pacing (words per minute)
    - Emotional impact
    - Clarity and pronunciation
    - Duration targeting
    """
    
    def __init__(self, prm: VoicePRM, beam_width: int = 3):
        self.prm = prm
        self.beam_width = beam_width
    
    async def optimize(
        self,
        scene: SceneVoiceover,
        max_iterations: int = 3
    ) -> Tuple[str, float]:
        """
        Optimize script text for voiceover
        
        Returns: (optimized_text, quality_score)
        """
        current_text = scene.text
        best_score = 0.0
        
        for iteration in range(max_iterations):
            # Generate candidates
            candidates = await self._generate_candidates(current_text, scene)
            
            # Score each candidate with PRM
            scored = []
            for candidate in candidates:
                score = await self.prm.evaluate_script_optimization(
                    current_text, candidate, scene.scene_type
                )
                scored.append((candidate, score))
            
            # Select best candidate (beam search)
            scored.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_candidate_score = scored[0]
            
            # Check if we've improved
            if best_candidate_score > best_score:
                current_text = best_candidate
                best_score = best_candidate_score
            else:
                # Convergence
                break
        
        return current_text, best_score
    
    async def _generate_candidates(
        self,
        text: str,
        scene: SceneVoiceover
    ) -> List[str]:
        """Generate optimization candidates"""
        candidates = [text]  # Include original
        
        # 1. Add pacing markers
        paced_text = self._add_pacing(text, scene.pacing)
        if paced_text != text:
            candidates.append(paced_text)
        
        # 2. Emphasize keywords
        if scene.emphasis_words:
            emphasized = self._add_emphasis(text, scene.emphasis_words)
            candidates.append(emphasized)
        
        # 3. Adjust for duration
        adjusted = self._adjust_length(text, scene.duration_target_seconds)
        if adjusted != text:
            candidates.append(adjusted)
        
        # 4. Simplify complex sentences
        simplified = self._simplify(text)
        if simplified != text:
            candidates.append(simplified)
        
        return candidates[:self.beam_width + 1]
    
    def _add_pacing(self, text: str, pacing: str) -> str:
        """Add pacing markers based on pacing preference"""
        if pacing == "slow":
            # Add pauses after sentences
            text = text.replace(". ", "... ")
            text = text.replace("! ", "!... ")
        elif pacing == "fast":
            # Remove extra pauses
            text = text.replace("...", ".")
            text = text.replace("  ", " ")
        return text
    
    def _add_emphasis(self, text: str, emphasis_words: List[str]) -> str:
        """Add emphasis markers for important words"""
        for word in emphasis_words:
            # In production, this would use SSML
            # For now, we just ensure the word is present
            pass
        return text
    
    def _adjust_length(self, text: str, target_duration: float) -> str:
        """Adjust text length for target duration"""
        words = text.split()
        current_word_count = len(words)
        
        # Target WPM is ~150 for natural speech
        target_word_count = int(target_duration * 2.5)  # 150 WPM = 2.5 words/sec
        
        if current_word_count > target_word_count * 1.2:
            # Too long - try to shorten
            # Remove filler words
            filler_words = ["very", "really", "just", "quite", "actually", "basically"]
            words = [w for w in words if w.lower() not in filler_words]
        
        return " ".join(words)
    
    def _simplify(self, text: str) -> str:
        """Simplify complex sentences for better TTS"""
        # Replace complex punctuation
        text = text.replace(";", ",")
        text = text.replace(" - ", ", ")
        
        # Expand common abbreviations
        replacements = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "it's": "it is",
            "we're": "we are",
            "you're": "you are",
            "&": "and",
            "%": "percent"
        }
        
        for abbr, expanded in replacements.items():
            text = text.replace(abbr, expanded)
        
        return text


# =============================================================================
# MAIN VOICEOVER MASTER AGENT
# =============================================================================

class VoiceoverMasterAgent:
    """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║           VOICEOVER MASTER RAG AGENT - LEGENDARY EDITION                   ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                            ║
    ║  Neural RAG Brain Cognitive Architecture:                                  ║
    ║  • Test-Time Reasoning with PRM Beam Search                               ║
    ║  • Dual-Process System 1/System 2 Routing                                 ║
    ║  • Self-Reflective RAG with Reflection Tokens                             ║
    ║  • CRAG Corrective Actions (ElevenLabs → Azure → gTTS)                   ║
    ║  • 4-Tier Hierarchical Memory (Working/Episodic/Semantic/Procedural)      ║
    ║  • Circuit Breakers with Provider-Level Isolation                         ║
    ║  • Google Drive Asset Integration                                         ║
    ║                                                                            ║
    ║  Performance Targets:                                                      ║
    ║  • Voice Match Accuracy: 0.94+                                            ║
    ║  • Audio Quality Score: 0.96+                                             ║
    ║  • Fallback Success Rate: 99.5%                                           ║
    ║  • p95 Latency: <8s per scene                                             ║
    ║  • Cost/Scene: <$0.15 (tiered routing)                                    ║
    ║                                                                            ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, config: VoiceoverConfig = None):
        self.config = config or VoiceoverConfig.from_env()
        
        # Initialize components
        self.memory = VoiceMemorySystem(self.config)
        self.prm = VoicePRM()
        
        # Initialize providers
        self.providers: Dict[VoiceProvider, TTSProvider] = {}
        self._init_providers()
        
        # Initialize circuit breakers
        self.circuit_breakers = {
            provider: CircuitBreaker(
                name=provider.value,
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout
            )
            for provider in VoiceProvider
        }
        
        # Initialize cognitive components
        self.router = VoiceDualProcessRouter(self.memory, self.prm)
        self.reflector = VoiceSelfReflectiveRAG(self.memory, self.prm)
        self.crag = VoiceoverCRAG(self.config, self.providers, self.circuit_breakers)
        self.voice_matcher = VoiceMatchingRAG(self.memory, self.providers, self.prm)
        self.script_optimizer = ScriptOptimizer(self.prm)
        
        # Google Drive
        self.drive = GoogleDriveAssetManager(self.config)
        
        # Agent metadata
        self.name = "voiceover_master"
        self.version = "1.0.0"
        self.status = "LEGENDARY"
    
    def _init_providers(self):
        """Initialize TTS providers based on available credentials"""
        # ElevenLabs (Tier 1)
        if self.config.elevenlabs_api_key:
            self.providers[VoiceProvider.ELEVENLABS] = ElevenLabsProvider(self.config)
            logger.info("✓ ElevenLabs provider initialized")
        
        # Azure (Tier 2)
        if self.config.azure_speech_key:
            self.providers[VoiceProvider.AZURE] = AzureTTSProvider(self.config)
            logger.info("✓ Azure TTS provider initialized")
        
        # gTTS (Tier 3 - always available)
        self.providers[VoiceProvider.GTTS] = GTTSProvider(self.config)
        logger.info("✓ gTTS provider initialized (fallback)")
    
    @tracer.start_as_current_span("voiceover_master.generate")
    async def generate(
        self,
        request: VoiceoverRequest
    ) -> VoiceoverResult:
        """
        Main entry point for voiceover generation
        
        Full cognitive pipeline:
        1. Dual-process routing
        2. Voice matching with RAG
        3. Script optimization with PRM
        4. Audio generation with CRAG fallback
        5. Quality verification with self-reflection
        6. Google Drive asset storage
        """
        start_time = time.time()
        
        span = trace.get_current_span()
        span.set_attribute("client_id", request.client_id)
        span.set_attribute("scene_count", len(request.scenes))
        
        # Initialize results
        scene_results = []
        drive_assets = []
        total_cost = 0.0
        quality_scores = []
        
        # Step 1: Route request (System 1 vs System 2)
        routing = await self.router.route(request)
        logger.info(f"Routing decision: System {routing['system']} ({routing['reason']})")
        
        # Step 2: Get voice context from memory
        first_scene = request.scenes[0] if request.scenes else {}
        scene_type = SceneType(first_scene.get("type", "hook"))
        
        memory_context = await self.memory.retrieve_voice_context(
            client_id=request.client_id,
            scene_type=scene_type,
            brand_description=request.brand_voice_guidelines or ""
        )
        
        # Step 3: Match optimal voice
        sample_scene = SceneVoiceover(
            scene_id=1,
            scene_type=scene_type,
            text=first_scene.get("text", ""),
            duration_target_seconds=first_scene.get("duration", 5.0),
            emotion=VoiceEmotion(first_scene.get("emotion", "confident"))
        )
        
        voice_profile, match_score = await self.voice_matcher.match_voice(
            scene=sample_scene,
            brand_guidelines=request.brand_voice_guidelines or "",
            client_id=request.client_id,
            quality_tier=request.quality_tier
        )
        
        logger.info(f"Voice matched: {voice_profile.name} (score={match_score:.2f})")
        
        # Step 4: Process each scene
        for i, scene_config in enumerate(request.scenes):
            scene = SceneVoiceover(
                scene_id=i + 1,
                scene_type=SceneType(scene_config.get("type", "hook")),
                text=scene_config.get("text", ""),
                duration_target_seconds=scene_config.get("duration", 5.0),
                emotion=VoiceEmotion(scene_config.get("emotion", "confident")),
                pacing=scene_config.get("pacing", "normal"),
                emphasis_words=scene_config.get("emphasis", [])
            )
            
            # [RET] Should we retrieve additional context?
            should_retrieve, retrieve_reason = await self.reflector.should_retrieve(
                scene, memory_context
            )
            
            if should_retrieve:
                # Get procedural patterns for this scene type
                patterns = self.memory.procedural_query(scene.scene_type)
                if patterns:
                    memory_context["procedural"] = patterns
            
            # [REL] Check voice relevance
            is_relevant, relevance = await self.reflector.is_relevant(
                scene, voice_profile, match_score
            )
            
            if not is_relevant and match_score < 0.6:
                # Try to find better voice
                voice_profile, match_score = await self.voice_matcher.match_voice(
                    scene=scene,
                    brand_guidelines=request.brand_voice_guidelines or "",
                    client_id=request.client_id,
                    quality_tier=request.quality_tier
                )
            
            # Optimize script with PRM
            optimized_text, optimization_score = await self.script_optimizer.optimize(scene)
            scene.text = optimized_text
            
            # PRM verification before generation
            prm_prediction = await self.prm.evaluate_audio_quality_prediction(
                scene.text, voice_profile, scene.duration_target_seconds
            )
            
            if prm_prediction < self.config.prm_threshold:
                logger.warning(f"PRM prediction low ({prm_prediction:.2f}) - may need fallback")
            
            # Generate audio with CRAG fallback
            audio_bytes, metadata, provider_used = await self.crag.generate_with_fallback(
                text=scene.text,
                voice_id=voice_profile.id,
                scene=scene,
                preferred_provider=voice_profile.provider
            )
            
            # [SUP] Check if audio meets requirements
            quality_metrics = {
                "duration_seconds": len(audio_bytes) / 44100,  # Approximate
                "quality_score": metadata.get("quality_score", 0.8),
                "has_silence_issues": len(audio_bytes) < 1000
            }
            
            is_supported, support_score = await self.reflector.is_supported(
                audio_bytes, scene, quality_metrics
            )
            
            if not is_supported:
                # Retry with different settings
                logger.warning(f"Scene {scene.scene_id} not supported - retrying")
                audio_bytes, metadata, provider_used = await self.crag.generate_with_fallback(
                    text=scene.text,
                    voice_id=voice_profile.id,
                    scene=scene,
                    preferred_provider=VoiceProvider.ELEVENLABS if voice_profile.provider != VoiceProvider.ELEVENLABS else VoiceProvider.AZURE
                )
                metadata["retry_count"] = 1
            
            # Upload to Google Drive
            drive_result = await self.drive.upload_audio(
                client_id=request.client_id,
                commercial_id=request.commercial_id,
                scene_id=scene.scene_id,
                audio_bytes=audio_bytes,
                metadata={
                    "voice_id": voice_profile.id,
                    "provider": provider_used.value,
                    "text": scene.text[:100],
                    "scene_type": scene.scene_type.value
                }
            )
            
            if drive_result:
                drive_assets.append(drive_result)
            
            # [USE] Check usefulness
            scene_result = {
                "scene_id": scene.scene_id,
                "scene_type": scene.scene_type.value,
                "text": scene.text,
                "audio_bytes": len(audio_bytes),
                "provider": provider_used.value,
                "voice_id": voice_profile.id,
                "quality_score": metadata.get("quality_score", 0.8),
                "cost_usd": metadata.get("cost_usd", 0),
                "drive": drive_result
            }
            
            is_useful, usefulness_score = await self.reflector.is_useful(
                scene, {"audio_bytes": audio_bytes, **scene_result}
            )
            
            scene_result["usefulness_score"] = usefulness_score
            scene_results.append(scene_result)
            
            total_cost += metadata.get("cost_usd", 0)
            quality_scores.append(metadata.get("quality_score", 0.8))
            
            # Store in memory
            self.memory.episodic_store(
                request.client_id,
                voice_profile.id,
                scene_result
            )
        
        # Store brand-voice mapping in semantic memory
        self.memory.semantic_store(
            request.client_id,
            voice_profile,
            match_score
        )
        
        # Calculate totals
        total_duration = sum(s.get("duration", 5.0) for s in request.scenes)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get reflection summary
        reflection_summary = self.reflector.get_reflection_summary()
        
        result = VoiceoverResult(
            client_id=request.client_id,
            commercial_id=request.commercial_id,
            scenes=scene_results,
            total_duration_seconds=total_duration,
            total_cost_usd=total_cost,
            provider_used=voice_profile.provider.value,
            voice_profile={
                "id": voice_profile.id,
                "name": voice_profile.name,
                "provider": voice_profile.provider.value,
                "match_score": match_score
            },
            drive_assets=drive_assets,
            quality_metrics={
                "avg_quality_score": avg_quality,
                "reflection_summary": reflection_summary,
                "prm_prediction": prm_prediction,
                "optimization_score": optimization_score
            },
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"✅ Voiceover generation complete: "
            f"{len(scene_results)} scenes, "
            f"${total_cost:.3f} cost, "
            f"{processing_time_ms:.0f}ms"
        )
        
        return result
    
    async def generate_scene_voiceover(
        self,
        client_id: str,
        scene_text: str,
        scene_type: str = "hook",
        duration_seconds: float = 5.0,
        emotion: str = "confident",
        voice_id: str = None
    ) -> Dict[str, Any]:
        """
        Simplified API for single scene voiceover
        
        Useful for quick testing or individual scene regeneration
        """
        request = VoiceoverRequest(
            client_id=client_id,
            commercial_id=f"single_{int(time.time())}",
            scenes=[{
                "type": scene_type,
                "text": scene_text,
                "duration": duration_seconds,
                "emotion": emotion
            }],
            quality_tier="premium"
        )
        
        result = await self.generate(request)
        
        return {
            "audio_url": result.drive_assets[0]["web_view_link"] if result.drive_assets else None,
            "download_url": result.drive_assets[0]["download_link"] if result.drive_assets else None,
            "quality_score": result.quality_metrics.get("avg_quality_score", 0.8),
            "cost_usd": result.total_cost_usd,
            "provider": result.provider_used
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and health"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "providers": {
                provider.value: {
                    "available": provider in self.providers,
                    "circuit_state": self.circuit_breakers[provider].state.value,
                    "failure_count": self.circuit_breakers[provider].failure_count
                }
                for provider in VoiceProvider
            },
            "memory": {
                "working_memory_size": len(self.memory.working_memory),
                "episodic_count": len(self.memory.episodic_memory),
                "semantic_count": len(self.memory.semantic_memory),
                "procedural_count": len(self.memory.procedural_memory)
            },
            "cognitive_components": [
                "test_time_reasoning",
                "dual_process_routing",
                "self_reflective_rag",
                "crag_fallback",
                "hierarchical_memory",
                "circuit_breakers"
            ]
        }


# =============================================================================
# VALIDATION CHECKLIST
# =============================================================================

def validate_voiceover_agent(agent: VoiceoverMasterAgent) -> Dict[str, bool]:
    """
    Validate that the agent has all required Neural RAG Brain components
    """
    checks = {
        # Cognitive Architecture
        "has_test_time_reasoning": hasattr(agent, "prm") and agent.prm is not None,
        "has_dual_process": hasattr(agent, "router") and agent.router is not None,
        "has_reflection_tokens": hasattr(agent, "reflector") and agent.reflector is not None,
        "has_crag_fallback": hasattr(agent, "crag") and agent.crag is not None,
        "has_hierarchical_memory": (
            hasattr(agent, "memory") and 
            hasattr(agent.memory, "working_memory") and
            hasattr(agent.memory, "episodic_memory") and
            hasattr(agent.memory, "semantic_memory") and
            hasattr(agent.memory, "procedural_memory")
        ),
        
        # Production Quality
        "has_circuit_breakers": hasattr(agent, "circuit_breakers") and len(agent.circuit_breakers) > 0,
        "has_voice_matching_rag": hasattr(agent, "voice_matcher") and agent.voice_matcher is not None,
        "has_script_optimizer": hasattr(agent, "script_optimizer") and agent.script_optimizer is not None,
        "has_google_drive": hasattr(agent, "drive") and agent.drive is not None,
        
        # Provider Chain
        "has_multiple_providers": len(agent.providers) >= 2,
        "has_fallback_provider": VoiceProvider.GTTS in agent.providers,
        
        # Observability
        "has_tracing": "@tracer.start_as_current_span" in str(agent.generate.__wrapped__ if hasattr(agent.generate, "__wrapped__") else ""),
    }
    
    failed = [k for k, v in checks.items() if not v]
    
    if failed:
        logger.warning(f"Validation failed: {failed}")
    else:
        logger.info("✅ S+++++ Neural RAG Voiceover Agent validated")
    
    return checks


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the VoiceoverMasterAgent"""
    
    # Initialize agent
    config = VoiceoverConfig.from_env()
    agent = VoiceoverMasterAgent(config)
    
    # Validate
    validation = validate_voiceover_agent(agent)
    print(f"Validation: {sum(validation.values())}/{len(validation)} checks passed")
    
    # Get status
    status = agent.get_status()
    print(f"Agent Status: {json.dumps(status, indent=2)}")
    
    # Example request
    request = VoiceoverRequest(
        client_id="demo_client_001",
        commercial_id="commercial_001",
        scenes=[
            {
                "type": "hook",
                "text": "Tired of wasting hours on manual tasks?",
                "duration": 3.0,
                "emotion": "urgent"
            },
            {
                "type": "solution",
                "text": "Barrios A2I automates your business processes with intelligent AI agents that work 24/7.",
                "duration": 6.0,
                "emotion": "confident"
            },
            {
                "type": "benefits",
                "text": "Our clients save an average of 40 hours per week. Imagine what you could do with that time.",
                "duration": 5.0,
                "emotion": "warm"
            },
            {
                "type": "cta",
                "text": "Book your strategy call today. Visit Barrios A2I dot com.",
                "duration": 4.0,
                "emotion": "urgent"
            }
        ],
        brand_voice_guidelines="Male voice, confident but approachable, American accent",
        quality_tier="premium"
    )
    
    # Generate voiceover
    result = await agent.generate(request)
    
    print(f"\n{'='*60}")
    print("VOICEOVER GENERATION RESULT")
    print(f"{'='*60}")
    print(f"Total Duration: {result.total_duration_seconds}s")
    print(f"Total Cost: ${result.total_cost_usd:.3f}")
    print(f"Provider: {result.provider_used}")
    print(f"Voice: {result.voice_profile['name']} (match={result.voice_profile['match_score']:.2f})")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")
    print(f"\nScenes:")
    for scene in result.scenes:
        print(f"  Scene {scene['scene_id']}: {scene['scene_type']} - {scene['quality_score']:.2f} quality")
        if scene.get('drive'):
            print(f"    Drive: {scene['drive']['web_view_link']}")
    print(f"\nQuality Metrics: {json.dumps(result.quality_metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
