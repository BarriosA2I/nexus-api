"""
NEXUS BRAIN v5.0 APEX - Complexity Classifier
==============================================
System 1 (fast/intuitive) vs System 2 (slow/analytical) routing.

Inspired by Kahneman's dual-process theory:
- System 1: Fast, automatic, pattern-matching (greetings, FAQs, simple lookups)
- System 2: Slow, deliberate, reasoning (analysis, comparisons, multi-step)

Signals analyzed:
- Word count and sentence structure
- Question complexity markers
- Technical jargon density
- Multi-step reasoning indicators
- Comparison/analysis requests
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .state import ClassifierResult, ComplexityLevel

logger = logging.getLogger("nexus.classifier")


# =============================================================================
# SIGNAL PATTERNS
# =============================================================================

# System 1 patterns (fast response expected)
SYSTEM_1_PATTERNS: Dict[str, List[str]] = {
    "greetings": [
        r"\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b",
        r"^(hi|hello|hey)[\s!?.]*$",
    ],
    "simple_questions": [
        r"^(what|who|when|where)\s+(is|are|was|were)\s+\w+\??$",
        r"^how\s+much\s+(is|does|are)\s+",
        r"^(yes|no|ok|okay|sure|thanks?|thank\s+you)[\s!?.]*$",
    ],
    "faq_triggers": [
        r"\b(price|pricing|cost|how\s+much)\b",
        r"\b(contact|email|phone|call)\b",
        r"\b(hours|open|available)\b",
        r"\b(location|address|where\s+are\s+you)\b",
    ],
    "direct_lookups": [
        r"^what\s+(service|product|package)s?\s+(do\s+you|does\s+\w+)\s+offer",
        r"\b(list|show|tell\s+me)\s+(your|the)\s+(service|product|pricing)",
    ],
}

# System 2 patterns (complex reasoning required)
SYSTEM_2_PATTERNS: Dict[str, List[str]] = {
    "comparison": [
        r"\b(compare|comparison|versus|vs\.?|difference|better|worse)\b",
        r"\b(which\s+(is|should|would))\b",
        r"\bpros?\s+and\s+cons?\b",
    ],
    "analysis": [
        r"\b(analyze|analysis|evaluate|assessment)\b",
        r"\b(why|how\s+does|explain|elaborate)\b.*\?",
        r"\b(recommend|suggest|advise|help\s+me\s+(decide|choose))\b",
    ],
    "multi_step": [
        r"\b(first|then|after\s+that|next|finally|step\s+by\s+step)\b",
        r"\b(if.*then|when.*should|depending\s+on)\b",
        r"\b(and\s+also|additionally|furthermore|moreover)\b",
    ],
    "technical": [
        r"\b(implement|integrate|architecture|system|pipeline)\b",
        r"\b(api|sdk|database|server|deployment)\b",
        r"\b(algorithm|optimization|scalab|performance)\b",
    ],
    "strategic": [
        r"\b(strategy|strategic|roadmap|plan|approach)\b",
        r"\b(roi|revenue|budget|investment)\b",
        r"\b(long[\s-]term|short[\s-]term|timeline)\b",
    ],
}

# Question word weights (higher = more complex)
QUESTION_WEIGHTS: Dict[str, float] = {
    "who": 0.2,
    "what": 0.3,
    "when": 0.2,
    "where": 0.2,
    "how": 0.5,
    "why": 0.7,
    "which": 0.4,
    "should": 0.6,
    "could": 0.5,
    "would": 0.5,
}


@dataclass
class SignalScore:
    """Individual signal contribution to complexity."""
    name: str
    weight: float
    triggered: bool
    matches: List[str] = field(default_factory=list)


class ComplexityClassifier:
    """
    Classify query complexity for System 1/2 routing.

    Uses multi-signal analysis:
    1. Pattern matching (positive/negative indicators)
    2. Structural analysis (length, punctuation, clauses)
    3. Question word analysis
    4. Technical jargon density

    Threshold tuning:
    - < 0.4: System 1 (fast path)
    - >= 0.4: System 2 (slow path)
    """

    SYSTEM_2_THRESHOLD = 0.4
    MIN_CONFIDENCE = 0.6

    def __init__(
        self,
        system_1_patterns: Optional[Dict[str, List[str]]] = None,
        system_2_patterns: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.4,
    ):
        """
        Initialize classifier.

        Args:
            system_1_patterns: Fast-path pattern overrides
            system_2_patterns: Slow-path pattern overrides
            threshold: Complexity score threshold for System 2
        """
        self.system_1_patterns = system_1_patterns or SYSTEM_1_PATTERNS
        self.system_2_patterns = system_2_patterns or SYSTEM_2_PATTERNS
        self.threshold = threshold

        # Compile patterns for efficiency
        self._compiled_s1: Dict[str, List[re.Pattern]] = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.system_1_patterns.items()
        }
        self._compiled_s2: Dict[str, List[re.Pattern]] = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.system_2_patterns.items()
        }

    def classify(self, message: str) -> ClassifierResult:
        """
        Classify message complexity.

        Args:
            message: User's input message

        Returns:
            ClassifierResult with level, confidence, and signals
        """
        signals: Dict[str, SignalScore] = {}

        # 1. Pattern matching
        s1_score, s1_signals = self._score_patterns(message, self._compiled_s1, -1)
        s2_score, s2_signals = self._score_patterns(message, self._compiled_s2, 1)
        signals.update(s1_signals)
        signals.update(s2_signals)

        # 2. Structural analysis
        struct_score, struct_signals = self._analyze_structure(message)
        signals.update(struct_signals)

        # 3. Question word analysis
        question_score, question_signals = self._analyze_questions(message)
        signals.update(question_signals)

        # 4. Technical jargon
        tech_score, tech_signals = self._analyze_jargon(message)
        signals.update(tech_signals)

        # Combine scores (weighted average)
        weights = {
            "patterns": 0.35,
            "structure": 0.25,
            "questions": 0.20,
            "jargon": 0.20,
        }

        raw_score = (
            (s1_score + s2_score) * weights["patterns"] +
            struct_score * weights["structure"] +
            question_score * weights["questions"] +
            tech_score * weights["jargon"]
        )

        # Normalize to 0-1 range
        complexity_score = max(0.0, min(1.0, (raw_score + 1) / 2))

        # Determine level
        if complexity_score >= self.threshold:
            level = ComplexityLevel.SYSTEM_2
            reasoning = "Complex query requiring deliberate analysis"
        else:
            level = ComplexityLevel.SYSTEM_1
            reasoning = "Simple query suitable for fast response"

        # Calculate confidence based on signal agreement
        triggered_s1 = sum(1 for s in signals.values() if s.triggered and s.weight < 0)
        triggered_s2 = sum(1 for s in signals.values() if s.triggered and s.weight > 0)

        if triggered_s1 + triggered_s2 == 0:
            confidence = 0.5  # Uncertain
        else:
            # Higher confidence when signals agree
            dominance = abs(triggered_s2 - triggered_s1) / (triggered_s1 + triggered_s2)
            confidence = 0.5 + (dominance * 0.5)

        result = ClassifierResult(
            level=level,
            confidence=round(confidence, 3),
            signals={
                name: {
                    "weight": s.weight,
                    "triggered": s.triggered,
                    "matches": s.matches[:3],  # Limit matches for logging
                }
                for name, s in signals.items()
                if s.triggered
            },
            reasoning=reasoning,
        )

        logger.debug(
            f"Complexity: {level.value} (score={complexity_score:.2f}, "
            f"confidence={confidence:.2%}) - {reasoning}"
        )

        return result

    def _score_patterns(
        self,
        message: str,
        patterns: Dict[str, List[re.Pattern]],
        direction: int,  # -1 for S1, +1 for S2
    ) -> Tuple[float, Dict[str, SignalScore]]:
        """Score message against pattern dictionary."""
        signals: Dict[str, SignalScore] = {}
        total_score = 0.0

        for category, compiled in patterns.items():
            matches = []
            for pattern in compiled:
                found = pattern.findall(message)
                matches.extend(found)

            triggered = len(matches) > 0
            weight = direction * (0.2 + 0.1 * len(matches))  # Scale with match count

            signals[f"pattern_{category}"] = SignalScore(
                name=category,
                weight=weight if triggered else 0,
                triggered=triggered,
                matches=matches[:5],
            )

            if triggered:
                total_score += weight

        return total_score, signals

    def _analyze_structure(self, message: str) -> Tuple[float, Dict[str, SignalScore]]:
        """Analyze message structure for complexity indicators."""
        signals: Dict[str, SignalScore] = {}

        # Word count
        words = message.split()
        word_count = len(words)

        long_message = word_count > 30
        signals["struct_length"] = SignalScore(
            name="message_length",
            weight=0.3 if long_message else -0.1,
            triggered=True,
            matches=[f"{word_count} words"],
        )

        # Clause count (rough approximation via punctuation)
        clause_markers = len(re.findall(r"[,;:]", message))
        complex_structure = clause_markers >= 2

        signals["struct_clauses"] = SignalScore(
            name="clause_count",
            weight=0.2 if complex_structure else 0,
            triggered=complex_structure,
            matches=[f"{clause_markers} clause markers"],
        )

        # Multiple sentences
        sentences = len(re.findall(r"[.!?]+", message))
        multi_sentence = sentences >= 2

        signals["struct_sentences"] = SignalScore(
            name="sentence_count",
            weight=0.15 if multi_sentence else 0,
            triggered=multi_sentence,
            matches=[f"{sentences} sentences"],
        )

        total = sum(s.weight for s in signals.values() if s.triggered)
        return total, signals

    def _analyze_questions(self, message: str) -> Tuple[float, Dict[str, SignalScore]]:
        """Analyze question words for complexity."""
        signals: Dict[str, SignalScore] = {}
        message_lower = message.lower()

        total_weight = 0.0
        for word, weight in QUESTION_WEIGHTS.items():
            pattern = rf"\b{word}\b"
            if re.search(pattern, message_lower):
                signals[f"question_{word}"] = SignalScore(
                    name=f"question_{word}",
                    weight=weight,
                    triggered=True,
                    matches=[word],
                )
                total_weight += weight

        return total_weight, signals

    def _analyze_jargon(self, message: str) -> Tuple[float, Dict[str, SignalScore]]:
        """Analyze technical jargon density."""
        signals: Dict[str, SignalScore] = {}

        # Technical terms
        tech_terms = [
            r"\bapi\b", r"\bsdk\b", r"\bdatabase\b", r"\bserver\b",
            r"\bpipeline\b", r"\bdeployment\b", r"\bintegration\b",
            r"\barchitecture\b", r"\bscaling\b", r"\blatency\b",
            r"\brag\b", r"\bllm\b", r"\bml\b", r"\bai\b",
            r"\bagent\b", r"\borchestrat", r"\bvector\b",
        ]

        matches = []
        for term in tech_terms:
            if re.search(term, message, re.IGNORECASE):
                matches.append(term.replace(r"\b", ""))

        jargon_density = len(matches) / max(len(message.split()), 1)
        high_jargon = jargon_density > 0.1 or len(matches) >= 3

        signals["jargon_density"] = SignalScore(
            name="technical_jargon",
            weight=0.4 if high_jargon else 0.1 if len(matches) > 0 else 0,
            triggered=len(matches) > 0,
            matches=matches[:5],
        )

        return signals["jargon_density"].weight, signals


# =============================================================================
# SINGLETON CLASSIFIER
# =============================================================================

_classifier: Optional[ComplexityClassifier] = None


def get_complexity_classifier() -> ComplexityClassifier:
    """Get singleton complexity classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = ComplexityClassifier()
    return _classifier


def classify_complexity(message: str) -> ClassifierResult:
    """Convenience function to classify message complexity."""
    classifier = get_complexity_classifier()
    return classifier.classify(message)
