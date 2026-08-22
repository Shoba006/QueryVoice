import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.retrieval.hybrid_search import RetrievedChunk
from app.generation.generator import GenerationResult

logger = logging.getLogger(__name__)

# Basic safety keywords to block
UNSAFE_PATTERNS = [
    r"\b(bomb|explosive|weapon|kill|harm|hack|poison|suicide)\b",
    r"\b(माड़|बम|हत्या|हिंसा)\b"
]

class GuardrailVerdict(BaseModel):
    is_safe: bool = Field(default=True, description="True if query passed safety filter")
    is_on_topic: bool = Field(default=True, description="True if query is within domain scope")
    is_grounded: bool = Field(default=True, description="True if answer is supported by context")
    should_abstain: bool = Field(default=False, description="True if system must refuse/abstain")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection or abstention")
    guardrail_case: str = Field(default="Case 1: Normal Answer", description="Evaluated guardrail scenario (Case 1..5)")
    off_topic_score: float = Field(default=1.0, description="Similarity score to top retrieved context")
    groundedness_score: float = Field(default=1.0, description="Similarity score between answer and context")

class GuardrailSystem:
    """
    Implements 5 functional safety guardrail scenarios:
    - Case 1: On-topic + Grounded -> Answer normally
    - Case 2: Off-topic -> Refuse/redirect early without hallucination
    - Case 3: Unsafe/inappropriate input -> Safe refusal
    - Case 4: No relevant retrieved context -> Explicit abstention
    - Case 5: Contradictory / Un-grounded answer -> Reject/abstain
    """
    def __init__(
        self,
        off_topic_threshold: float = 0.22,
        groundedness_threshold: float = 0.30,
        embedder: Optional[Any] = None
    ):
        self.off_topic_threshold = off_topic_threshold
        self.groundedness_threshold = groundedness_threshold
        self.embedder = embedder

    def check_input_safety(self, query: str) -> bool:
        """Filter out unsafe/harmful queries."""
        query_lower = query.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Unsafe query detected matching pattern: {pattern}")
                return False
        return True

    def check_off_topic(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> tuple[bool, float]:
        """Verify query is answerable from retrieved corpus context."""
        if not retrieved_chunks:
            return False, 0.0

        top_dense = retrieved_chunks[0].dense_score
        if top_dense is not None:
            score = float(top_dense)
        else:
            q_words = set(query.lower().split())
            c_words = set(retrieved_chunks[0].text.lower().split())
            score = len(q_words.intersection(c_words)) / max(1, len(q_words))

        is_on_topic = score >= self.off_topic_threshold
        return is_on_topic, round(score, 3)

    def check_groundedness(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        cited_ids: List[str]
    ) -> tuple[bool, float]:
        """
        Verify generated answer claims are supported by cited/retrieved chunks.
        """
        if "don't have enough information" in answer.lower() or "पर्याप्त जानकारी नहीं" in answer:
            return True, 1.0

        if not retrieved_chunks:
            return False, 0.0

        target_chunks = [c for c in retrieved_chunks if c.chunk_id in cited_ids]
        if not target_chunks:
            target_chunks = retrieved_chunks[:2]

        context_combined = " ".join([c.text for c in target_chunks])

        # Fast word-overlap groundedness check to prevent extra neural encoding delays
        a_words = set(re.findall(r'\w+', answer.lower()))
        c_words = set(re.findall(r'\w+', context_combined.lower()))
        if not a_words:
            return True, 1.0
        score = len(a_words.intersection(c_words)) / float(len(a_words))

        is_grounded = score >= self.groundedness_threshold
        return is_grounded, round(score, 3)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        generation_result: Optional[GenerationResult] = None
    ) -> GuardrailVerdict:
        """Evaluates all guardrail checks and produces a unified verdict."""
        # Case 3: Unsafe input check
        is_safe = self.check_input_safety(query)
        if not is_safe:
            return GuardrailVerdict(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="Query flagged by input safety filter.",
                guardrail_case="Case 3: Unsafe Query Refusal",
                off_topic_score=0.0,
                groundedness_score=0.0
            )

        # Case 4: No relevant retrieved context
        if not retrieved_chunks:
            return GuardrailVerdict(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="No relevant context retrieved from index.",
                guardrail_case="Case 4: Empty Retrieval Abstention",
                off_topic_score=0.0,
                groundedness_score=0.0
            )

        # Case 2: Off-topic check
        is_on_topic, off_score = self.check_off_topic(query, retrieved_chunks)
        if not is_on_topic:
            return GuardrailVerdict(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="Query is out of scope / off-topic for corpus.",
                guardrail_case="Case 2: Off-Topic Refusal",
                off_topic_score=off_score,
                groundedness_score=0.0
            )

        # Case 5 & Case 1: Groundedness check (if generation was executed)
        is_grounded = True
        ground_score = 1.0
        if generation_result:
            is_grounded, ground_score = self.check_groundedness(
                generation_result.answer,
                retrieved_chunks,
                generation_result.cited_chunk_ids
            )

        if not is_grounded:
            return GuardrailVerdict(
                is_safe=True,
                is_on_topic=True,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="Generated answer failed context groundedness check (potential hallucination).",
                guardrail_case="Case 5: Low Groundedness Abstention",
                off_topic_score=off_score,
                groundedness_score=ground_score
            )

        return GuardrailVerdict(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            should_abstain=False,
            rejection_reason=None,
            guardrail_case="Case 1: Normal Answer",
            off_topic_score=off_score,
            groundedness_score=ground_score
        )
