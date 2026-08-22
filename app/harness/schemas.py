from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.stt.sarvam_client import TranscriptionResult
from app.retrieval.hybrid_search import RetrievedChunk, RetrievalResult
from app.generation.generator import GenerationResult
from app.guardrails.safety import GuardrailVerdict

class PipelineRequest(BaseModel):
    audio_bytes: Optional[bytes] = Field(default=None, description="Raw audio payload (wav/mp3/webm)")
    audio_filename: str = Field(default="audio.wav", description="Filename hint for audio format")
    text_query: Optional[str] = Field(default=None, description="Text query if bypassing STT directly")
    chunk_strategy: str = Field(default="semantic", description="Selected chunking strategy ('fixed', 'semantic', 'metadata_aware', 'recursive')")
    top_k: int = Field(default=5, description="Number of context chunks to retrieve")
    language_filter: Optional[str] = Field(default=None, description="Language code filter (e.g. 'hi', 'ta', 'mr')")

class StepError(BaseModel):
    stage: str = Field(..., description="Pipeline stage where error occurred ('stt', 'retrieval', 'generation', 'guardrails')")
    message: str = Field(..., description="Error message description")
    recoverable: bool = Field(default=True, description="Whether pipeline recovered using fallbacks")

class LatencyBreakdown(BaseModel):
    stt_ms: float = Field(default=0.0, description="STT network API latency")
    faiss_ms: float = Field(default=0.0, description="FAISS dense vector search latency")
    bm25_ms: float = Field(default=0.0, description="BM25 lexical search latency")
    fusion_ms: float = Field(default=0.0, description="RRF rank fusion latency")
    retrieval_ms: float = Field(default=0.0, description="Total parallel retrieval latency")
    generation_ms: float = Field(default=0.0, description="LLM answer generation latency")
    guardrails_ms: float = Field(default=0.0, description="Groundedness & safety guardrail latency")
    post_stt_pipeline_ms: float = Field(default=0.0, description="Total post-STT wall-clock elapsed time (<200ms target)")
    total_end_to_end_ms: float = Field(default=0.0, description="Complete end-to-end latency (STT + pipeline)")
    target_met: bool = Field(default=True, description="Whether post-STT pipeline met the <200ms latency requirement")

class PipelineResponse(BaseModel):
    transcript: str = Field(default="", description="Transcribed query text from audio")
    language: str = Field(default="hi", description="Detected language code")
    query_used: str = Field(default="", description="Query string sent to retrieval")
    chunk_strategy: str = Field(default="semantic", description="Active chunking strategy")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved top-k chunks")
    answer: str = Field(default="", description="Final answer text presented to user")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="IDs of cited context chunks")
    guardrail_verdict: GuardrailVerdict = Field(default_factory=GuardrailVerdict, description="Guardrail evaluation results")
    latency_breakdown: LatencyBreakdown = Field(default_factory=LatencyBreakdown, description="Per-stage latency measurements")
    errors: List[StepError] = Field(default_factory=list, description="List of non-fatal/fatal step errors encountered")
    is_success: bool = Field(default=True, description="Overall pipeline execution status")
