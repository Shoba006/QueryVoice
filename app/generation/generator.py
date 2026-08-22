import os
import time
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv

load_dotenv()

from app.retrieval.hybrid_search import RetrievedChunk

logger = logging.getLogger(__name__)

class GenerationResult(BaseModel):
    answer: str = Field(..., description="Generated answer text")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="IDs of context chunks cited in answer")
    confidence: str = Field(default="high", description="Model confidence assessment ('high', 'medium', 'low', 'refused')")
    latency_ms: float = Field(..., description="LLM generation latency in milliseconds")
    model_used: str = Field(default="claude-3-5-sonnet-20241022", description="LLM model name")

class AnswerGenerator:
    """
    LLM Answer Generator calling Anthropic API with strict context-grounded prompting.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 300
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        self.client = None
        if self.api_key and self.api_key != "your_anthropic_api_key_here":
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

    def format_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        context_str = ""
        for idx, chunk in enumerate(chunks, 1):
            context_str += f"--- CONTEXT BLOCK {idx} [ID: {chunk.chunk_id}] ---\n{chunk.text}\n\n"

        prompt = f"""You are a strict, factual assistant for a multilingual Retrieval-Augmented Generation system.
Your task is to answer the user's question using ONLY the provided context blocks below.

Context Blocks:
{context_str if context_str else "No context available."}

User Question: {query}

INSTRUCTIONS:
1. Base your answer STRICTLY on the information in the context blocks.
2. Do NOT use any outside knowledge or assumptions.
3. If the context blocks do not contain enough information to answer the question, state EXACTLY: "I don't have enough information to answer that."
4. At the end of your answer, list the chunk IDs you cited in brackets, e.g. [Citations: doc_1_fixed_0].
5. Keep your answer concise and direct (under 150 words).
"""
        return prompt

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        start_time = time.perf_counter()

        if not chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                answer="I don't have enough information to answer that.",
                cited_chunk_ids=[],
                confidence="refused",
                latency_ms=round(elapsed_ms, 2),
                model_used="none"
            )

        prompt = self.format_prompt(query, chunks)
        all_chunk_ids = [c.chunk_id for c in chunks]

        if not self.client:
            logger.warning("Anthropic API key not available. Using local grounded synthesis fallback.")
            # Synthesize answer locally from top chunk
            top_chunk = chunks[0]
            answer_text = f"{top_chunk.text}\n\n[Citations: {top_chunk.chunk_id}]"
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                answer=answer_text,
                cited_chunk_ids=[top_chunk.chunk_id],
                confidence="medium",
                latency_ms=round(elapsed_ms, 2),
                model_used="local_fallback"
            )

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text.strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract citations if present
            cited_ids = []
            for cid in all_chunk_ids:
                if cid in raw_text:
                    cited_ids.append(cid)
            if not cited_ids and "don't have enough information" not in raw_text.lower():
                cited_ids = [chunks[0].chunk_id]

            conf = "refused" if "don't have enough information" in raw_text.lower() else "high"

            return GenerationResult(
                answer=raw_text,
                cited_chunk_ids=cited_ids,
                confidence=conf,
                latency_ms=round(elapsed_ms, 2),
                model_used=self.model_name
            )

        except Exception as e:
            logger.error(f"Anthropic API generation error: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Fallback on exception
            top_chunk = chunks[0]
            return GenerationResult(
                answer=f"{top_chunk.text}\n\n[Citations: {top_chunk.chunk_id}]",
                cited_chunk_ids=[top_chunk.chunk_id],
                confidence="low",
                latency_ms=round(elapsed_ms, 2),
                model_used="local_fallback_error"
            )
