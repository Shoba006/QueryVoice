import time
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Union

from app.stt.sarvam_client import SarvamSTTClient, TranscriptionResult
from app.ingest.loader import ensure_dataset_loaded
from app.ingest.chunker import DataChunker, ChunkStrategy, Chunk
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25_store import BM25IndexStore
from app.retrieval.hybrid_search import HybridRetriever, RetrievalResult
from app.generation.generator import AnswerGenerator, GenerationResult
from app.guardrails.safety import GuardrailSystem, GuardrailVerdict
from app.harness.schemas import PipelineRequest, PipelineResponse, StepError, LatencyBreakdown

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    Central pipeline orchestrator wiring STT, retrieval, generation, and guardrails together.
    Pre-builds and caches indices across all chunking strategies for zero-latency strategy switching.
    Encodes query vectors ONCE and reuses them across vector search and guardrails to achieve <200ms target.
    """
    def __init__(self, dataset_path: str = "./data/msmarco-xi"):
        logger.info("Initializing RAG Orchestrator system...")
        self.dataset_path = dataset_path
        
        # 1. Initialize core clients and components
        self.stt_client = SarvamSTTClient()
        self.embedder_store = FAISSVectorStore()  # Shared model instance
        self.generator = AnswerGenerator()
        self.guardrails = GuardrailSystem(embedder=self.embedder_store.embedder)
        self.chunker = DataChunker(embedding_model=self.embedder_store.embedder)

        # Dictionary of strategy -> (FAISSVectorStore, BM25IndexStore, HybridRetriever)
        self.strategy_stores: Dict[str, tuple[FAISSVectorStore, BM25IndexStore, HybridRetriever]] = {}
        self.is_initialized = False

    def initialize_corpus_and_indices(self, strategies: Optional[Union[List[str], str]] = None, strategy: Optional[str] = None, force_rebuild: bool = False):
        """Loads corpus and pre-builds/loads indices for all requested chunking strategies."""
        if self.is_initialized and not force_rebuild:
            return

        if strategy:
            target_strategies = [strategy]
        elif isinstance(strategies, str):
            target_strategies = [strategies]
        elif isinstance(strategies, list):
            target_strategies = strategies
        else:
            target_strategies = ["fixed_size", "fixed_overlap", "sentence", "semantic", "metadata_aware"]
        logger.info("Loading MSMARCO-XI corpus...")
        df_corpus = ensure_dataset_loaded()

        for strat in target_strategies:
            logger.info(f"Setting up index for strategy '{strat}'...")
            v_store = FAISSVectorStore(index_dir=self.embedder_store.index_dir, embedder=self.embedder_store.embedder)
            b_store = BM25IndexStore(index_dir=self.embedder_store.index_dir)

            # Try loading cached indices from disk
            if not force_rebuild and v_store.load(strat) and b_store.load(strat):
                logger.info(f"Loaded indices for strategy '{strat}' from disk cache.")
            else:
                logger.info(f"Building chunks and indices for strategy '{strat}'...")
                all_chunks: List[Chunk] = []

                for idx, row in df_corpus.iterrows():
                    passage_text = str(row.get("passage", "")).strip()
                    if not passage_text or passage_text.lower() == "none":
                        continue
                    
                    doc_id = str(row.get("passage_id", f"doc_{idx}"))
                    lang = str(row.get("language", "hi"))
                    
                    meta = {
                        "query_id": str(row.get("query_id", "")),
                        "source_query": str(row.get("query", "")),
                        "is_selected": int(row.get("is_selected", 0)) if "is_selected" in row else 1
                    }

                    chunks = self.chunker.chunk_document(passage_text, doc_id, lang, strategy=strat, metadata=meta)
                    all_chunks.extend(chunks)

                logger.info(f"Generated {len(all_chunks)} chunks for strategy '{strat}'.")
                if len(all_chunks) > 5000:
                    all_chunks = all_chunks[:5000]

                v_store.build_index(all_chunks)
                v_store.save(strat)

                b_store.build_index(all_chunks)
                b_store.save(strat)

            h_retriever = HybridRetriever(v_store, b_store)
            self.strategy_stores[strat] = (v_store, b_store, h_retriever)

        self.is_initialized = True
        logger.info(f"RAG Orchestrator initialization complete for strategies: {list(self.strategy_stores.keys())}")

    def run_pipeline(self, request: PipelineRequest) -> PipelineResponse:
        """
        Executes the end-to-end Voice-Enabled RAG pipeline:
        Audio -> STT -> Pre-guardrail -> Hybrid Retrieval -> Generation -> Post-guardrail -> Response
        """
        pipeline_start = time.perf_counter()
        errors: List[StepError] = []

        active_strategy = request.chunk_strategy if request.chunk_strategy in self.strategy_stores else "semantic"

        # Ensure active strategy index is loaded
        if not self.is_initialized or active_strategy not in self.strategy_stores:
            try:
                self.initialize_corpus_and_indices(strategies=[active_strategy])
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                errors.append(StepError(stage="ingest", message=f"Index init failed: {e}", recoverable=True))

        v_store, b_store, retriever = self.strategy_stores[active_strategy]

        # --- Stage 1: Speech-To-Text (or text input bypass) ---
        stt_start = time.perf_counter()
        query_text = ""
        detected_lang = request.language_filter or "hi"
        stt_latency = 0.0

        if request.audio_bytes:
            stt_res: TranscriptionResult = self.stt_client.transcribe(
                audio_bytes=request.audio_bytes,
                filename=request.audio_filename,
                language_code=request.language_filter
            )
            stt_latency = stt_res.raw_latency_ms
            query_text = stt_res.text
            detected_lang = stt_res.language

            if not stt_res.is_success:
                errors.append(StepError(stage="stt", message=stt_res.error or "STT failed", recoverable=True))
                if not query_text:
                    query_text = "भारत की राजधानी क्या है?"
        elif request.text_query:
            query_text = request.text_query.strip()
            stt_latency = (time.perf_counter() - stt_start) * 1000.0
        else:
            query_text = "भारत की राजधानी क्या है?"
            stt_latency = (time.perf_counter() - stt_start) * 1000.0

        post_stt_start = time.perf_counter()

        # --- Stage 2: Encode Query Embedding Vector ONCE ---
        t_enc_start = time.perf_counter()
        query_emb_list = v_store.embedder.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb_list, dtype=np.float32)
        enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

        # --- Stage 3: Input Safety Guardrail Check ---
        g_start = time.perf_counter()
        is_safe = self.guardrails.check_input_safety(query_text)
        g_ms = (time.perf_counter() - g_start) * 1000.0

        if not is_safe:
            total_post_stt = (time.perf_counter() - post_stt_start) * 1000.0
            total_e2e = (time.perf_counter() - pipeline_start) * 1000.0

            verdict = GuardrailVerdict(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="Query flagged by input safety filter.",
                guardrail_case="Case 3: Unsafe Query Refusal"
            )
            return PipelineResponse(
                transcript=query_text,
                language=detected_lang,
                query_used=query_text,
                chunk_strategy=active_strategy,
                retrieved_chunks=[],
                answer="I don't have enough information to answer that.",
                cited_chunk_ids=[],
                guardrail_verdict=verdict,
                latency_breakdown=LatencyBreakdown(
                    stt_ms=round(stt_latency, 2),
                    guardrails_ms=round(g_ms, 2),
                    post_stt_pipeline_ms=round(total_post_stt, 2),
                    total_end_to_end_ms=round(total_e2e, 2),
                    target_met=total_post_stt < 200.0
                ),
                errors=errors,
                is_success=False
            )

        # --- Stage 4: Parallel Hybrid Retrieval ---
        retrieval_res: RetrievalResult = retriever.search(
            query=query_text,
            top_k=request.top_k,
            language_filter=request.language_filter,
            strategy_name=active_strategy,
            precomputed_embedding=query_emb
        )

        # --- Stage 5: Off-topic / Domain Scope Check ---
        off_start = time.perf_counter()
        is_on_topic, off_score = self.guardrails.check_off_topic(query_text, retrieval_res.chunks)
        off_ms = (time.perf_counter() - off_start) * 1000.0

        if not is_on_topic:
            total_post_stt = (time.perf_counter() - post_stt_start) * 1000.0
            total_e2e = (time.perf_counter() - pipeline_start) * 1000.0

            verdict = GuardrailVerdict(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                should_abstain=True,
                rejection_reason="Query is out of scope / off-topic for corpus.",
                guardrail_case="Case 2: Off-Topic Refusal",
                off_topic_score=off_score
            )
            return PipelineResponse(
                transcript=query_text,
                language=detected_lang,
                query_used=query_text,
                chunk_strategy=active_strategy,
                retrieved_chunks=retrieval_res.chunks,
                answer="I don't have enough information to answer that.",
                cited_chunk_ids=[],
                guardrail_verdict=verdict,
                latency_breakdown=LatencyBreakdown(
                    stt_ms=round(stt_latency, 2),
                    faiss_ms=retrieval_res.faiss_ms,
                    bm25_ms=retrieval_res.bm25_ms,
                    fusion_ms=retrieval_res.fusion_ms,
                    retrieval_ms=retrieval_res.latency_ms,
                    guardrails_ms=round(g_ms + off_ms, 2),
                    post_stt_pipeline_ms=round(total_post_stt, 2),
                    total_end_to_end_ms=round(total_e2e, 2),
                    target_met=total_post_stt < 200.0
                ),
                errors=errors,
                is_success=True
            )

        # --- Stage 6: Answer Generation ---
        gen_res: Optional[GenerationResult] = None
        try:
            gen_res = self.generator.generate(query_text, retrieval_res.chunks)
        except Exception as e:
            logger.error(f"Generation stage failure: {e}")
            errors.append(StepError(stage="generation", message=str(e), recoverable=True))

        gen_latency = gen_res.latency_ms if gen_res else 0.0

        # --- Stage 7: Groundedness Evaluation ---
        post_g_start = time.perf_counter()
        verdict = self.guardrails.evaluate(query_text, retrieval_res.chunks, gen_res)
        post_g_ms = (time.perf_counter() - post_g_start) * 1000.0

        final_answer = gen_res.answer if gen_res else "I don't have enough information to answer that."
        if verdict.should_abstain:
            final_answer = "I don't have enough information to answer that."

        cited_ids = gen_res.cited_chunk_ids if gen_res and not verdict.should_abstain else []

        total_post_stt = (time.perf_counter() - post_stt_start) * 1000.0
        total_e2e = (time.perf_counter() - pipeline_start) * 1000.0

        return PipelineResponse(
            transcript=query_text,
            language=detected_lang,
            query_used=query_text,
            chunk_strategy=active_strategy,
            retrieved_chunks=retrieval_res.chunks,
            answer=final_answer,
            cited_chunk_ids=cited_ids,
            guardrail_verdict=verdict,
            latency_breakdown=LatencyBreakdown(
                stt_ms=round(stt_latency, 2),
                faiss_ms=retrieval_res.faiss_ms,
                bm25_ms=retrieval_res.bm25_ms,
                fusion_ms=retrieval_res.fusion_ms,
                retrieval_ms=retrieval_res.latency_ms,
                generation_ms=round(gen_latency, 2),
                guardrails_ms=round(g_ms + off_ms + post_g_ms, 2),
                post_stt_pipeline_ms=round(total_post_stt, 2),
                total_end_to_end_ms=round(total_e2e, 2),
                target_met=total_post_stt < 200.0
            ),
            errors=errors,
            is_success=True
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = RAGOrchestrator()
    req = PipelineRequest(text_query="भारत की राजधानी क्या है?", chunk_strategy="semantic")
    resp = orchestrator.run_pipeline(req)
    print("\n--- Pipeline Execution Output ---")
    print(resp.model_dump_json(indent=2))
