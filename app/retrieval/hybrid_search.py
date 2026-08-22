import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

from app.ingest.chunker import Chunk
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25_store import BM25IndexStore

logger = logging.getLogger(__name__)

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    language: str
    chunk_strategy: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    strategy: str
    faiss_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    latency_ms: float = 0.0

class HybridRetriever:
    """
    Hybrid retriever combining FAISS dense vector search and BM25 sparse keyword search
    using Reciprocal Rank Fusion (RRF). Executes searches concurrently.
    """
    def __init__(self, vector_store: FAISSVectorStore, bm25_store: BM25IndexStore, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k
        self._executor = ThreadPoolExecutor(max_workers=2)

    def search(
        self,
        query: str,
        top_k: int = 5,
        language_filter: Optional[str] = None,
        strategy_name: str = "hybrid",
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> RetrievalResult:
        start_total = time.perf_counter()

        def _do_dense():
            t0 = time.perf_counter()
            res = self.vector_store.search(
                query,
                top_k=top_k * 4,
                language_filter=language_filter,
                precomputed_embedding=precomputed_embedding
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return res, dt

        def _do_sparse():
            t0 = time.perf_counter()
            res = self.bm25_store.search(
                query,
                top_k=top_k * 4,
                language_filter=language_filter
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return res, dt

        # Run FAISS dense search and BM25 sparse search concurrently
        fut_dense = self._executor.submit(_do_dense)
        fut_sparse = self._executor.submit(_do_sparse)

        dense_results, faiss_ms = fut_dense.result()
        sparse_results, bm25_ms = fut_sparse.result()

        # Reciprocal Rank Fusion & Deduplication
        t_fusion_start = time.perf_counter()
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        dense_scores_map: Dict[str, float] = {}
        bm25_scores_map: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (chunk, score) in enumerate(sparse_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            bm25_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort chunks by final RRF score and pick top-k unique chunks
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        retrieved_chunks = []
        for cid in sorted_cids:
            chunk = chunk_map[cid]
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_id=chunk.doc_id,
                language=chunk.language,
                chunk_strategy=chunk.chunk_strategy,
                dense_score=dense_scores_map.get(cid),
                bm25_score=bm25_scores_map.get(cid),
                rrf_score=round(rrf_scores[cid], 5),
                metadata=chunk.metadata,
                embedding=chunk.embedding
            ))

        fusion_ms = (time.perf_counter() - t_fusion_start) * 1000.0
        total_retrieval_ms = (time.perf_counter() - start_total) * 1000.0

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            strategy=strategy_name,
            faiss_ms=round(faiss_ms, 2),
            bm25_ms=round(bm25_ms, 2),
            fusion_ms=round(fusion_ms, 2),
            latency_ms=round(total_retrieval_ms, 2)
        )
