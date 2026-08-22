import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer

from app.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_CACHE_DIR = Path("./data/indices")

class FAISSVectorStore:
    """
    In-memory FAISS vector index with L2 normalized inner-product search (cosine similarity).
    Supports serialization to disk for fast server startup.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_dir: Path = INDEX_CACHE_DIR,
        embedder: Optional[Any] = None
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if embedder is not None:
            self.embedder = embedder
        else:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)

        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []

    def build_index(self, chunks: List[Chunk]) -> int:
        """Encodes chunk texts and builds the FAISS index."""
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return 0

        self.chunks = chunks
        texts = [c.text for c in chunks]
        
        logger.info(f"Encoding {len(texts)} chunks with {self.model_name}...")
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)

        # Store embeddings in Chunk objects for downstream guardrail groundedness check
        for idx, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings_np[idx].tolist()

        logger.info(f"Successfully indexed {self.index.ntotal} vectors into FAISS index.")
        return self.index.ntotal

    def save(self, name_prefix: str = "default") -> None:
        """Saves FAISS index and chunk metadata to disk."""
        if self.index is None or not self.chunks:
            logger.warning("Index empty. Nothing to save.")
            return

        faiss_path = self.index_dir / f"faiss_{name_prefix}.index"
        chunks_path = self.index_dir / f"chunks_{name_prefix}.pkl"

        faiss.write_index(self.index, str(faiss_path))
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved FAISS index to {faiss_path} and metadata to {chunks_path}")

    def load(self, name_prefix: str = "default") -> bool:
        """Loads FAISS index and chunk metadata from disk cache if present."""
        faiss_path = self.index_dir / f"faiss_{name_prefix}.index"
        chunks_path = self.index_dir / f"chunks_{name_prefix}.pkl"

        if not faiss_path.exists() or not chunks_path.exists():
            logger.info("Index cache not found on disk.")
            return False

        try:
            self.index = faiss.read_index(str(faiss_path))
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Successfully loaded cached FAISS index ({self.index.ntotal} vectors) from {faiss_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index cache: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Dense vector search for query with optional language filtering.
        """
        if self.index is None or not self.chunks:
            logger.warning("FAISS index is not built or empty.")
            return []

        if precomputed_embedding is not None:
            query_emb_np = precomputed_embedding
            if query_emb_np.ndim == 1:
                query_emb_np = np.expand_dims(query_emb_np, axis=0)
            query_emb_np = query_emb_np.astype(np.float32)
        else:
            query_emb = self.embedder.encode([query], normalize_embeddings=True)
            query_emb_np = np.array(query_emb, dtype=np.float32)

        # Retrieve extra candidates if filtering by language
        fetch_k = top_k * 3 if language_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_emb_np, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            if language_filter and chunk.language != language_filter:
                continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
