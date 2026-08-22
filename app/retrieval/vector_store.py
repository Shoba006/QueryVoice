import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel

from app.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_CACHE_DIR = Path("./data/indices")

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS_ST = True
except Exception as e:
    logger.warning(f"FAISS/SentenceTransformers not available ({e}). Using lightweight numpy fallback.")
    HAS_FAISS_ST = False
    faiss = None
    SentenceTransformer = None


class LightweightEmbedder:
    """Lightweight fallback embedder using term hashing and numpy when PyTorch is omitted."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        sentences: Any,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        vecs = []
        for text in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = abs(hash(w)) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


class FAISSVectorStore:
    """
    In-memory FAISS / numpy vector index with inner-product search (cosine similarity).
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
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.index_dir = Path("/tmp/indices")
            try:
                self.index_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        if embedder is not None:
            self.embedder = embedder
        elif HAS_FAISS_ST:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.embedder = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using lightweight embedder.")
                self.embedder = LightweightEmbedder()
        else:
            logger.info("Using lightweight numpy vector embedder.")
            self.embedder = LightweightEmbedder()

        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index: Optional[Any] = None
        self.chunks: List[Chunk] = []
        self.embeddings_np: Optional[np.ndarray] = None

    def build_index(self, chunks: List[Chunk]) -> int:
        """Encodes chunk texts and builds the vector index."""
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return 0

        self.chunks = chunks
        texts = [c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings_np = np.array(embeddings, dtype=np.float32)

        if HAS_FAISS_ST and faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings_np)
            total = self.index.ntotal
        else:
            total = len(self.chunks)

        for idx, chunk in enumerate(self.chunks):
            chunk.embedding = self.embeddings_np[idx].tolist()

        logger.info(f"Successfully indexed {total} vectors into store.")
        return total

    def save(self, name_prefix: str = "default") -> None:
        """Saves vector index and chunk metadata to disk."""
        if not self.chunks:
            return

        chunks_path = self.index_dir / f"chunks_{name_prefix}.pkl"
        try:
            if HAS_FAISS_ST and faiss is not None and self.index is not None:
                faiss_path = self.index_dir / f"faiss_{name_prefix}.index"
                faiss.write_index(self.index, str(faiss_path))
            with open(chunks_path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings_np}, f)
        except Exception as e:
            logger.warning(f"Could not save index cache: {e}")

    def load(self, name_prefix: str = "default") -> bool:
        """Loads vector index and chunk metadata from disk cache if present."""
        chunks_path = self.index_dir / f"chunks_{name_prefix}.pkl"

        if not chunks_path.exists():
            return False

        try:
            if HAS_FAISS_ST and faiss is not None:
                faiss_path = self.index_dir / f"faiss_{name_prefix}.index"
                if faiss_path.exists():
                    self.index = faiss.read_index(str(faiss_path))
            with open(chunks_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.chunks = data["chunks"]
                    self.embeddings_np = data.get("embeddings")
                else:
                    self.chunks = data
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
        if not self.chunks:
            return []

        if precomputed_embedding is not None:
            query_emb_np = precomputed_embedding
            if query_emb_np.ndim == 1:
                query_emb_np = np.expand_dims(query_emb_np, axis=0)
            query_emb_np = query_emb_np.astype(np.float32)
        else:
            query_emb = self.embedder.encode([query], normalize_embeddings=True)
            query_emb_np = np.array(query_emb, dtype=np.float32)

        fetch_k = top_k * 3 if language_filter else top_k
        fetch_k = min(fetch_k, len(self.chunks))

        if HAS_FAISS_ST and faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_emb_np, fetch_k)
            iter_pairs = zip(scores[0], indices[0])
        else:
            if self.embeddings_np is None:
                texts = [c.text for c in self.chunks]
                self.embeddings_np = self.embedder.encode(texts, normalize_embeddings=True)
            sims = np.dot(self.embeddings_np, query_emb_np.flatten())
            top_indices = np.argsort(sims)[::-1][:fetch_k]
            iter_pairs = [(sims[i], i) for i in top_indices]

        results = []
        for score, idx in iter_pairs:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            if language_filter and chunk.language != language_filter:
                continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
