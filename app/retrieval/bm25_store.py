import pickle
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi

from app.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_CACHE_DIR = Path("./data/indices")

def tokenize_indic(text: str) -> List[str]:
    """Tokenizes Indic/multilingual text into lowercase word tokens."""
    return [w.strip().lower() for w in text.split() if w.strip()]

class BM25IndexStore:
    """
    BM25 keyword search index wrapping rank_bm25 with disk serialization.
    """
    def __init__(self, index_dir: Path = INDEX_CACHE_DIR):
        self.index_dir = Path(index_dir)
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.index_dir = Path("/tmp/indices")
            try:
                self.index_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []

    def build_index(self, chunks: List[Chunk]) -> int:
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return 0

        self.chunks = chunks
        corpus_tokens = [tokenize_indic(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Successfully built BM25 index over {len(chunks)} chunks.")
        return len(chunks)

    def save(self, name_prefix: str = "default") -> None:
        if self.bm25 is None or not self.chunks:
            return

        bm25_path = self.index_dir / f"bm25_{name_prefix}.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"Saved BM25 index to {bm25_path}")

    def load(self, name_prefix: str = "default") -> bool:
        bm25_path = self.index_dir / f"bm25_{name_prefix}.pkl"
        if not bm25_path.exists():
            return False

        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
            logger.info(f"Successfully loaded cached BM25 index ({len(self.chunks)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 cache: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        if self.bm25 is None or not self.chunks:
            return []

        tokens = tokenize_indic(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
            chunk = self.chunks[idx]
            if language_filter and chunk.language != language_filter:
                continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results
