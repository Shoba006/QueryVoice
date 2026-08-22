import re
import uuid
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    FIXED_OVERLAP = "fixed_overlap"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"
    RECURSIVE = "recursive"

class Chunk(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    language: str
    chunk_strategy: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for multilingual text (approx 4 chars per token or space-separated words)."""
    words = text.split()
    return max(len(words), len(text) // 4)

def split_indic_sentences(text: str) -> List[str]:
    """Splits text into sentences supporting Indic delimiters (| , ।, ?, !) as well as standard punctuation."""
    sentences = re.split(r'(?<=[।|\.!\?])\s+', text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned if cleaned else [text]

class DataChunker:
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model

    def chunk_fixed_size(
        self,
        text: str,
        doc_id: str,
        language: str,
        chunk_size_tokens: int = 128,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Fixed-size token chunking with NO overlap."""
        return self._fixed_chunk_impl(text, doc_id, language, chunk_size_tokens, 0, ChunkStrategy.FIXED_SIZE.value, metadata)

    def chunk_fixed_overlap(
        self,
        text: str,
        doc_id: str,
        language: str,
        chunk_size_tokens: int = 128,
        overlap_tokens: int = 40,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Fixed-size token chunking WITH overlap."""
        return self._fixed_chunk_impl(text, doc_id, language, chunk_size_tokens, overlap_tokens, ChunkStrategy.FIXED_OVERLAP.value, metadata)

    def _fixed_chunk_impl(
        self,
        text: str,
        doc_id: str,
        language: str,
        chunk_size_tokens: int,
        overlap_tokens: int,
        strategy_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        idx = 0

        step = max(1, chunk_size_tokens - overlap_tokens)

        while start < len(words):
            end = min(start + chunk_size_tokens, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            c_meta = dict(metadata or {})
            c_meta.update({"start_word_idx": start, "end_word_idx": end})

            chunk_obj = Chunk(
                chunk_id=f"{doc_id}_{strategy_name}_{idx}",
                text=chunk_text,
                doc_id=doc_id,
                language=language,
                chunk_strategy=strategy_name,
                token_count=estimate_tokens(chunk_text),
                metadata=c_meta
            )
            chunks.append(chunk_obj)
            idx += 1

            if end == len(words):
                break
            start += step

        return chunks

    def chunk_sentence(
        self,
        text: str,
        doc_id: str,
        language: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Sentence-based chunking: each sentence becomes a standalone chunk."""
        sentences = split_indic_sentences(text)
        chunks = []
        for idx, s in enumerate(sentences):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_sent_{idx}",
                text=s,
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.SENTENCE.value,
                token_count=estimate_tokens(s),
                metadata=dict(metadata or {}, sentence_index=idx)
            ))
        return chunks

    def chunk_semantic(
        self,
        text: str,
        doc_id: str,
        language: str,
        similarity_threshold: float = 0.5,
        max_tokens_per_chunk: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Semantic chunking: splits on sentence boundaries and merges adjacent sentences
        until cosine similarity drops below threshold.
        """
        sentences = split_indic_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Chunk(
                chunk_id=f"{doc_id}_sem_0",
                text=sentences[0],
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.SEMANTIC.value,
                token_count=estimate_tokens(sentences[0]),
                metadata=metadata or {}
            )]

        embeddings = None

        chunks = []
        current_sentences = [sentences[0]]
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sent_curr = sentences[i-1]
            sent_next = sentences[i]

            s1 = set(sent_curr.lower().split())
            s2 = set(sent_next.lower().split())
            sim = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))

            candidate_text = " ".join(current_sentences + [sent_next])
            token_count = estimate_tokens(candidate_text)

            if sim >= similarity_threshold and token_count <= max_tokens_per_chunk:
                current_sentences.append(sent_next)
            else:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_sem_{chunk_idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    language=language,
                    chunk_strategy=ChunkStrategy.SEMANTIC.value,
                    token_count=estimate_tokens(chunk_text),
                    metadata=dict(metadata or {}, sim_score=round(sim, 3))
                ))
                chunk_idx += 1
                current_sentences = [sent_next]

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_sem_{chunk_idx}",
                text=chunk_text,
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.SEMANTIC.value,
                token_count=estimate_tokens(chunk_text),
                metadata=metadata or {}
            ))

        return chunks

    def chunk_metadata_aware(
        self,
        text: str,
        doc_id: str,
        language: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Metadata-aware chunking: keeps short passages intact, enriches chunks with
        structural fields (doc_id, language, source query cluster, relevance).
        """
        meta = dict(metadata or {})
        meta.update({
            "doc_id": doc_id,
            "language": language,
            "char_length": len(text),
            "is_intact": len(text) <= 500
        })

        if len(text) <= 500:
            return [Chunk(
                chunk_id=f"{doc_id}_meta_0",
                text=text,
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.METADATA_AWARE.value,
                token_count=estimate_tokens(text),
                metadata=meta
            )]

        sentences = split_indic_sentences(text)
        chunks = []
        curr = []
        curr_len = 0
        c_idx = 0

        for s in sentences:
            if curr_len + len(s) > 400 and curr:
                chunk_text = " ".join(curr)
                c_meta = dict(meta)
                c_meta["chunk_index"] = c_idx
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_meta_{c_idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    language=language,
                    chunk_strategy=ChunkStrategy.METADATA_AWARE.value,
                    token_count=estimate_tokens(chunk_text),
                    metadata=c_meta
                ))
                c_idx += 1
                curr = [s]
                curr_len = len(s)
            else:
                curr.append(s)
                curr_len += len(s)

        if curr:
            chunk_text = " ".join(curr)
            c_meta = dict(meta)
            c_meta["chunk_index"] = c_idx
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_meta_{c_idx}",
                text=chunk_text,
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.METADATA_AWARE.value,
                token_count=estimate_tokens(chunk_text),
                metadata=c_meta
            ))

        return chunks

    def chunk_recursive(
        self,
        text: str,
        doc_id: str,
        language: str,
        max_tokens: int = 150,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Recursive/hierarchical chunking: splits by paragraphs -> sentences -> fixed token limits.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        raw_chunks = []
        for p in paragraphs:
            if estimate_tokens(p) <= max_tokens:
                raw_chunks.append(p)
            else:
                sents = split_indic_sentences(p)
                for s in sents:
                    if estimate_tokens(s) <= max_tokens:
                        raw_chunks.append(s)
                    else:
                        words = s.split()
                        for i in range(0, len(words), max_tokens):
                            raw_chunks.append(" ".join(words[i:i+max_tokens]))

        chunks = []
        for idx, c_text in enumerate(raw_chunks):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_rec_{idx}",
                text=c_text,
                doc_id=doc_id,
                language=language,
                chunk_strategy=ChunkStrategy.RECURSIVE.value,
                token_count=estimate_tokens(c_text),
                metadata=dict(metadata or {}, hierarchical_level="recursive")
            ))

        return chunks

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        language: str,
        strategy: Union[ChunkStrategy, str] = ChunkStrategy.SEMANTIC,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Main dispatcher for chunking strategies."""
        strat_val = strategy.value if isinstance(strategy, ChunkStrategy) else str(strategy)
        if strat_val == ChunkStrategy.FIXED_SIZE.value or strat_val == "fixed":
            return self.chunk_fixed_size(text, doc_id, language, metadata=metadata)
        elif strat_val == ChunkStrategy.FIXED_OVERLAP.value:
            return self.chunk_fixed_overlap(text, doc_id, language, metadata=metadata)
        elif strat_val == ChunkStrategy.SENTENCE.value:
            return self.chunk_sentence(text, doc_id, language, metadata=metadata)
        elif strat_val == ChunkStrategy.SEMANTIC.value:
            return self.chunk_semantic(text, doc_id, language, metadata=metadata)
        elif strat_val == ChunkStrategy.METADATA_AWARE.value:
            return self.chunk_metadata_aware(text, doc_id, language, metadata=metadata)
        elif strat_val == ChunkStrategy.RECURSIVE.value:
            return self.chunk_recursive(text, doc_id, language, metadata=metadata)
        else:
            return self.chunk_semantic(text, doc_id, language, metadata=metadata)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunker = DataChunker()
    sample_text = "भारत एक विशाल देश है। इसकी सांस्कृतिक विविधता विश्व प्रसिद्ध है।\n\nगोवा भारत का एक खूबसूरत तटीय राज्य है। यहाँ के समुद्र तट और व्यंजन पर्यटकों को आकर्षित करते हैं।"
    
    print("Testing Chunking Strategies:")
    for strat in ChunkStrategy:
        res = chunker.chunk_document(sample_text, "doc_101", "hi", strategy=strat)
        print(f"\n--- Strategy: {strat.value} ({len(res)} chunks) ---")
        for c in res:
            print(f"[{c.chunk_id}] Tokens: {c.token_count} | Text: {c.text}")
