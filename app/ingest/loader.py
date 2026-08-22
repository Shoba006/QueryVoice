import os
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DATASET_DIR = Path("./data/msmarco-xi")
LOCAL_PARQUET = DATASET_DIR / "passages.parquet"

# Supported Indic / multilingual splits in ai4bharat/MSMARCO-XI
DEFAULT_SPLITS = [
    ("validation/hinval.parquet", "hi"),
]

def ensure_dataset_loaded(target_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """
    Ensures local dataset exists at `./data/msmarco-xi/passages.parquet`.
    If absent, downloads representative splits from ai4bharat/MSMARCO-XI and saves locally.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    local_file = target_dir / "passages.parquet"

    if local_file.exists():
        logger.info(f"Loading existing corpus from {local_file}")
        return pd.read_parquet(local_file)

    logger.info("Dataset not found locally. Downloading from HuggingFace ai4bharat/MSMARCO-XI...")
    dfs = []
    
    for filename, lang_code in DEFAULT_SPLITS:
        try:
            downloaded_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=filename,
                repo_type="dataset"
            )
            df_split = pd.read_parquet(downloaded_path)
            # Sample max 2000 records for fast indexing
            if len(df_split) > 2000:
                df_split = df_split.head(2000)
            
            # Normalize column names if needed
            if "language" not in df_split.columns:
                df_split["language"] = lang_code
            
            dfs.append(df_split)
            logger.info(f"Loaded {len(df_split)} sampled records for language '{lang_code}' ({filename})")
        except Exception as e:
            logger.warning(f"Failed to download split {filename}: {e}")

    if not dfs:
        # Create synthetic corpus as ultimate fallback if network is completely down
        logger.warning("Network download failed. Creating synthetic Indic corpus for testing.")
        synthetic_data = [
            {
                "query_id": "q1",
                "query": "भारत की राजधानी क्या है?",
                "passage_id": "p1",
                "passage": "भारत की राजधानी नई दिल्ली है। नई दिल्ली भारत सरकार के तीनों अंगों का केंद्र है।",
                "language": "hi",
                "is_selected": 1
            },
            {
                "query_id": "q2",
                "query": "गोवा का प्रसिद्ध भोजन क्या है?",
                "passage_id": "p2",
                "passage": "गोवा का प्रसिद्ध भोजन फिश करी और चावल है। गोवा का तटीय व्यंजन बहुत ही स्वादिष्ट माना जाता है।",
                "language": "hi",
                "is_selected": 1
            },
            {
                "query_id": "q3",
                "query": "What is Retrieval Augmented Generation?",
                "passage_id": "p3",
                "passage": "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM responses by grounding the model on external knowledge bases.",
                "language": "en",
                "is_selected": 1
            }
        ]
        df_merged = pd.DataFrame(synthetic_data)
    else:
        df_merged = pd.concat(dfs, ignore_index=True)

    # Standardize column naming
    column_mapping = {
        "passages": "passage",
        "passage_text": "passage",
        "doc_id": "passage_id",
        "id": "passage_id"
    }
    df_merged = df_merged.rename(columns=column_mapping)

    # Clean missing fields
    if "passage_id" not in df_merged.columns:
        df_merged["passage_id"] = [f"doc_{i}" for i in range(len(df_merged))]
    if "passage" not in df_merged.columns and "text" in df_merged.columns:
        df_merged["passage"] = df_merged["text"]
    if "language" not in df_merged.columns:
        df_merged["language"] = "hi"

    # Save to disk for fast caching
    df_merged.to_parquet(local_file, index=False)
    logger.info(f"Saved dataset corpus with {len(df_merged)} passages to {local_file}")
    return df_merged

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = ensure_dataset_loaded()
    print("Corpus Loaded:")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(2))
