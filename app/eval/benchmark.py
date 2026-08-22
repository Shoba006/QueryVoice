import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from app.harness.orchestrator import RAGOrchestrator
from app.harness.schemas import PipelineRequest, PipelineResponse
from app.ingest.loader import ensure_dataset_loaded

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./app/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sample Indic / English query pool from MSMARCO-XI
BENCHMARK_QUERIES = [
    "भारत की राजधानी क्या है?",
    "गोवा का मुख्य भोजन क्या है?",
    "what is retrieval augmented generation?",
    "कॉर्पोरेशन की परिभाषा क्या है?",
    "भारत का राष्ट्रीय पक्षी कौन सा है?",
    "ताजमहल कहाँ स्थित है?",
    "who is the Prime Minister of India?",
    "what is artificial intelligence?",
    "भारतीय संविधान कब लागू हुआ?",
    "गोवा का निकटतम हवाई अड्डा कौन सा है?",
    "what is the population of Goa?",
    "हिन्दी दिवस कब मनाया जाता है?",
    "what is machine learning?",
    "भारत में कुल कितने राज्य हैं?",
    "गोवा में कौन सी भाषा बोली जाती है?",
    "what is vector database?",
    "FAISS क्या है?",
    "BM25 एल्गोरिदम कैसे काम करता है?",
    "what is speech to text?",
    "सर्वाम एआई क्या है?",
    "what is MSMARCO dataset?",
    "भारत की सबसे लंबी नदी कौन सी है?",
    "गोवा की राजधानी क्या है?",
    "what is LLM hallucination?",
    "गार्डरेल्स क्या हैं?",
    "what is cosine similarity?",
    "भारतीय अंतरिक्ष अनुसंधान संगठन कहाँ है?",
    "what is natural language processing?",
    "गोवा के प्रसिद्ध त्योहार कौन से हैं?",
    "what is deep learning?",
    "गणतंत्र दिवस कब मनाया जाता है?",
    "what is Python programming language?",
    "फास्टएपीआई क्या है?",
    "what is REST API?",
    "पायडैंटिक क्या है?",
    "what is hybrid search?",
    "Reciprocal Rank Fusion क्या है?",
    "what is semantic chunking?",
    "गोवा के समुद्र तट कौन से हैं?",
    "what is tokenization?",
    "अन्थ्रोपिक क्लॉड क्या है?",
    "what is sentence transformers?",
    "कन्नड़ भाषा कहाँ बोली जाती है?",
    "what is Indic language processing?",
    "मराठी साहित्य का इतिहास क्या है?",
    "what is speech recognition accuracy?",
    "तमिलनाडु की राजधानी क्या है?",
    "what is open source AI?",
    "बंगाल का प्रसिद्ध त्यौहार कौन सा है?",
    "what is latency benchmark?"
]

class LatencyBenchmark:
    def __init__(self):
        self.orchestrator = RAGOrchestrator()

    def run_benchmark(self, num_queries: int = 50, strategy: str = "semantic") -> Dict[str, Any]:
        """
        Executes benchmark over representative test queries and calculates latency stats.
        """
        logger.info(f"Starting Latency Benchmark over {num_queries} queries (Strategy: {strategy})...")
        self.orchestrator.initialize_corpus_and_indices(strategy=strategy)

        # Sample or loop queries
        queries = BENCHMARK_QUERIES
        while len(queries) < num_queries:
            queries = queries + BENCHMARK_QUERIES
        queries = queries[:num_queries]

        results: List[Dict[str, Any]] = []

        for idx, q in enumerate(queries, 1):
            req = PipelineRequest(text_query=q, chunk_strategy=strategy)
            t0 = time.perf_counter()
            resp: PipelineResponse = self.orchestrator.run_pipeline(req)
            t_total = (time.perf_counter() - t0) * 1000.0

            b = resp.latency_breakdown
            results.append({
                "query_id": idx,
                "query": q,
                "stt_ms": b.stt_ms,
                "faiss_ms": b.faiss_ms,
                "bm25_ms": b.bm25_ms,
                "fusion_ms": b.fusion_ms,
                "retrieval_ms": b.retrieval_ms,
                "generation_ms": b.generation_ms,
                "guardrails_ms": b.guardrails_ms,
                "post_stt_pipeline_ms": b.post_stt_pipeline_ms,
                "total_end_to_end_ms": b.total_end_to_end_ms or t_total
            })

            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{num_queries} benchmark queries.")

        df_bench = pd.DataFrame(results)

        # Compute Percentiles P50, P70, P100 (max)
        metrics = ["stt_ms", "faiss_ms", "bm25_ms", "fusion_ms", "retrieval_ms", "generation_ms", "guardrails_ms", "post_stt_pipeline_ms", "total_end_to_end_ms"]
        stats: Dict[str, Dict[str, float]] = {}

        for m in metrics:
            vals = df_bench[m].values if m in df_bench.columns else np.zeros(len(df_bench))
            stats[m] = {
                "P50": round(float(np.percentile(vals, 50)), 2),
                "P70": round(float(np.percentile(vals, 70)), 2),
                "P100": round(float(np.max(vals)), 2),
                "Mean": round(float(np.mean(vals)), 2),
                "Min": round(float(np.min(vals)), 2)
            }

        pass_target = bool(stats["post_stt_pipeline_ms"]["P50"] < 200.0 and stats["post_stt_pipeline_ms"]["P70"] < 200.0)

        # Save results JSON
        summary = {
            "num_queries": num_queries,
            "chunk_strategy": strategy,
            "target_200ms_requirement": "<200 ms post-STT pipeline",
            "target_met": pass_target,
            "status": "PASS" if pass_target else "FAIL",
            "latency_stats": stats,
            "raw_runs": results
        }
        
        json_path = RESULTS_DIR / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.generate_chart(stats, pass_target)
        self.generate_markdown_report(stats, pass_target)

        logger.info(f"Benchmark completed! Target Status: {'PASS' if pass_target else 'FAIL'}. Stats saved to {RESULTS_DIR}")
        return summary

    def generate_chart(self, stats: Dict[str, Dict[str, float]], pass_target: bool):
        """Generates latency benchmark chart comparing stage percentiles."""
        stages = ["FAISS Dense", "BM25 Sparse", "RRF Fusion", "Retrieval Total", "Generation", "Guardrails", "Post-STT Total"]
        p50 = [stats["faiss_ms"]["P50"], stats["bm25_ms"]["P50"], stats["fusion_ms"]["P50"], stats["retrieval_ms"]["P50"], stats["generation_ms"]["P50"], stats["guardrails_ms"]["P50"], stats["post_stt_pipeline_ms"]["P50"]]
        p70 = [stats["faiss_ms"]["P70"], stats["bm25_ms"]["P70"], stats["fusion_ms"]["P70"], stats["retrieval_ms"]["P70"], stats["generation_ms"]["P70"], stats["guardrails_ms"]["P70"], stats["post_stt_pipeline_ms"]["P70"]]
        p100 = [stats["faiss_ms"]["P100"], stats["bm25_ms"]["P100"], stats["fusion_ms"]["P100"], stats["retrieval_ms"]["P100"], stats["generation_ms"]["P100"], stats["guardrails_ms"]["P100"], stats["post_stt_pipeline_ms"]["P100"]]

        x = np.arange(len(stages))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, p50, width, label='P50 (Median)', color='#2563eb')
        ax.bar(x, p70, width, label='P70', color='#7c3aed')
        ax.bar(x + width, p100, width, label='P100 (Max)', color='#db2777')

        ax.set_ylabel('Latency (ms)')
        status_str = "PASS (<200ms Target Met)" if pass_target else "FAIL (>200ms)"
        ax.set_title(f'Voice RAG System Latency Benchmark — Status: {status_str}')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=15)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Target threshold line at 200ms for Post-STT pipeline
        ax.axhline(y=200, color='r', linestyle='--', label='Sub-200ms Pipeline Target')

        plt.tight_layout()
        chart_path = RESULTS_DIR / "latency_chart.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        logger.info(f"Chart saved to {chart_path}")

    def generate_markdown_report(self, stats: Dict[str, Dict[str, float]], pass_target: bool):
        """Generates markdown table summary."""
        status_badge = "✅ **PASS**" if pass_target else "❌ **FAIL**"
        md_text = f"""# Latency Benchmark Summary — Target Verdict: {status_badge}

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 / Max (ms) | Mean (ms) | Min (ms) |
|---|---|---|---|---|---|
| **FAISS Dense Search** | {stats['faiss_ms']['P50']} | {stats['faiss_ms']['P70']} | {stats['faiss_ms']['P100']} | {stats['faiss_ms']['Mean']} | {stats['faiss_ms']['Min']} |
| **BM25 Lexical Search** | {stats['bm25_ms']['P50']} | {stats['bm25_ms']['P70']} | {stats['bm25_ms']['P100']} | {stats['bm25_ms']['Mean']} | {stats['bm25_ms']['Min']} |
| **RRF Rank Fusion** | {stats['fusion_ms']['P50']} | {stats['fusion_ms']['P70']} | {stats['fusion_ms']['P100']} | {stats['fusion_ms']['Mean']} | {stats['fusion_ms']['Min']} |
| **Total Parallel Retrieval** | {stats['retrieval_ms']['P50']} | {stats['retrieval_ms']['P70']} | {stats['retrieval_ms']['P100']} | {stats['retrieval_ms']['Mean']} | {stats['retrieval_ms']['Min']} |
| **Answer Generation (LLM)** | {stats['generation_ms']['P50']} | {stats['generation_ms']['P70']} | {stats['generation_ms']['P100']} | {stats['generation_ms']['Mean']} | {stats['generation_ms']['Min']} |
| **Guardrails & Safety** | {stats['guardrails_ms']['P50']} | {stats['guardrails_ms']['P70']} | {stats['guardrails_ms']['P100']} | {stats['guardrails_ms']['Mean']} | {stats['guardrails_ms']['Min']} |
| **Post-STT Pipeline Total** | **{stats['post_stt_pipeline_ms']['P50']}** | **{stats['post_stt_pipeline_ms']['P70']}** | **{stats['post_stt_pipeline_ms']['P100']}** | **{stats['post_stt_pipeline_ms']['Mean']}** | **{stats['post_stt_pipeline_ms']['Min']}** |
| **STT Network API (Sarvam)** | {stats['stt_ms']['P50']} | {stats['stt_ms']['P70']} | {stats['stt_ms']['P100']} | {stats['stt_ms']['Mean']} | {stats['stt_ms']['Min']} |
| **Total End-to-End** | {stats['total_end_to_end_ms']['P50']} | {stats['total_end_to_end_ms']['P70']} | {stats['total_end_to_end_ms']['P100']} | {stats['total_end_to_end_ms']['Mean']} | {stats['total_end_to_end_ms']['Min']} |

> [!NOTE]
> **Sub-200ms Target Verification**: Target Status = {status_badge}. Post-STT pipeline latency (retrieval + generation + guardrails) is measured via high-resolution monotonic wall-clock timing.
"""

        report_path = RESULTS_DIR / "benchmark_table.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info(f"Markdown table report saved to {report_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bm = LatencyBenchmark()
    bm.run_benchmark(num_queries=50)
