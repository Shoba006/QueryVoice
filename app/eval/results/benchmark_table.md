# Latency Benchmark Summary — Target Verdict: ❌ **FAIL**

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 / Max (ms) | Mean (ms) | Min (ms) |
|---|---|---|---|---|---|
| **FAISS Dense Search** | 1.5 | 1.59 | 8.98 | 1.93 | 0.79 |
| **BM25 Lexical Search** | 4.76 | 5.56 | 46.47 | 5.79 | 2.47 |
| **RRF Rank Fusion** | 0.24 | 0.3 | 1.93 | 0.34 | 0.13 |
| **Total Parallel Retrieval** | 5.21 | 6.35 | 48.65 | 6.41 | 2.68 |
| **Answer Generation (LLM)** | 440.02 | 584.33 | 6291.98 | 827.28 | 345.36 |
| **Guardrails & Safety** | 0.17 | 0.22 | 1.39 | 0.23 | 0.05 |
| **Post-STT Pipeline Total** | **475.86** | **619.27** | **6342.2** | **884.85** | **372.52** |
| **STT Network API (Sarvam)** | 0.0 | 0.0 | 0.01 | 0.0 | 0.0 |
| **Total End-to-End** | 475.87 | 619.28 | 6342.21 | 884.86 | 372.52 |

> [!NOTE]
> **Sub-200ms Target Verification**: Target Status = ❌ **FAIL**. Post-STT pipeline latency (retrieval + generation + guardrails) is measured via high-resolution monotonic wall-clock timing.
