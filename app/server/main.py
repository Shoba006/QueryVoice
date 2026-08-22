import os
import json
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.harness.orchestrator import RAGOrchestrator
from app.harness.schemas import PipelineRequest, PipelineResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice-Enabled RAG System (HH Goa 2026)",
    description="Multilingual Voice RAG operating over ai4bharat/MSMARCO-XI Indic dataset",
    version="1.0.0"
)

# Initialize Orchestrator singleton
orchestrator = RAGOrchestrator()

# Serve static web assets
WEB_DIR = Path("./app/web")
try:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RAG Server and building default index...")
    try:
        orchestrator.initialize_corpus_and_indices(strategy="semantic")
    except Exception as e:
        logger.error(f"Startup index initialization failed: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Voice-Enabled RAG System",
        "version": "1.0.0",
        "indices_ready": orchestrator.is_initialized
    }

@app.get("/benchmark")
def get_benchmark():
    """Returns the latest latency benchmark stats."""
    bench_file = Path("./app/eval/results/benchmark_summary.json")
    if not bench_file.exists():
        return {"status": "no_benchmark_run_yet", "message": "Run python app/eval/benchmark.py to generate stats."}
    
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/ask", response_model=PipelineResponse)
async def ask_endpoint(
    audio: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None),
    chunk_strategy: str = Form("semantic"),
    top_k: int = Form(5),
    language_filter: Optional[str] = Form(None)
):
    """
    Primary voice & text RAG endpoint.
    Accepts mic audio upload or text query, returns full PipelineResponse JSON.
    """
    audio_bytes = None
    filename = "audio.wav"

    if audio:
        audio_bytes = await audio.read()
        filename = audio.filename or "audio.wav"

    req = PipelineRequest(
        audio_bytes=audio_bytes,
        audio_filename=filename,
        text_query=text_query,
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        language_filter=language_filter
    )

    response = orchestrator.run_pipeline(req)
    return response

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the web frontend user interface."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h2>Frontend building... Please refresh.</h2>")
    
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
