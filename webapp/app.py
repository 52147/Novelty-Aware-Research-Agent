"""
app.py — FastAPI backend for the Novelty-Aware Research Agent

Run:
    uvicorn app:app --reload --port 8000

Then open:
    http://localhost:8000
"""

import json
import os
import sys
import time
import threading
import queue
from typing import Optional, Generator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Add project root to path so pipeline modules are importable ───────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Novelty-Aware Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Request models ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_n: int = 5
    api_key: str = ""

# ── Helper: run pipeline in thread, emit SSE events via queue ─────────────────
def run_pipeline_threaded(query: str, top_n: int, event_queue: queue.Queue, api_key: str = ""):
    """
    Runs the full pipeline in a background thread.
    Pushes SSE-formatted strings into event_queue.
    """
    def emit(event_type: str, data: dict):
        payload = json.dumps({"type": event_type, **data})
        event_queue.put(f"data: {payload}\n\n")

    try:
        # ── Import here so we don't fail at startup if API key is missing ──
        from llm_client       import LLMClient
        from corpus_builder   import CorpusBuilder
        from query_analyzer   import QueryAnalyzer
        from retriever        import Retriever
        from ranker           import Ranker
        from extractor        import ContributionExtractor
        from comparison_agent import ComparisonAgent
        from answer_generator import AnswerGenerator

        t0 = time.time()

        # ── Init ──────────────────────────────────────────────────────────
        emit("init", {"message": "Initializing pipeline…"})

        # Use key from request, then env var
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            emit("error", {"message": "No API key provided. Enter your OpenAI API key in the sidebar."})
            event_queue.put(None)
            return
        llm = LLMClient(api_key=key)
        corpus = CorpusBuilder()
        corpus.load_index(CORPUS_PATH)

        paper_texts_path = os.path.join(CORPUS_PATH, "paper_texts.json")
        paper_texts = {}
        if os.path.exists(paper_texts_path):
            with open(paper_texts_path) as f:
                paper_texts = json.load(f)

        emit("init", {"message": f"Corpus loaded: {corpus.index.ntotal} vectors"})

        # ── Stage 1: Query Analyzer ────────────────────────────────────────
        emit("stage", {"stage": 1, "name": "Query Analyzer", "status": "running",
                        "detail": f'Analyzing: "{query}"'})
        qa = QueryAnalyzer(llm)
        analysis = qa.analyze(query)
        emit("stage", {"stage": 1, "name": "Query Analyzer", "status": "done",
                        "detail": f"{len(analysis.intents)} intents · "
                                  f"{len(analysis.reformulated_queries)} reformulated queries"})

        # ── Stage 2: Retriever ────────────────────────────────────────────
        emit("stage", {"stage": 2, "name": "Retriever (ReAct)", "status": "running",
                        "detail": "Searching corpus…"})
        retriever = Retriever(corpus, llm)
        retrieved, react_log = retriever.retrieve(
            query=query,
            reformulated_queries=analysis.reformulated_queries,
            top_k=15,
        )
        emit("stage", {"stage": 2, "name": "Retriever (ReAct)", "status": "done",
                        "detail": f"{len(retrieved)} candidates · "
                                  f"{len(react_log)} ReAct iteration(s)"})

        if not retrieved:
            emit("error", {"message": "No papers retrieved. Check corpus."})
            event_queue.put(None)
            return

        # ── Stage 3: Ranker ───────────────────────────────────────────────
        emit("stage", {"stage": 3, "name": "Ranker", "status": "running",
                        "detail": f"Scoring {len(retrieved)} candidates…"})
        ranker  = Ranker(llm)
        ranked  = ranker.rank(query, retrieved, top_n=top_n)
        top_titles = [f"{p.title[:45]}…" if len(p.title) > 45 else p.title
                      for p in ranked]
        emit("stage", {"stage": 3, "name": "Ranker", "status": "done",
                        "detail": f"Top {len(ranked)} selected",
                        "papers": top_titles})

        # ── Stage 4: Extractor ────────────────────────────────────────────
        emit("stage", {"stage": 4, "name": "Contribution Extractor",
                        "status": "running",
                        "detail": "Extracting structured fields (schema-guided)…"})
        extractor     = ContributionExtractor(llm)
        contributions = extractor.extract(ranked, paper_texts)
        compliance    = extractor.validate_schema_compliance(contributions)
        emit("stage", {"stage": 4, "name": "Contribution Extractor", "status": "done",
                        "detail": f"Schema compliance: "
                                  f"{compliance['compliance_rate']:.0%} "
                                  f"({compliance['compliant']}/{compliance['total']})"})

        # ── Stage 5: Comparison Agent ─────────────────────────────────────
        emit("stage", {"stage": 5, "name": "Comparison Agent", "status": "running",
                        "detail": "Running 3-pass comparison…"})
        ca         = ComparisonAgent(llm)
        comparison = ca.compare(contributions)
        emit("stage", {"stage": 5, "name": "Comparison Agent", "status": "done",
                        "detail": f"Overlaps: {len(comparison.overlaps)} · "
                                  f"Differences: {len(comparison.differences)} · "
                                  f"Gaps: {len(comparison.gaps)}"})

        # ── Stage 6: Answer Generator ─────────────────────────────────────
        emit("stage", {"stage": 6, "name": "Answer Generator", "status": "running",
                        "detail": "Generating final report…"})
        ag     = AnswerGenerator(llm)
        report = ag.generate(query, contributions, comparison)

        elapsed = round(time.time() - t0, 2)
        emit("stage", {"stage": 6, "name": "Answer Generator", "status": "done",
                        "detail": f"Report ready in {elapsed}s"})

        # ── Save result ───────────────────────────────────────────────────
        ts          = int(time.time())
        result_path = os.path.join(RESULTS_DIR, f"run_{ts}.json")
        result_data = {
            "report": report.model_dump(),
            "metadata": {
                "query":           query,
                "elapsed_seconds": elapsed,
                "stages": {
                    "query_analysis": analysis.model_dump(),
                    "retrieval": {
                        "n_candidates": len(retrieved),
                        "react_steps":  len(react_log),
                    },
                    "ranking": {"selected": [p.paper_id for p in ranked]},
                    "extraction": {"compliance": compliance},
                    "comparison": {
                        "n_overlaps":    len(comparison.overlaps),
                        "n_differences": len(comparison.differences),
                        "n_gaps":        len(comparison.gaps),
                    },
                },
            },
        }
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)

        # ── Done ──────────────────────────────────────────────────────────
        emit("complete", {
            "elapsed": elapsed,
            "result_file": os.path.basename(result_path),
            "report": result_data["report"],
            "metadata": result_data["metadata"],
        })

    except Exception as e:
        event_queue.put(f'data: {json.dumps({"type":"error","message":str(e)})}\n\n')

    finally:
        event_queue.put(None)   # sentinel — stream is done


# ── SSE stream generator ───────────────────────────────────────────────────────
def sse_generator(q: str, top_n: int, api_key: str = "") -> Generator[str, None, None]:
    event_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=run_pipeline_threaded,
        args=(q, top_n, event_queue, api_key),
        daemon=True,
    )
    thread.start()

    while True:
        item = event_queue.get()
        if item is None:
            break
        yield item


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h2>Frontend not found. Place index.html in /static/</h2>")


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """Stream pipeline progress as Server-Sent Events."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return StreamingResponse(
        sse_generator(req.query, req.top_n, req.api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/results")
async def list_results():
    """List all saved result files."""
    files = []
    for f in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if f.endswith(".json") and f.startswith("run_"):
            path = os.path.join(RESULTS_DIR, f)
            try:
                with open(path) as fh:
                    data = json.load(fh)
                files.append({
                    "filename":        f,
                    "query":           data.get("metadata", {}).get("query", ""),
                    "elapsed_seconds": data.get("metadata", {}).get("elapsed_seconds", 0),
                    "timestamp":       f.replace("run_", "").replace(".json", ""),
                })
            except Exception:
                pass
    return {"results": files}


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    """Get a specific result by filename."""
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/corpus/stats")
async def corpus_stats():
    """Return basic corpus statistics."""
    chunks_path     = os.path.join(CORPUS_PATH, "chunks.json")
    paper_txt_path  = os.path.join(CORPUS_PATH, "paper_texts.json")
    n_papers = 0
    n_chunks = 0
    if os.path.exists(chunks_path):
        with open(chunks_path) as f:
            chunks   = json.load(f)
            n_chunks = len(chunks)
            n_papers = len(set(c.get("paper_id", "") for c in chunks))
    return {
        "corpus_path":  CORPUS_PATH,
        "n_papers":     n_papers,
        "n_chunks":     n_chunks,
        "index_exists": os.path.exists(os.path.join(CORPUS_PATH, "faiss.index")),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
