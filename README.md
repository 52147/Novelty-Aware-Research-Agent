# 🔬 Novelty-Aware Research Agent

> An agentic AI system for identifying, extracting, and comparing the claimed contributions of academic papers in a focused domain.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-OpenAI-412991?logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Unlike standard RAG systems that summarize papers independently, this system performs **structured multi-step reasoning** over retrieved papers to surface contribution-level differences, overlaps, and corpus-level methodological gaps.

---

## ✨ Features

- **6-component agentic pipeline** — Query Analyzer → ReAct Retriever → Ranker → Extractor → Comparison Agent → Generator
- **Schema-guided extraction** — enforces 4-field structured records per paper (Problem / Method / Contribution / Novelty)
- **Three-pass Comparison Agent** — Overlap detection, Differentiation analysis, Problem × Method gap matrix
- **Query-sensitive retrieval** — different queries return meaningfully different paper sets
- **Web UI** — real-time 6-stage progress, structured results display, past runs history
- **Baseline RAG** — included for comparison

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/novelty-aware-research-agent.git
cd novelty-aware-research-agent
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Build the corpus

```bash
python create_sample_corpus.py      # generates sample_papers.json (20 papers)
python main.py --build-corpus sample_papers.json
```

### 4a. Run via CLI

```bash
python main.py --query "Compare multi-agent LLM frameworks for collaborative reasoning" \
               --output results/run_01.json
```

### 4b. Run via Web UI

```bash
pip install -r webapp/requirements_web.txt
export PYTHONPATH="$(pwd)"
uvicorn webapp.app:app --reload --port 8001
```

Open **http://localhost:8001** — paste your API key in the sidebar and run a query.

---

## 📁 Project Structure

```
novelty_agent/
├── pipeline.py              # End-to-end orchestrator
├── query_analyzer.py        # Component 1 — query decomposition (T=0.3)
├── retriever.py             # Component 2 — ReAct FAISS retrieval (T=0.3)
├── ranker.py                # Component 3 — relevance scoring (T=0.1)
├── extractor.py             # Component 4 — schema-guided extraction (T=0.1)
├── comparison_agent.py      # Component 5 — 3-pass comparison (T=0.2)
├── answer_generator.py      # Component 6 — report synthesis (T=0.7)
├── baseline_rag.py          # Baseline: retrieve + direct GPT-4o summarization
├── corpus_builder.py        # FAISS index builder and search
├── create_sample_corpus.py  # Generates 20-paper agentic AI corpus
├── evaluate.py              # Evaluation metrics + rubric export
├── llm_client.py            # OpenAI wrapper with per-component temperature map
├── schemas.py               # Pydantic v2 inter-component contracts
├── main.py                  # CLI entry point
├── requirements.txt         # Core dependencies
├── sample_papers.json       # 20-paper corpus (generated)
├── corpus/                  # Built FAISS index (generated)
│   ├── faiss.index
│   ├── chunks.json
│   └── paper_texts.json
├── results/                 # Query results (generated)
│   └── run_*.json
└── webapp/
    ├── app.py               # FastAPI backend with SSE streaming
    ├── requirements_web.txt # Web dependencies
    └── static/
        └── index.html       # Single-page frontend
```

---

## 🏗️ Pipeline Architecture

```
User Query
    │
    ▼
① Query Analyzer (T=0.3)    — decompose into intents + reformulate queries
    │
    ▼
② Retriever / ReAct (T=0.3) — FAISS search with reasoning-based refinement (max 3 iter)
    │
    ▼
③ Ranker (T=0.1)            — score relevance, return top-N papers
    │
    ▼
④ Contribution Extractor    — schema-constrained JSON:
   (T=0.1)                    Problem · Method · Key Contribution · Claimed Novelty
    │
    ▼
⑤ Comparison Agent (T=0.2)  — Pass 1: Overlap detection
                              — Pass 2: Differentiation analysis
                              — Pass 3: Problem × method gap matrix
    │
    ▼
⑥ Answer Generator (T=0.7)  — synthesize citation-grounded report
    │
    ▼
Structured Report (JSON + readable text)
```

---

## 📊 Results (20-paper corpus, 3 experimental runs)

| Metric              | Run 1 | Run 2 | Run 3 | Avg  |
|---------------------|-------|-------|-------|------|
| Candidates retrieved| 9     | 8     | 8     | 8.3  |
| Schema compliance   | 80%   | 80%   | 100%  | 87%  |
| Overlaps detected   | 3     | 2     | 3     | 2.7  |
| Gaps identified     | 5     | 4     | 5     | 4.7  |
| Runtime (s)         | 32.2  | 31.2  | 41.7  | 35.0 |

**Key finding:** Only 1 of 5 papers appeared in all three runs — 12 of 15 paper slots changed across queries, confirming query-sensitive retrieval.

---

## 🌐 Deploy to Railway (free)

```bash
# 1. Push this repo to GitHub (make sure corpus/ is included)

# 2. Go to railway.app → New Project → Deploy from GitHub repo

# 3. Railway auto-detects Python via Procfile

# 4. Add environment variables in Railway dashboard:
#    CORPUS_PATH = corpus
#    RESULTS_DIR = results
#    (No OPENAI_API_KEY needed — users bring their own key via the UI)
```

The app will be live at `https://your-app.up.railway.app`

**Why not Vercel?** — Vercel serverless functions have a 60s timeout and limited SSE support. The pipeline takes ~35s and uses Server-Sent Events for real-time progress. Railway runs a persistent server with no timeout issues.

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Schema-guided decoding | Guarantees 4-field JSON output without retry logic |
| Per-component temperature | Low T for precision, high T for prose synthesis |
| ReAct retrieval loop | Architectural fallback for sparse or ambiguous queries |
| Structured records as input | Bounds comparison reasoning space, improves auditability |
| Corpus-level gap framing | Prevents overclaiming gaps as field-wide research absences |

---

## ⚠️ Limitations

- **Corpus scale** — 20 papers. A 40–80 paper corpus would improve retrieval precision metrics.
- **ReAct loop** — never triggered in current experiments (all queries retrieved enough papers in one iteration).
- **Schema compliance 87%** — survey papers and merged-contribution papers produce non-compliant records. Fix: add a fallback extraction prompt.
- **Gap detection** — produces cross-application observations in some cases rather than structural matrix gaps.
- **Author-only evaluation** — rubric scores need independent evaluators for unbiased assessment.

---

## 📄 Paper

**Novelty-Aware Research Agent: An Agentic AI System for Comparing Research Contributions Across Papers**  
Shou-Tzu Han · University of South Dakota · CSC 792: Topics in Agentic AI · Spring 2026

---

## 🛠️ Tech Stack

- **LLM inference** — GPT-4o (OpenAI)
- **Vector search** — FAISS IndexFlatL2
- **Embeddings** — Sentence-Transformers (all-MiniLM-L6-v2, 384-dim)
- **Data contracts** — Pydantic v2
- **Backend** — FastAPI + uvicorn
- **Frontend** — Vanilla JS / HTML / CSS (no framework)
- **Streaming** — Server-Sent Events (SSE)

---

## 📝 License

MIT — see [LICENSE](LICENSE)
