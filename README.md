---
title: Novelty-Aware Research Agent
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🔬 Novelty-Aware Research Agent

**An agentic AI system for identifying, extracting, and comparing the claimed contributions of academic papers in a focused domain.**

Unlike standard RAG systems that summarize papers independently, this system performs structured multi-step reasoning over retrieved papers to surface **contribution-level differences**, **overlaps**, and **corpus-level methodological gaps**.

---

## 🚀 How to Use

1. **Enter your OpenAI API key** in the sidebar (`sk-...`) — stored in your browser only, never logged
2. **Type a comparison query** — e.g. *"Compare multi-agent LLM frameworks for collaborative reasoning"*
3. **Watch the 6-stage pipeline** run in real time (~35 seconds)
4. **Explore structured results** — per-paper summaries, overlaps, differentiating aspects, gap matrix, synthesis

---

## 🏗️ Pipeline Architecture

```
① Query Analyzer    →    decompose query into intents            (T=0.3)
② Retriever (ReAct) →    FAISS search with reasoning loop        (T=0.3)
③ Ranker            →    score and select top-N papers           (T=0.1)
④ Extractor         →    schema-guided 4-field extraction        (T=0.1)
⑤ Comparison Agent  →    overlap · differentiation · gap matrix  (T=0.2)
⑥ Answer Generator  →    citation-grounded synthesis report      (T=0.7)
```

Each component has a typed input/output contract enforced by Pydantic v2.
The **Contribution Extractor** uses JSON schema-guided decoding to guarantee structured output without retries.

---

## 📊 What the System Produces

For each query, the system returns:

- **Per-paper summaries** — one-sentence contribution summary per retrieved paper
- **Overlaps** — papers sharing the same problem formulation, dataset, or method family
- **Differentiating aspects** — what each paper does distinctly
- **Problem × Method gap matrix** — combinations absent from the retrieved corpus
- **Synthesis paragraph** — cross-paper narrative with citations

---

## 📈 Experimental Results (20-paper corpus)

| Metric | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Schema compliance | 80% | 80% | 100% | **87%** |
| Overlaps detected | 3 | 2 | 3 | 2.7 |
| Gaps identified | 5 | 4 | 5 | 4.7 |
| Runtime (s) | 32.2 | 31.2 | 41.7 | **35.0** |

**Key finding:** Only 1 of 5 selected papers was shared across all three runs — 12 of 15 paper slots changed — confirming query-sensitive retrieval and ranking.

---

## ⚠️ Important Notes

- The corpus contains **20 papers** in the agentic AI domain (multi-agent frameworks, reasoning techniques, tool-use systems, evaluation/survey work)
- Each query takes **~35 seconds** — the structured pipeline is slower than a basic summarization call by design
- Gap findings are **corpus-level observations only** — not claims about the broader research literature
- Your API key is used solely for OpenAI inference and never stored server-side

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM inference | GPT-4o (OpenAI) |
| Vector search | FAISS IndexFlatL2 |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Data contracts | Pydantic v2 |
| Backend | FastAPI + uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Vanilla JS / HTML / CSS |

---

## 📄 Paper

> **Novelty-Aware Research Agent: An Agentic AI System for Comparing Research Contributions Across Papers**
> Shou-Tzu Han · Department of Data Science and Engineering · University of South Dakota
> CSC 792: Topics in Agentic AI · Spring 2026

---

## 📝 License

MIT