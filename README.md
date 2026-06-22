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

**An agentic retrieval system for identifying, extracting, and comparing the claimed contributions of academic papers in a focused domain.**

Unlike standard RAG systems that summarize papers independently, this system performs structured multi-step reasoning over retrieved papers to surface **contribution-level differences**, **overlaps**, and **corpus-level methodological gaps**.

---

## 🎬 Demo GIF

---

## 🚀 How to Use

1. **Enter your OpenAI API key** in the sidebar (`sk-...`) — stored in your browser only, never logged
2. **Type a comparison query** — e.g. *"Compare multi-agent LLM frameworks for collaborative reasoning"*
3. **Watch the 6-stage pipeline** run in real time (~23 seconds)
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

## 📈 Experimental Results (100-paper corpus)

Automated metrics across the three main queries:

| Metric | R1 | R2 | R3 | Avg |
|---|---|---|---|---|
| Schema compliance | 80% | 80% | 100% | **86.7%** |
| Overlaps detected | 3 | 3 | 2 | 2.7 |
| Gaps identified | 5 | 4 | 5 | 4.7 |
| Runtime (s) | 22.3 | 23.5 | 22.9 | **22.9** |

Retrieval quality under author-assigned graded relevance:

| Metric | 3 main queries | 10 queries (extended) |
|---|---|---|
| Precision@5 | 1.000 | 0.980 |
| nDCG@5 | 0.752 | 0.739 |
| Recall@5 | 0.527 | 0.489 |
| Mean pairwise Jaccard | 0.12 | 0.115 |

**Key finding:** No paper appears in all three main top-5 sets — 12 of the 15 ranked slots hold distinct papers, and across the extended ten-query evaluation 18 of 29 retrieved papers are query-exclusive. This confirms query-sensitive retrieval and ranking. The ranker leads BM25, dense, and hybrid retrieval on the main queries, and Precision@5 stays high but non-saturated (0.980) over the wider ten-query set.

---

## ⚠️ Important Notes

- The corpus contains **100 papers** in the agentic AI domain (multi-agent frameworks, reasoning techniques, tool-use systems, evaluation/survey work)
- Each query takes **~23 seconds** — the structured pipeline is slower than a basic summarization call by design (roughly sevenfold over a single-call RAG baseline)
- Relevance labels are **author-assigned** and the corpus is small-scale; results are a prototype evaluation, not a benchmark
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

> **Novelty-Aware Agentic Retrieval: Comparing Research Contributions Through Structured Multi-Step Reasoning**
> Shou-Tzu Han · Department of Computer Science · University of South Dakota

---

## 📝 License

MIT