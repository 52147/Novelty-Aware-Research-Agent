---
title: Novelty-Aware Research Agent
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
app_port: 7860
---

# 🔬 Novelty-Aware Research Agent

> An agentic AI system for identifying, extracting, and comparing the claimed contributions of academic papers in a focused domain.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-OpenAI-412991?logo=openai)](https://openai.com)

Unlike standard RAG systems that summarize papers independently, this system performs **structured multi-step reasoning** over retrieved papers to surface contribution-level differences, overlaps, and corpus-level methodological gaps.

## 🚀 How to Use

1. **Enter your OpenAI API key** in the sidebar (`sk-...`)
2. **Type a comparison query** — e.g. *"Compare multi-agent LLM frameworks for collaborative reasoning"*
3. **Watch the 6-stage pipeline** run in real time
4. **Explore structured results** — overlaps, differentiating aspects, gap matrix, synthesis

## 🏗️ Pipeline

```
Query Analyzer → Retriever (ReAct) → Ranker → Extractor → Comparison Agent → Generator
    T=0.3            T=0.3           T=0.1     T=0.1          T=0.2           T=0.7
```

## ⚠️ Notes

- The corpus contains **100 papers** in the agentic AI domain
- Each query takes **~25 seconds** (structured pipeline vs. ~4s for basic RAG)
- Your API key is stored in your browser session only — never logged server-side
- Gap findings are **corpus-level observations only**, not claims about the broader literature

## 📄 Paper

**Novelty-Aware Research Agent: An Agentic AI System for Comparing Research Contributions Across Papers**
Shou-Tzu Han · University of South Dakota · CSC 792: Topics in Agentic AI · Spring 2026

## 🛠️ Tech Stack

Python · FastAPI · GPT-4o · FAISS · Sentence-Transformers · Pydantic v2 · Server-Sent Events
