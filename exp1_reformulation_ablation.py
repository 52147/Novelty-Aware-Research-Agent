#!/usr/bin/env python3
"""
exp1_reformulation_ablation.py
==============================
Does query reformulation actually help retrieval?

For each of the three main queries, we run retrieval + ranking using
1, 2, and 4 reformulated queries, and measure Recall@5 against the
author-assigned relevance labels in relevance_labels.json.

If recall rises as more reformulations are added, the Query Analyzer's
reformulation step is doing useful work. If it doesn't, that's an honest
finding worth reporting.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python exp1_reformulation_ablation.py

Requires relevance_labels.json (from: python ir_metrics.py --init, then graded).
"""

import os
import json

from llm_client import LLMClient
from corpus_builder import CorpusBuilder
from query_analyzer import QueryAnalyzer
from retriever import Retriever
from ranker import Ranker

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus")
LABELS_FILE = "relevance_labels.json"
K = 5
REFORMULATION_COUNTS = [1, 2, 4]

QUERIES = {
    "run_01.json": "Compare multi-agent LLM frameworks for collaborative reasoning",
    "run_02.json": "What evaluation methods exist for LLM reasoning agents?",
    "run_03.json": "Compare verbal reinforcement and role-playing approaches in LLM agents",
}


def recall_at_k(ranked_ids, rel, k=K):
    total_relevant = sum(1 for g in rel.values() if g > 0)
    if total_relevant == 0:
        return None
    hits = sum(1 for pid in ranked_ids[:k] if rel.get(pid, 0) > 0)
    return hits / total_relevant


def main():
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found. Run `python ir_metrics.py --init` and grade it first.")
        return

    with open(LABELS_FILE) as f:
        labels = json.load(f)

    llm = LLMClient()
    corpus = CorpusBuilder()
    corpus.load_index(CORPUS_PATH)
    qa = QueryAnalyzer(llm)
    retriever = Retriever(corpus, llm)
    ranker = Ranker(llm)

    rows = []  # (run, query, {n: recall})
    for run_file, query in QUERIES.items():
        if run_file not in labels["queries"]:
            print(f"  skip {run_file}: no labels")
            continue
        rel = {k: int(v) for k, v in labels["queries"][run_file]["relevance"].items()}

        # Get the full set of reformulations once (deterministic-ish at low T)
        analysis = qa.analyze(query)
        all_refs = analysis.reformulated_queries

        recalls = {}
        for n in REFORMULATION_COUNTS:
            refs = all_refs[:n]
            retrieved, _ = retriever.retrieve(query=query, reformulated_queries=refs, top_k=15)
            ranked = ranker.rank(query, retrieved, top_n=K)
            ranked_ids = [p.paper_id for p in ranked]
            recalls[n] = recall_at_k(ranked_ids, rel)
        rows.append((run_file, query, recalls))

    # ── Table ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  EXPERIMENT 1: REFORMULATION ABLATION (Recall@5)")
    print("=" * 78)
    hdr = f"{'query':46}" + "".join(f"{'n='+str(n):>10}" for n in REFORMULATION_COUNTS)
    print(hdr)
    print("-" * len(hdr))

    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else "n/a"

    means = {n: [] for n in REFORMULATION_COUNTS}
    for run_file, query, recalls in rows:
        line = f"{query[:46]:46}"
        for n in REFORMULATION_COUNTS:
            line += f"{fmt(recalls[n]):>10}"
            if isinstance(recalls[n], float):
                means[n].append(recalls[n])
        print(line)
    print("-" * len(hdr))
    mean_line = f"{'MEAN':46}"
    for n in REFORMULATION_COUNTS:
        m = sum(means[n]) / len(means[n]) if means[n] else None
        mean_line += f"{fmt(m):>10}"
    print(mean_line)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "reformulation_counts": REFORMULATION_COUNTS,
        "per_query": [
            {"run": rf, "query": q, "recall_at_5_by_n": rec}
            for rf, q, rec in rows
        ],
        "mean_recall_by_n": {
            str(n): (sum(means[n]) / len(means[n]) if means[n] else None)
            for n in REFORMULATION_COUNTS
        },
    }
    os.makedirs("results", exist_ok=True)
    with open("results/reformulation_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/reformulation_ablation.json")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  LATEX TABLE")
    print("=" * 78)
    print(r"\begin{table}[t]")
    print(r"\caption{Reformulation ablation: Recall@5 as the number of "
          r"reformulated queries increases.}")
    print(r"\label{tab:reformulation}")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"\textbf{Query} & \textbf{1 ref.} & \textbf{2 ref.} & \textbf{4 ref.} \\")
    print(r"\midrule")
    short = {"run_01.json": "R1 (multi-agent)", "run_02.json": "R2 (evaluation)",
             "run_03.json": "R3 (verbal RL)"}
    for rf, q, rec in rows:
        cells = " & ".join(fmt(rec[n]) for n in REFORMULATION_COUNTS)
        print(f"{short.get(rf, rf)} & {cells} \\\\")
    print(r"\midrule")
    mcells = " & ".join(
        (f"{sum(means[n])/len(means[n]):.3f}" if means[n] else "--")
        for n in REFORMULATION_COUNTS)
    print(f"\\textbf{{Mean}} & {mcells} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
