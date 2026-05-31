#!/usr/bin/env python3
"""
exp3_baseline_all_queries.py
============================
Runs the baseline RAG system on ALL THREE main queries (not just R1), so the
baseline comparison is no longer anecdotal.

For each query it also computes the same IR metrics (P@5, nDCG@5, Recall@5, MRR)
on the baseline's retrieved set, using relevance_labels.json. This lets you
report a head-to-head retrieval comparison: full system vs baseline.

Note: the baseline retrieves by raw FAISS similarity with no ranker, so its
ordering is the dedup retrieval order. Comparing its IR metrics against the
full system's ranker output isolates the value of the ranking stage.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python exp3_baseline_all_queries.py
"""

import os
import json
import math

from baseline_rag import run_baseline

LABELS_FILE = "relevance_labels.json"
K = 5

QUERIES = {
    "run_01.json": "Compare multi-agent LLM frameworks for collaborative reasoning",
    "run_02.json": "What evaluation methods exist for LLM reasoning agents?",
    "run_03.json": "Compare verbal reinforcement and role-playing approaches in LLM agents",
}
OUT_FILES = {
    "run_01.json": "results/baseline_run_01.json",
    "run_02.json": "results/baseline_run_02.json",
    "run_03.json": "results/baseline_run_03.json",
}


def precision_at_k(ranked, rel, k=K):
    topk = ranked[:k]
    return sum(1 for pid in topk if rel.get(pid, 0) > 0) / k if topk else 0.0


def recall_at_k(ranked, rel, k=K):
    total = sum(1 for g in rel.values() if g > 0)
    if total == 0:
        return None
    return sum(1 for pid in ranked[:k] if rel.get(pid, 0) > 0) / total


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked, rel, k=K):
    gains = [rel.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else None


def mrr(ranked, rel):
    for i, pid in enumerate(ranked):
        if rel.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def main():
    labels = None
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    else:
        print("WARNING: no relevance_labels.json — will run baseline but skip IR metrics.")

    os.makedirs("results", exist_ok=True)
    rows = []

    for run_file, query in QUERIES.items():
        print(f"\nRunning baseline on {run_file}: {query[:50]}…")
        result = run_baseline(query)
        with open(OUT_FILES[run_file], "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved → {OUT_FILES[run_file]} ({result['elapsed_seconds']}s)")

        metrics = None
        if labels and run_file in labels["queries"]:
            rel = {k: int(v) for k, v in labels["queries"][run_file]["relevance"].items()}
            ranked = result["papers_retrieved"]
            metrics = {
                "P@5": precision_at_k(ranked, rel),
                "nDCG@5": ndcg_at_k(ranked, rel),
                "Recall@5": recall_at_k(ranked, rel),
                "MRR": mrr(ranked, rel),
            }
        rows.append((run_file, query, result["elapsed_seconds"], metrics))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  EXPERIMENT 3: BASELINE ON ALL THREE QUERIES")
    print("=" * 78)
    if any(m for _, _, _, m in rows):
        hdr = f"{'run':12}{'sec':>7}{'P@5':>8}{'nDCG@5':>9}{'Recall@5':>10}{'MRR':>7}"
        print(hdr); print("-" * len(hdr))

        def fmt(x): return f"{x:.3f}" if isinstance(x, float) else "n/a"
        agg = {"P@5": [], "nDCG@5": [], "Recall@5": [], "MRR": []}
        for rf, q, sec, m in rows:
            if m:
                print(f"{rf:12}{sec:>7}{fmt(m['P@5']):>8}{fmt(m['nDCG@5']):>9}"
                      f"{fmt(m['Recall@5']):>10}{fmt(m['MRR']):>7}")
                for c in agg:
                    if isinstance(m[c], float): agg[c].append(m[c])
        print("-" * len(hdr))
        def avg(l): return sum(l)/len(l) if l else None
        print(f"{'MEAN':12}{'':>7}{fmt(avg(agg['P@5'])):>8}{fmt(avg(agg['nDCG@5'])):>9}"
              f"{fmt(avg(agg['Recall@5'])):>10}{fmt(avg(agg['MRR'])):>7}")
        print("\nCompare these to the full-system numbers in results/ir_metrics.json")
    else:
        print("Baseline outputs saved. (No labels found, so no IR metrics computed.)")


if __name__ == "__main__":
    main()
