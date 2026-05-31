#!/usr/bin/env python3
"""
exp2_metrics_at_k.py
====================
Computes Precision, nDCG, and Recall at BOTH k=5 and k=10, plus MRR.

The saved run_*.json files only store the top-5, so this script re-runs the
retrieval + ranking stage with top_n=10 to obtain the top-10 ranking, then
computes metrics at both cutoffs. Reporting two cutoffs counters any
"cherry-picked k=5" concern.

If new papers appear in the top-10 that you have not graded, the script lists
them so you can add grades to relevance_labels.json and re-run.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python exp2_metrics_at_k.py

Requires relevance_labels.json.
"""

import os
import json
import math

from llm_client import LLMClient
from corpus_builder import CorpusBuilder
from query_analyzer import QueryAnalyzer
from retriever import Retriever
from ranker import Ranker

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus")
LABELS_FILE = "relevance_labels.json"
CUTOFFS = [5, 10]

QUERIES = {
    "run_01.json": "Compare multi-agent LLM frameworks for collaborative reasoning",
    "run_02.json": "What evaluation methods exist for LLM reasoning agents?",
    "run_03.json": "Compare verbal reinforcement and role-playing approaches in LLM agents",
}


def precision_at_k(ranked, rel, k):
    topk = ranked[:k]
    if not topk:
        return 0.0
    return sum(1 for pid in topk if rel.get(pid, 0) > 0) / k


def recall_at_k(ranked, rel, k):
    total = sum(1 for g in rel.values() if g > 0)
    if total == 0:
        return None
    return sum(1 for pid in ranked[:k] if rel.get(pid, 0) > 0) / total


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked, rel, k):
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
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found.")
        return

    with open(LABELS_FILE) as f:
        labels = json.load(f)

    llm = LLMClient()
    corpus = CorpusBuilder()
    corpus.load_index(CORPUS_PATH)
    qa = QueryAnalyzer(llm)
    retriever = Retriever(corpus, llm)
    ranker = Ranker(llm)

    rows = []
    ungraded_warnings = []

    for run_file, query in QUERIES.items():
        if run_file not in labels["queries"]:
            continue
        rel = {k: int(v) for k, v in labels["queries"][run_file]["relevance"].items()}

        analysis = qa.analyze(query)
        retrieved, _ = retriever.retrieve(
            query=query, reformulated_queries=analysis.reformulated_queries, top_k=20)
        ranked = ranker.rank(query, retrieved, top_n=10)
        ranked_ids = [p.paper_id for p in ranked]

        # warn about ungraded papers in top-10
        for pid in ranked_ids:
            if pid not in rel:
                ungraded_warnings.append((run_file, pid))

        metrics = {}
        for k in CUTOFFS:
            metrics[f"P@{k}"] = precision_at_k(ranked_ids, rel, k)
            metrics[f"nDCG@{k}"] = ndcg_at_k(ranked_ids, rel, k)
            metrics[f"Recall@{k}"] = recall_at_k(ranked_ids, rel, k)
        metrics["MRR"] = mrr(ranked_ids, rel)
        rows.append((run_file, query, metrics, ranked_ids))

    if ungraded_warnings:
        print("\n⚠️  UNGRADED papers appeared in top-10 (grade them in "
              "relevance_labels.json for accurate @10 metrics):")
        for rf, pid in ungraded_warnings:
            print(f"     {rf}: {pid}")
        print("   (Treated as grade 0 for now.)\n")

    # ── Table ─────────────────────────────────────────────────────────────────
    cols = [f"P@{k}" for k in CUTOFFS] + [f"nDCG@{k}" for k in CUTOFFS] \
         + [f"Recall@{k}" for k in CUTOFFS] + ["MRR"]
    print("=" * (40 + 9 * len(cols)))
    print("  EXPERIMENT 2: METRICS AT k=5 AND k=10")
    print("=" * (40 + 9 * len(cols)))
    hdr = f"{'query':30}" + "".join(f"{c:>9}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else "n/a"

    agg = {c: [] for c in cols}
    for rf, q, m, _ in rows:
        line = f"{q[:30]:30}"
        for c in cols:
            line += f"{fmt(m[c]):>9}"
            if isinstance(m[c], float):
                agg[c].append(m[c])
        print(line)
    print("-" * len(hdr))
    mline = f"{'MEAN':30}"
    for c in cols:
        v = sum(agg[c]) / len(agg[c]) if agg[c] else None
        mline += f"{fmt(v):>9}"
    print(mline)

    out = {
        "cutoffs": CUTOFFS,
        "per_query": [{"run": rf, "query": q, "metrics": m, "top10": ids}
                      for rf, q, m, ids in rows],
        "mean": {c: (sum(agg[c]) / len(agg[c]) if agg[c] else None) for c in cols},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_at_k.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/metrics_at_k.json")


if __name__ == "__main__":
    main()
