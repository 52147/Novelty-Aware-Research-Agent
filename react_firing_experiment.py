#!/usr/bin/env python3
"""
react_firing_experiment.py
==========================
Demonstrates and measures the ReAct retrieval loop activating under sparse-query
and strict-threshold conditions.

Usage:
    export OPENAI_API_KEY="sk-..."
    mkdir -p results
    python react_firing_experiment.py
"""

import os
import json

import retriever as retriever_module
from llm_client import LLMClient
from corpus_builder import CorpusBuilder
from query_analyzer import QueryAnalyzer
from retriever import Retriever

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus")

SPARSE_QUERIES = [
    "verbal self-reflection memory for single-agent code repair",
    "chunked cross-attention retrieval from trillion-token datastores",
    "zero-ablation component attribution in transformer reasoning",
    "operating-system style paged memory for unbounded LLM context",
    "dialectic multi-robot motion-validated collaboration",
]

THRESHOLDS = {
    "default (L2<400)": 400.0,
    "strict (L2<1)": 1.0,
}


def get_step_action(step):
    if isinstance(step, dict):
        return step.get("action")
    return getattr(step, "action", None)


def get_step_observation(step):
    if isinstance(step, dict):
        return step.get("observation", "")
    return getattr(step, "observation", "")


def run_condition(label: str, threshold: float, llm, corpus, qa):
    retriever_module.L2_RELEVANCE_THRESHOLD = threshold
    retriever = Retriever(corpus, llm)

    rows = []
    fired = 0

    for q in SPARSE_QUERIES:
        analysis = qa.analyze(q)
        retrieved, react_log = retriever.retrieve(
            query=q,
            reformulated_queries=analysis.reformulated_queries,
            top_k=15,
        )

        refine_issued = any(get_step_action(step) == "REFINE" for step in react_log)

        iter1_papers = None
        if react_log:
            first_obs = get_step_observation(react_log[0])
            try:
                iter1_papers = int(
                    first_obs.split("Retrieved")[1].split("unique")[0].strip()
                )
            except Exception:
                iter1_papers = None

        n_iters = len(react_log)
        final_papers = len(retrieved)

        if refine_issued:
            fired += 1

        rows.append({
            "query": q[:48],
            "iter1": iter1_papers,
            "refine": "YES" if refine_issued else "no",
            "iters": n_iters,
            "final": final_papers,
        })

    return rows, fired


def main():
    os.makedirs("results", exist_ok=True)

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: no OPENAI_API_KEY set — runs in mock mode, loop may not fire realistically.")

    print("=" * 78)
    print("  ReAct Firing Experiment")
    print("=" * 78)

    llm = LLMClient()
    corpus = CorpusBuilder()
    corpus.load_index(CORPUS_PATH)
    qa = QueryAnalyzer(llm)

    summary = {}

    for label, thr in THRESHOLDS.items():
        print(f"\n\n### Condition: {label} ###")
        rows, fired = run_condition(label, thr, llm, corpus, qa)
        summary[label] = (rows, fired)

    print("\n\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)

    for label, (rows, fired) in summary.items():
        print(f"\nCondition: {label}")
        print(f"{'query':50}{'iter1':>7}{'refine':>8}{'iters':>7}{'final':>7}")
        print("-" * 79)

        for r in rows:
            print(
                f"{r['query']:50}"
                f"{str(r['iter1']):>7}"
                f"{r['refine']:>8}"
                f"{r['iters']:>7}"
                f"{r['final']:>7}"
            )

        print(f"\n  Loop fired on {fired}/{len(rows)} queries under this condition.")

    print("\n" + "=" * 78)
    print("  HEADLINE FOR PAPER")
    print("=" * 78)

    strict_rows, strict_fired = summary["strict (L2<1)"]
    default_rows, default_fired = summary["default (L2<400)"]

    recovered = sum(
        1 for r in strict_rows
        if r["refine"] == "YES" and (r["final"] or 0) >= 3
    )

    print(f"  Default threshold:  loop fired on {default_fired}/{len(default_rows)} queries.")
    print(f"  Strict threshold:   loop fired on {strict_fired}/{len(strict_rows)} queries.")
    print(
        f"  Of the queries that fired under the strict condition, "
        f"{recovered} recovered to >= 3 papers via refinement."
    )

    out = {label: rows for label, (rows, _) in summary.items()}

    with open("results/react_firing.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n  Saved → results/react_firing.json")


if __name__ == "__main__":
    main()