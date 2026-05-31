#!/usr/bin/env python3
"""
ir_metrics.py — Precision@5, nDCG@5, Recall@5, MRR for the ranker output.

Two-step workflow
-----------------
1) Generate a relevance-labeling template from your run files + corpus:

       python ir_metrics.py --init

   This writes relevance_labels.json with every retrieved paper pre-listed
   per query (grade left at 0) plus the full corpus list for reference.

2) Open relevance_labels.json and assign a relevance grade to each paper:

       3 = highly relevant   (directly answers the query)
       2 = relevant
       1 = marginally relevant
       0 = not relevant

   Grade at least the retrieved top-5 for each query. To get a meaningful
   Recall@5, also grade any OTHER corpus papers you consider relevant.

3) Compute metrics:

       python ir_metrics.py

Notes
-----
- Ground-truth relevance is author-assigned. This is standard for small IR
  studies; report it as such in the paper.
- per_paper_summaries in each run file is in rank order (top-1 first).
"""

import os
import sys
import json
import glob
import math

from compare_retrievers import avg

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
CORPUS_FILE = "sample_papers.json"
LABELS_FILE = "relevance_labels.json"
K = 5

# The three main run files (edit if your filenames differ)
RUN_FILES = ["run_01.json", "run_02.json", "run_03.json"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_run(path):
    with open(path) as f:
        data = json.load(f)
    report = data.get("report", data)
    query = report.get("query", "")
    ranked = [p["paper_id"] for p in report.get("per_paper_summaries", [])]
    return query, ranked


def load_corpus_ids():
    if not os.path.exists(CORPUS_FILE):
        return {}
    with open(CORPUS_FILE) as f:
        papers = json.load(f)
    return {p["paper_id"]: p["title"] for p in papers}


# ── Step 1: generate template ─────────────────────────────────────────────────
def init_template():
    corpus = load_corpus_ids()
    template = {
        "_instructions": (
            "Assign a relevance grade to each paper per query. "
            "3=highly relevant, 2=relevant, 1=marginal, 0=not relevant. "
            "Grade at least the retrieved papers; grade other corpus papers "
            "too for a meaningful Recall@5."
        ),
        "_grade_scale": {"3": "highly relevant", "2": "relevant",
                         "1": "marginal", "0": "not relevant"},
        "queries": {},
        "_corpus_reference": corpus,
    }

    for rf in RUN_FILES:
        path = os.path.join(RESULTS_DIR, rf)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        query, ranked = load_run(path)
        template["queries"][rf] = {
            "query": query,
            "retrieved_top5_in_rank_order": ranked,
            # pre-fill retrieved papers at grade 0 for the user to edit
            "relevance": {pid: 0 for pid in ranked},
        }

    with open(LABELS_FILE, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Wrote template → {LABELS_FILE}")
    print("Now open it and assign relevance grades, then run: python ir_metrics.py")


# ── Metrics ───────────────────────────────────────────────────────────────────
def precision_at_k(ranked, rel, k=K):
    topk = ranked[:k]
    hits = sum(1 for pid in topk if rel.get(pid, 0) > 0)
    return hits / k


def recall_at_k(ranked, rel, k=K):
    total_relevant = sum(1 for g in rel.values() if g > 0)
    if total_relevant == 0:
        return None
    topk = ranked[:k]
    hits = sum(1 for pid in topk if rel.get(pid, 0) > 0)
    return hits / total_relevant


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked, rel, k=K):
    gains = [rel.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg(gains) / idcg


def mrr(ranked, rel):
    for i, pid in enumerate(ranked):
        if rel.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ── Step 2: compute ───────────────────────────────────────────────────────────
def compute():
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found. Run `python ir_metrics.py --init` first.")
        return

    with open(LABELS_FILE) as f:
        labels = json.load(f)

    rows = []
    for rf, entry in labels["queries"].items():
        path = os.path.join(RESULTS_DIR, rf)
        if not os.path.exists(path):
            continue
        _, ranked = load_run(path)
        rel = {k: int(v) for k, v in entry["relevance"].items()}

        p = precision_at_k(ranked, rel)
        n = ndcg_at_k(ranked, rel)
        r = recall_at_k(ranked, rel)
        m = mrr(ranked, rel)
        rows.append((rf, entry["query"][:42], p, n, r, m))

    if not rows:
        print("No runs to score.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("  IR RETRIEVAL METRICS (author-assigned relevance)")
    print("=" * 86)
    hdr = f"{'run':12}{'query':44}{'P@5':>7}{'nDCG@5':>9}{'Recall@5':>10}{'MRR':>7}"
    print(hdr)
    print("-" * len(hdr))

    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else "n/a"

    sums = {"p": [], "n": [], "r": [], "m": []}
    for rf, q, p, n, r, m in rows:
        print(f"{rf:12}{q:44}{fmt(p):>7}{fmt(n):>9}{fmt(r) if r is not None else 'n/a':>10}{fmt(m):>7}")
        sums["p"].append(p)
        if n is not None: sums["n"].append(n)
        if r is not None: sums["r"].append(r)
        sums["m"].append(m)

    print("-" * len(hdr))
    def avg(lst): return sum(lst) / len(lst) if lst else None
    print(f"{'MEAN':12}{'':44}"
          f"{fmt(avg(sums['p'])):>7}"
          f"{fmt(avg(sums['n'])):>9}"
          f"{fmt(avg(sums['r'])) if sums['r'] else 'n/a':>10}"
          f"{fmt(avg(sums['m'])):>7}")

    # ── Save ────────────────────────────────────────────────────────────────
    out = {
        "per_query": [
            {"run": rf, "query": q, "precision_at_5": p,
             "ndcg_at_5": n, "recall_at_5": r, "mrr": m}
            for rf, q, p, n, r, m in rows
        ],
        "mean": {
            "precision_at_5": avg(sums["p"]),
            "ndcg_at_5": avg(sums["n"]),
            "recall_at_5": avg(sums["r"]),
            "mrr": avg(sums["m"]),
        },
    }
    with open(os.path.join(RESULTS_DIR, "ir_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {os.path.join(RESULTS_DIR, 'ir_metrics.json')}")

    # ── LaTeX table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("  LATEX TABLE (paste into the paper)")
    print("=" * 86)
    print(r"\begin{table}[t]")
    print(r"\caption{Retrieval quality on the three main queries "
          r"(author-assigned graded relevance).}")
    print(r"\label{tab:ir-metrics}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"\textbf{Query} & \textbf{P@5} & \textbf{nDCG@5} & "
          r"\textbf{Recall@5} & \textbf{MRR} \\")
    print(r"\midrule")
    labels_short = {"run_01.json": "R1 (multi-agent)",
                    "run_02.json": "R2 (evaluation)",
                    "run_03.json": "R3 (verbal RL)"}
    for rf, q, p, n, r, m in rows:
        lbl = labels_short.get(rf, rf)
        rstr = f"{r:.3f}" if r is not None else "--"
        nstr = f"{n:.3f}" if n is not None else "--"
        print(f"{lbl} & {p:.3f} & {nstr} & {rstr} & {m:.3f} \\\\")
    print(r"\midrule")
    mean_p = avg(sums["p"])
    mean_n = avg(sums["n"])
    mean_r = avg(sums["r"]) if sums["r"] else None
    mean_m = avg(sums["m"])
    def s(x): return f"{x:.3f}" if x is not None else "--"
    print(f"\\textbf{{Mean}} & {s(mean_p)} & {s(mean_n)} & {s(mean_r)} & {s(mean_m)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        init_template()
    else:
        compute()
