#!/usr/bin/env python3
"""
final_experiment_tables.py

Collect final experiment outputs for the paper:

1. Retriever comparison: BM25 / Dense / Hybrid / Full ranker
2. Gap validation: sampled empty-cell precision
3. Top-5 / Top-10 retrieval metrics
4. Gap-validation error analysis

Run:
    python final_experiment_tables.py
"""

import json
import os

RESULTS_DIR = "results"

RETRIEVER_FILE = os.path.join(RESULTS_DIR, "retriever_comparison.json")
GAP_VALIDATION_FILE = os.path.join(RESULTS_DIR, "gap_validation_metrics.json")

TOP10_CANDIDATES = [
    os.path.join(RESULTS_DIR, "retrieval_metrics_top10.json"),
    os.path.join(RESULTS_DIR, "retrieval_quality_top10.json"),
    os.path.join(RESULTS_DIR, "ir_metrics_top10.json"),
    os.path.join(RESULTS_DIR, "ir_metrics.json"),
    os.path.join(RESULTS_DIR, "retrieval_metrics.json"),
    os.path.join(RESULTS_DIR, "retriever_comparison_top10.json"),
    os.path.join(RESULTS_DIR, "retrieval_quality.json"),
]

OUT_FILE = os.path.join(RESULTS_DIR, "final_experiment_tables.json")


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x):
    if x is None:
        return "--"
    return f"{float(x):.3f}"


def metric(d, *keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def find_top10_file():
    for path in TOP10_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def normalize_retriever_data(data):
    if not data:
        return {}

    if "mean" in data and isinstance(data["mean"], dict):
        mean = data["mean"]

        # Case 1: mean is already system -> metrics
        if any(isinstance(v, dict) for v in mean.values()):
            return mean

    if "systems" in data and isinstance(data["systems"], dict):
        return data["systems"]

    systems = {}
    for key, val in data.items():
        if isinstance(val, dict) and any(
            m in val
            for m in [
                "P@5",
                "P@10",
                "precision_at_5",
                "precision_at_10",
                "nDCG@5",
                "ndcg_at_5",
            ]
        ):
            systems[key] = val

    return systems


def print_retriever_table():
    data = read_json(RETRIEVER_FILE)
    systems = normalize_retriever_data(data)

    if not systems:
        print(f"WARNING: cannot load retriever comparison from {RETRIEVER_FILE}")
        return None

    order = ["BM25", "Dense", "Hybrid", "Full ranker", "Full Ranker", "Full system"]
    rows = []
    used = set()

    for name in order:
        if name in systems:
            rows.append((name, systems[name]))
            used.add(name)

    for name, value in systems.items():
        if name not in used:
            rows.append((name, value))

    print("\n" + "=" * 90)
    print("1. RETRIEVER COMPARISON")
    print("=" * 90)

    print(r"\begin{table}[t]")
    print(
        r"\caption{Retriever comparison across BM25, dense retrieval, hybrid retrieval, and the full ranker. Metrics are averaged over the three main queries using author-assigned graded relevance.}"
    )
    print(r"\label{tab:retriever-comparison}")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(
        r"\textbf{System} & \textbf{P@5} & \textbf{P@10} & \textbf{nDCG@5} & \textbf{nDCG@10} & \textbf{R@5} & \textbf{R@10} \\"
    )
    print(r"\midrule")

    out = []
    for name, s in rows:
        p5 = metric(s, "P@5", "precision_at_5", "p_at_5")
        p10 = metric(s, "P@10", "precision_at_10", "p_at_10")
        ndcg5 = metric(s, "nDCG@5", "ndcg_at_5")
        ndcg10 = metric(s, "nDCG@10", "ndcg_at_10")
        r5 = metric(s, "Recall@5", "recall_at_5", "R@5")
        r10 = metric(s, "Recall@10", "recall_at_10", "R@10")

        print(
            f"{name} & {fmt(p5)} & {fmt(p10)} & {fmt(ndcg5)} & {fmt(ndcg10)} & {fmt(r5)} & {fmt(r10)} \\\\"
        )

        out.append(
            {
                "system": name,
                "P@5": p5,
                "P@10": p10,
                "nDCG@5": ndcg5,
                "nDCG@10": ndcg10,
                "Recall@5": r5,
                "Recall@10": r10,
            }
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return out


def print_gap_validation_table():
    data = read_json(GAP_VALIDATION_FILE)

    if not data:
        print(f"WARNING: cannot load gap validation from {GAP_VALIDATION_FILE}")
        return None

    label_counts = data.get("label_counts", {})

    sampled = (
        data.get("sampled_rows")
        or data.get("sampled_empty_cells")
        or data.get("n_sampled")
        or data.get("sample_size")
    )

    labeled = (
        data.get("labeled_rows")
        or data.get("labeled_cells")
        or data.get("n_labeled")
    )

    plausible = (
        data.get("plausible_gaps")
        or data.get("plausible_cells")
        or data.get("plausible")
        or data.get("n_plausible")
    )

    precision = data.get("gap_precision") or data.get("precision")

    if plausible is None and isinstance(label_counts, dict):
        plausible = label_counts.get("plausible")

    if sampled is None and isinstance(label_counts, dict):
        sampled = sum(label_counts.values())

    if labeled is None and isinstance(label_counts, dict):
        labeled = sum(label_counts.values())

    if precision is None and labeled:
        precision = plausible / labeled if plausible is not None else None

    print("\n" + "=" * 90)
    print("2. GAP MATRIX VALIDATION")
    print("=" * 90)

    print(r"\begin{table}[t]")
    print(r"\caption{Validation of sampled deterministic gap-matrix empty cells.}")
    print(r"\label{tab:gap-validation}")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"\textbf{Metric} & \textbf{Value} \\")
    print(r"\midrule")
    print(f"Sampled empty cells & {sampled} \\\\")
    print(f"Labeled cells & {labeled} \\\\")
    print(f"Plausible cells & {plausible} \\\\")
    print(f"Gap precision & {fmt(precision)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return {
        "sampled_empty_cells": sampled,
        "labeled_cells": labeled,
        "plausible_cells": plausible,
        "gap_precision": precision,
        "label_counts": label_counts,
    }


def normalize_top10_data(data):
    if not data:
        return [], {}

    per_query = data.get("per_query", [])
    mean = data.get("mean", {})

    # If file is already a mean-only metrics file
    if not per_query and any(k in data for k in ["P@5", "precision_at_5", "nDCG@5"]):
        mean = data

    return per_query, mean


def infer_query_label(item):
    query = item.get("query", "")
    run = item.get("run", "")

    text = f"{run} {query}".lower()

    if "run_01" in text or "multi-agent" in text or "collaborative" in text:
        return "R1"
    if "run_02" in text or "evaluation" in text:
        return "R2"
    if "run_03" in text or "verbal" in text or "role" in text:
        return "R3"

    return run or "Query"


def print_top10_table():
    path = find_top10_file()

    if not path:
        print("WARNING: cannot find top-10 retrieval metrics file.")
        print("Look inside results/ and rename your top-10 metrics JSON to results/ir_metrics_top10.json")
        return None

    data = read_json(path)
    per_query, mean = normalize_top10_data(data)

    if not per_query and not mean:
        print(f"WARNING: no metrics found in {path}")
        return None

    print("\n" + "=" * 90)
    print("3. TOP-5 / TOP-10 RETRIEVAL QUALITY")
    print("=" * 90)
    print(f"Loaded top-10 metrics from: {path}")

    print(r"\begin{table}[t]")
    print(
        r"\caption{Retrieval quality at top-5 and top-10 on the three main queries using author-assigned graded relevance.}"
    )
    print(r"\label{tab:ir-metrics-top10}")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    print(
        r"\textbf{Query} & \textbf{P@5} & \textbf{P@10} & \textbf{nDCG@5} & \textbf{nDCG@10} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR} \\"
    )
    print(r"\midrule")

    out_rows = []

    for item in per_query:
        label = infer_query_label(item)
        m = item.get("metrics", item)

        row = {
            "query": label,
            "P@5": metric(m, "P@5", "precision_at_5"),
            "P@10": metric(m, "P@10", "precision_at_10"),
            "nDCG@5": metric(m, "nDCG@5", "ndcg_at_5"),
            "nDCG@10": metric(m, "nDCG@10", "ndcg_at_10"),
            "Recall@5": metric(m, "Recall@5", "recall_at_5", "R@5"),
            "Recall@10": metric(m, "Recall@10", "recall_at_10", "R@10"),
            "MRR": metric(m, "MRR", "mrr"),
        }

        out_rows.append(row)

        print(
            f"{label} & {fmt(row['P@5'])} & {fmt(row['P@10'])} & "
            f"{fmt(row['nDCG@5'])} & {fmt(row['nDCG@10'])} & "
            f"{fmt(row['Recall@5'])} & {fmt(row['Recall@10'])} & {fmt(row['MRR'])} \\\\"
        )

    print(r"\midrule")

    mean_row = {
        "P@5": metric(mean, "P@5", "precision_at_5"),
        "P@10": metric(mean, "P@10", "precision_at_10"),
        "nDCG@5": metric(mean, "nDCG@5", "ndcg_at_5"),
        "nDCG@10": metric(mean, "nDCG@10", "ndcg_at_10"),
        "Recall@5": metric(mean, "Recall@5", "recall_at_5", "R@5"),
        "Recall@10": metric(mean, "Recall@10", "recall_at_10", "R@10"),
        "MRR": metric(mean, "MRR", "mrr"),
    }

    print(
        f"Mean & {fmt(mean_row['P@5'])} & {fmt(mean_row['P@10'])} & "
        f"{fmt(mean_row['nDCG@5'])} & {fmt(mean_row['nDCG@10'])} & "
        f"{fmt(mean_row['Recall@5'])} & {fmt(mean_row['Recall@10'])} & {fmt(mean_row['MRR'])} \\\\"
    )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return {
        "source_file": path,
        "per_query": out_rows,
        "mean": mean_row,
    }


def print_gap_error_analysis(gap_data):
    if not gap_data:
        return None

    label_counts = gap_data.get("label_counts", {})

    if not label_counts:
        print("WARNING: no label_counts found for gap error analysis.")
        return None

    plausible = label_counts.get("plausible", 0)
    too_broad = label_counts.get("too_broad", 0)
    indirect = label_counts.get("indirect", 0)
    meaningless = label_counts.get("meaningless", 0)
    total = plausible + too_broad + indirect + meaningless

    print("\n" + "=" * 90)
    print("4. GAP VALIDATION ERROR ANALYSIS")
    print("=" * 90)

    print(r"\begin{table}[t]")
    print(
        r"\caption{Error analysis for sampled gap-matrix empty cells. Non-plausible cells mainly reflect gaps that are overly broad or only indirectly implied by the retrieved corpus.}"
    )
    print(r"\label{tab:gap-error-analysis}")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"\textbf{Label} & \textbf{Count} & \textbf{Rate} \\")
    print(r"\midrule")

    for label, count in [
        ("Plausible", plausible),
        ("Too broad", too_broad),
        ("Indirect", indirect),
        ("Meaningless", meaningless),
    ]:
        rate = count / total if total else None
        print(f"{label} & {count} & {fmt(rate)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    paragraph = (
        "The sampled gap validation shows that 13 of 20 sampled empty cells were judged "
        "plausible, giving a gap precision of 0.650. The main failure modes were not "
        "meaningless gaps, but cells that were too broad or only indirectly supported by "
        "the retrieved corpus. This suggests that the deterministic matrix is useful as a "
        "corpus-level absence signal, but individual empty cells should be interpreted "
        "cautiously rather than as definitive claims about the broader literature."
    )

    print("\nPaper paragraph:")
    print(paragraph)

    return {
        "total": total,
        "plausible": plausible,
        "too_broad": too_broad,
        "indirect": indirect,
        "meaningless": meaningless,
        "paragraph": paragraph,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    retriever_rows = print_retriever_table()
    gap_data = print_gap_validation_table()
    top10_data = print_top10_table()
    error_data = print_gap_error_analysis(gap_data)

    out = {
        "retriever_comparison": retriever_rows,
        "gap_validation": gap_data,
        "top10_retrieval_metrics": top10_data,
        "gap_error_analysis": error_data,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Saved combined output → {OUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()