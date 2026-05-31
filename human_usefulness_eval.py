

#!/usr/bin/env python3
"""
human_usefulness_eval.py

Create and score a small human usefulness evaluation comparing Basic RAG vs the
Full Novelty-Aware Research Agent outputs.

Workflow
--------
1) Generate the evaluation CSV:

    python human_usefulness_eval.py --init

This writes human_eval_form.csv. Each row is one query-system pair. Fill the
score columns manually using a 1--5 scale.

2) Fill these columns:

    relevance_score
    comparison_usefulness_score
    gap_usefulness_score
    structure_clarity_score
    overall_usefulness_score

Optional:

    preferred_system
    notes

Use preferred_system only if you want pairwise preference counts. Valid values:
Basic RAG, Full system, tie

3) Compute metrics:

    python human_usefulness_eval.py

Outputs:

    results/human_usefulness_metrics.json

The script also prints a LaTeX table.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

RESULTS_DIR = "results"
FORM_CSV = "human_eval_form.csv"
OUT_JSON = os.path.join(RESULTS_DIR, "human_usefulness_metrics.json")

RUN_FILES = [
    "results/run_01.json",
    "results/run_02.json",
    "results/run_03.json",
]

BASELINE_FILES = [
    "results/baseline_run_01.json",
    "results/baseline_run_02.json",
    "results/baseline_run_03.json",
]

SYSTEM_FULL = "Full system"
SYSTEM_BASE = "Basic RAG"
VALID_PREFS = {SYSTEM_FULL, SYSTEM_BASE, "tie", ""}

SCORE_FIELDS = [
    "relevance_score",
    "comparison_usefulness_score",
    "gap_usefulness_score",
    "structure_clarity_score",
    "overall_usefulness_score",
]

QUERY_LABELS = {
    "run_01.json": "R1 (multi-agent)",
    "run_02.json": "R2 (evaluation)",
    "run_03.json": "R3 (verbal RL)",
    "baseline_run_01.json": "R1 (multi-agent)",
    "baseline_run_02.json": "R2 (evaluation)",
    "baseline_run_03.json": "R3 (verbal RL)",
}


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short_text(text, limit=1200):
    text = str(text or "").replace("\r", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_full_output(data):
    report = data.get("report", data)
    query = report.get("query", "")

    parts = []

    summaries = report.get("per_paper_summaries", [])
    if summaries:
        parts.append("PER-PAPER SUMMARIES:")
        for p in summaries:
            parts.append(
                f"- {p.get('paper_id', '')}: {p.get('one_sentence_summary', '')}"
            )

    overlaps = report.get("overlaps", [])
    if overlaps:
        parts.append("\nOVERLAPS:")
        for o in overlaps:
            pids = ", ".join(o.get("paper_ids", []))
            parts.append(f"- [{pids}] {o.get('description', '')}")

    differences = report.get("differences", [])
    if differences:
        parts.append("\nDIFFERENCES:")
        for d in differences:
            parts.append(f"- {d.get('paper_id', '')}: {d.get('description', '')}")

    gaps = report.get("gaps", [])
    if gaps:
        parts.append("\nGAPS:")
        for g in gaps:
            parts.append(
                f"- Problem: {g.get('problem_formulation', '')}; "
                f"Missing method: {g.get('missing_method', '')}; "
                f"Description: {g.get('description', '')}"
            )

    if report.get("synthesis"):
        parts.append("\nSYNTHESIS:")
        parts.append(report.get("synthesis", ""))

    return query, short_text("\n".join(parts))


def extract_baseline_output(data):
    query = data.get("query", "")
    output = data.get("output", "")
    return query, short_text(output)


def init_form():
    rows = []

    for full_path, base_path in zip(RUN_FILES, BASELINE_FILES):
        full_data = read_json(full_path)
        base_data = read_json(base_path)

        if full_data is None:
            print(f"WARNING: missing {full_path}; skipping full-system row.")
        else:
            query, output = extract_full_output(full_data)
            rows.append(
                {
                    "query_id": QUERY_LABELS.get(os.path.basename(full_path), os.path.basename(full_path)),
                    "query": query,
                    "system": SYSTEM_FULL,
                    "output_excerpt": output,
                    "relevance_score": "",
                    "comparison_usefulness_score": "",
                    "gap_usefulness_score": "",
                    "structure_clarity_score": "",
                    "overall_usefulness_score": "",
                    "preferred_system": "",
                    "notes": "",
                }
            )

        if base_data is None:
            print(f"WARNING: missing {base_path}; skipping baseline row.")
        else:
            query, output = extract_baseline_output(base_data)
            rows.append(
                {
                    "query_id": QUERY_LABELS.get(os.path.basename(base_path), os.path.basename(base_path)),
                    "query": query,
                    "system": SYSTEM_BASE,
                    "output_excerpt": output,
                    "relevance_score": "",
                    "comparison_usefulness_score": "",
                    "gap_usefulness_score": "",
                    "structure_clarity_score": "",
                    "overall_usefulness_score": "",
                    "preferred_system": "",
                    "notes": "",
                }
            )

    if not rows:
        raise FileNotFoundError(
            "No run files found. Expected results/run_01.json ... and "
            "results/baseline_run_01.json ..."
        )

    with open(FORM_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote human evaluation form → {FORM_CSV}")
    print("Fill the five score columns using 1--5.")
    print("Optional preferred_system values: Basic RAG / Full system / tie")
    print("Then run: python human_usefulness_eval.py")


def parse_score(value, field, row_num):
    value = str(value).strip()
    if value == "":
        return None
    try:
        score = float(value)
    except ValueError:
        raise ValueError(f"Row {row_num}: {field} must be a number from 1 to 5.")
    if score < 1 or score > 5:
        raise ValueError(f"Row {row_num}: {field} must be between 1 and 5.")
    return score


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def load_completed_form(path=FORM_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Run `python human_usefulness_eval.py --init` first.")

    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            parsed = dict(row)
            for field in SCORE_FIELDS:
                parsed[field] = parse_score(row.get(field, ""), field, i)
            pref = str(row.get("preferred_system", "")).strip()
            if pref not in VALID_PREFS:
                raise ValueError(
                    f"Row {i}: preferred_system must be one of: Basic RAG, Full system, tie, or blank."
                )
            parsed["preferred_system"] = pref
            rows.append(parsed)
    return rows


def compute_metrics(path=FORM_CSV):
    ensure_results_dir()
    rows = load_completed_form(path)

    scored_rows = [
        r for r in rows
        if any(r[field] is not None for field in SCORE_FIELDS)
    ]
    if not scored_rows:
        raise ValueError("No scored rows found. Fill the score columns first.")

    by_system = defaultdict(list)
    for row in scored_rows:
        by_system[row["system"]].append(row)

    system_metrics = {}
    for system, items in by_system.items():
        system_metrics[system] = {"n_rows": len(items), "scores": {}}
        for field in SCORE_FIELDS:
            vals = [r[field] for r in items if r[field] is not None]
            system_metrics[system]["scores"][field] = {
                "mean": mean(vals),
                "std": sample_std(vals),
                "n": len(vals),
            }

    preference_counts = Counter(
        r["preferred_system"] for r in rows if r.get("preferred_system", "")
    )
    total_preferences = sum(preference_counts.values())
    preference_rates = {
        key: (value / total_preferences if total_preferences else None)
        for key, value in preference_counts.items()
    }

    pairwise_by_query = defaultdict(dict)
    for row in scored_rows:
        pairwise_by_query[row["query_id"]][row["system"]] = row

    pairwise_wins = Counter()
    pairwise_diffs = []
    for query_id, pair in pairwise_by_query.items():
        if SYSTEM_FULL not in pair or SYSTEM_BASE not in pair:
            continue
        full_score = pair[SYSTEM_FULL].get("overall_usefulness_score")
        base_score = pair[SYSTEM_BASE].get("overall_usefulness_score")
        if full_score is None or base_score is None:
            continue
        diff = full_score - base_score
        pairwise_diffs.append({"query_id": query_id, "full_minus_baseline": diff})
        if diff > 0:
            pairwise_wins[SYSTEM_FULL] += 1
        elif diff < 0:
            pairwise_wins[SYSTEM_BASE] += 1
        else:
            pairwise_wins["tie"] += 1

    out = {
        "n_scored_rows": len(scored_rows),
        "systems": system_metrics,
        "preference_counts": dict(preference_counts),
        "preference_rates": preference_rates,
        "pairwise_overall_wins": dict(pairwise_wins),
        "pairwise_overall_differences": pairwise_diffs,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 88)
    print("HUMAN USEFULNESS EVALUATION")
    print("=" * 88)
    print(f"Scored rows: {len(scored_rows)}")
    print(f"Saved → {OUT_JSON}")

    print("\nMean scores by system:")
    header = f"{'System':18}{'Relevance':>11}{'Comparison':>13}{'Gap':>9}{'Structure':>12}{'Overall':>10}"
    print(header)
    print("-" * len(header))
    for system in [SYSTEM_BASE, SYSTEM_FULL]:
        if system not in system_metrics:
            continue
        scores = system_metrics[system]["scores"]
        print(
            f"{system:18}"
            f"{fmt(scores['relevance_score']['mean']):>11}"
            f"{fmt(scores['comparison_usefulness_score']['mean']):>13}"
            f"{fmt(scores['gap_usefulness_score']['mean']):>9}"
            f"{fmt(scores['structure_clarity_score']['mean']):>12}"
            f"{fmt(scores['overall_usefulness_score']['mean']):>10}"
        )

    if pairwise_wins:
        print("\nPairwise overall wins by query:")
        for key in [SYSTEM_FULL, SYSTEM_BASE, "tie"]:
            print(f"  {key}: {pairwise_wins.get(key, 0)}")

    if preference_counts:
        print("\nExplicit preference counts:")
        for key, value in preference_counts.items():
            print(f"  {key}: {value}")

    print("\nLaTeX table:")
    print(r"\begin{table}[t]")
    print(r"\caption{Human usefulness evaluation comparing Basic RAG and the full system. Scores use a 1--5 scale.}")
    print(r"\label{tab:human-usefulness}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"\textbf{System} & \textbf{Rel.} & \textbf{Comp.} & \textbf{Gap} & \textbf{Struct.} & \textbf{Overall} \\")
    print(r"\midrule")
    for system in [SYSTEM_BASE, SYSTEM_FULL]:
        if system not in system_metrics:
            continue
        scores = system_metrics[system]["scores"]
        print(
            f"{system} & "
            f"{fmt(scores['relevance_score']['mean'])} & "
            f"{fmt(scores['comparison_usefulness_score']['mean'])} & "
            f"{fmt(scores['gap_usefulness_score']['mean'])} & "
            f"{fmt(scores['structure_clarity_score']['mean'])} & "
            f"{fmt(scores['overall_usefulness_score']['mean'])} \\\\" 
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Create human_eval_form.csv.")
    parser.add_argument("--form", default=FORM_CSV, help="Path to completed evaluation form.")
    args = parser.parse_args()

    if args.init:
        init_form()
    else:
        compute_metrics(args.form)


if __name__ == "__main__":
    main()