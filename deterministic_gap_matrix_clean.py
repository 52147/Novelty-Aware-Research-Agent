#!/usr/bin/env python3
"""
deterministic_gap_matrix_clean.py

Clean deterministic problem × method matrix using only run_01, run_02, run_03.
It maps each retrieved paper_id to a fixed problem category and method category.
No LLM gap generation is used.
"""

import os
import json
import csv
from collections import defaultdict

RESULTS_DIR = "results"
RUN_FILES = [
    "results/run_01.json",
    "results/run_02.json",
    "results/run_03.json",
]

OUT_CSV = "results/deterministic_gap_matrix_clean.csv"
OUT_MD = "results/deterministic_gap_matrix_clean.md"
OUT_JSON = "results/deterministic_gap_matrix_clean_stats.json"

PAPER_LABELS = {
    "metagpt_2023": {
        "problem": "Multi-agent coordination",
        "method": "Role-based multi-agent framework",
    },
    "agentverse_2023": {
        "problem": "Multi-agent coordination",
        "method": "Multi-agent collaboration framework",
    },
    "autogen_2023": {
        "problem": "Multi-agent coordination",
        "method": "Conversational multi-agent framework",
    },
    "agentsurvey_2023": {
        "problem": "Agent survey / taxonomy",
        "method": "Survey and taxonomy",
    },
    "agentbench_2023": {
        "problem": "Agent evaluation",
        "method": "Benchmarking",
    },
    "cot_2022": {
        "problem": "Agent reasoning",
        "method": "Reasoning prompting",
    },
    "react_2022": {
        "problem": "Reasoning-action interaction",
        "method": "ReAct reasoning-action loop",
    },
    "rewoo_2023": {
        "problem": "Reasoning-action efficiency",
        "method": "Decoupled reasoning from observation",
    },
    "reflexion_2023": {
        "problem": "Agent self-improvement",
        "method": "Verbal reflection",
    },
    "camel_2023": {
        "problem": "Multi-agent coordination",
        "method": "Role-playing / inception prompting",
    },
    "metaagents_2023": {
        "problem": "Multi-agent coordination",
        "method": "Simulation-based multi-agent coordination",
    },
    "generativeagents_2023": {
        "problem": "Human behavior simulation",
        "method": "Memory + reflection + planning",
    },
}


def load_paper_ids():
    paper_ids = []

    for path in RUN_FILES:
        with open(path) as f:
            data = json.load(f)

        summaries = data["report"]["per_paper_summaries"]

        for item in summaries:
            pid = item["paper_id"]
            if pid not in paper_ids:
                paper_ids.append(pid)

    return paper_ids


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    paper_ids = load_paper_ids()

    missing = [pid for pid in paper_ids if pid not in PAPER_LABELS]
    if missing:
        print("Missing labels for paper_ids:")
        for pid in missing:
            print("  ", pid)
        raise SystemExit("Add these paper_ids to PAPER_LABELS and rerun.")

    cells = defaultdict(list)

    for pid in paper_ids:
        problem = PAPER_LABELS[pid]["problem"]
        method = PAPER_LABELS[pid]["method"]
        cells[(problem, method)].append(pid)

    problems = sorted(set(PAPER_LABELS[pid]["problem"] for pid in paper_ids))
    methods = sorted(set(PAPER_LABELS[pid]["method"] for pid in paper_ids))

    total_cells = len(problems) * len(methods)
    filled_cells = sum(1 for p in problems for m in methods if cells.get((p, m)))
    empty_cells = total_cells - filled_cells

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem / Method"] + methods)
        for p in problems:
            row = [p]
            for m in methods:
                row.append(", ".join(cells[(p, m)]) if cells.get((p, m)) else "—")
            writer.writerow(row)

    with open(OUT_MD, "w") as f:
        f.write("| Problem / Method | " + " | ".join(methods) + " |\n")
        f.write("|---" + "|---" * len(methods) + "|\n")
        for p in problems:
            row = [p]
            for m in methods:
                row.append(", ".join(cells[(p, m)]) if cells.get((p, m)) else "—")
            f.write("| " + " | ".join(row) + " |\n")

    stats = {
        "run_files": RUN_FILES,
        "unique_papers": len(paper_ids),
        "problem_categories": len(problems),
        "method_categories": len(methods),
        "total_cells": total_cells,
        "filled_cells": filled_cells,
        "empty_cells_corpus_level_gaps": empty_cells,
        "matrix_density": filled_cells / total_cells if total_cells else 0,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print("Clean deterministic gap matrix built.")
    print(f"Unique papers used: {len(paper_ids)}")
    print(f"Problem categories: {len(problems)}")
    print(f"Method categories: {len(methods)}")
    print(f"Filled cells: {filled_cells}")
    print(f"Empty cells / corpus-level gaps: {empty_cells}")
    print(f"Matrix density: {stats['matrix_density']:.3f}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
