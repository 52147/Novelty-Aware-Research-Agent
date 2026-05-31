#!/usr/bin/env python3
"""
deterministic_gap_matrix.py

Builds a deterministic problem × method gap matrix from extracted contribution records.

Input:
    results/run_*.json

Output:
    results/deterministic_gap_matrix.csv
    results/deterministic_gap_matrix.md
"""

import os
import json
import glob
import csv
from collections import defaultdict

RESULTS_DIR = "results"
OUT_CSV = os.path.join(RESULTS_DIR, "deterministic_gap_matrix.csv")
OUT_MD = os.path.join(RESULTS_DIR, "deterministic_gap_matrix.md")


def normalize_problem(text: str) -> str:
    t = (text or "").lower()

    if any(x in t for x in ["evaluation", "benchmark", "assess", "measure"]):
        return "Agent evaluation"
    if any(x in t for x in ["memory", "context", "long-term", "episodic"]):
        return "Agent memory / context management"
    if any(x in t for x in ["multi-agent", "collaboration", "coordination", "role"]):
        return "Multi-agent coordination"
    if any(x in t for x in ["reasoning", "chain-of-thought", "planning", "decomposition"]):
        return "Agent reasoning"
    if any(x in t for x in ["retrieval", "rag", "search", "document"]):
        return "Retrieval-augmented generation"
    if any(x in t for x in ["tool", "api", "environment", "web"]):
        return "Tool use / environment interaction"
    if any(x in t for x in ["self-improvement", "reflection", "reinforcement", "feedback"]):
        return "Agent self-improvement"

    return "Other / mixed problem"


def normalize_method(text: str) -> str:
    t = (text or "").lower()

    if any(x in t for x in ["benchmark", "evaluation suite", "testbed"]):
        return "Benchmarking"
    if any(x in t for x in ["retrieval", "rag", "index", "search"]):
        return "Retrieval"
    if any(x in t for x in ["reflection", "verbal reinforcement", "self-reflection"]):
        return "Verbal reflection"
    if any(x in t for x in ["role", "inception", "multi-agent", "conversation"]):
        return "Role-playing / multi-agent prompting"
    if any(x in t for x in ["memory", "planning", "stream"]):
        return "Memory + planning"
    if any(x in t for x in ["tool", "api", "function", "environment"]):
        return "Tool use"
    if any(x in t for x in ["chain-of-thought", "reasoning", "decomposition"]):
        return "Reasoning prompt"
    if any(x in t for x in ["schema", "structured", "json"]):
        return "Structured extraction"

    return "Other / mixed method"


def extract_records_from_json(obj):
    records = []

    def walk(x):
        if isinstance(x, dict):
            keys = {k.lower(): k for k in x.keys()}

            problem_key = None
            method_key = None
            paper_key = None

            for k in x.keys():
                kl = k.lower()
                if "problem" in kl:
                    problem_key = k
                if "method" in kl or "proposed" in kl:
                    method_key = k
                if "paper" in kl or "title" in kl or "id" in kl:
                    paper_key = k

            if problem_key and method_key:
                records.append({
                    "paper": str(x.get(paper_key, "unknown")),
                    "problem": str(x.get(problem_key, "")),
                    "method": str(x.get(method_key, "")),
                })

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return records


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "run*.json")) + glob.glob(os.path.join(RESULTS_DIR, "*run*.json")))

    if not files:
        print("No run JSON files found in results/.")
        print("Expected files like results/run1.json, results/run2.json, results/run3.json")
        return

    all_records = []

    for path in files:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            records = extract_records_from_json(obj)
            for r in records:
                r["source_file"] = os.path.basename(path)
            all_records.extend(records)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not all_records:
        print("No extraction records found.")
        print("Your JSON may use different field names. Paste one run JSON if this happens.")
        return

    cells = defaultdict(list)
    problems = set()
    methods = set()

    for r in all_records:
        p = normalize_problem(r["problem"])
        m = normalize_method(r["method"])
        problems.add(p)
        methods.add(m)
        cells[(p, m)].append(r["paper"])

    problems = sorted(problems)
    methods = sorted(methods)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem / Method"] + methods)
        for p in problems:
            row = [p]
            for m in methods:
                papers = cells.get((p, m), [])
                row.append("; ".join(papers) if papers else "—")
            writer.writerow(row)

    with open(OUT_MD, "w") as f:
        f.write("| Problem / Method | " + " | ".join(methods) + " |\n")
        f.write("|---" + "|---" * len(methods) + "|\n")
        for p in problems:
            row = [p]
            for m in methods:
                papers = cells.get((p, m), [])
                row.append(", ".join(papers) if papers else "—")
            f.write("| " + " | ".join(row) + " |\n")

    total_cells = len(problems) * len(methods)
    filled = sum(1 for key in cells if cells[key])
    empty = total_cells - filled

    print("Deterministic gap matrix built.")
    print(f"Records used: {len(all_records)}")
    print(f"Problem categories: {len(problems)}")
    print(f"Method categories: {len(methods)}")
    print(f"Filled cells: {filled}")
    print(f"Empty cells / corpus-level gaps: {empty}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
