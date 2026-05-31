#!/usr/bin/env python3
import json

PATH = "results/retriever_comparison.json"

def fmt(x):
    return "--" if x is None else f"{float(x):.3f}"

def label(run):
    if "run_01" in run:
        return "R1"
    if "run_02" in run:
        return "R2"
    if "run_03" in run:
        return "R3"
    return run

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(r"\begin{table*}[t]")
print(r"\caption{Per-query retriever comparison across BM25, dense retrieval, hybrid retrieval, and the full ranker.}")
print(r"\label{tab:retriever-comparison-per-query}")
print(r"\begin{tabular}{llrrrrrr}")
print(r"\toprule")
print(r"\textbf{Query} & \textbf{System} & \textbf{P@5} & \textbf{P@10} & \textbf{nDCG@5} & \textbf{nDCG@10} & \textbf{R@5} & \textbf{R@10} \\")
print(r"\midrule")

for item in data["per_query"]:
    q = label(item["run"])
    for system in ["BM25", "Dense", "Hybrid", "Full ranker"]:
        m = item["systems"][system]["metrics"]
        print(
            f"{q} & {system} & "
            f"{fmt(m.get('P@5'))} & {fmt(m.get('P@10'))} & "
            f"{fmt(m.get('nDCG@5'))} & {fmt(m.get('nDCG@10'))} & "
            f"{fmt(m.get('Recall@5'))} & {fmt(m.get('Recall@10'))} \\\\"
        )
    print(r"\addlinespace")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")