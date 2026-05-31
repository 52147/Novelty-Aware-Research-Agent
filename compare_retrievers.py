

#!/usr/bin/env python3
"""
compare_retrievers.py

Compare retrieval variants for the Novelty-Aware Research Agent:

1. BM25 lexical retrieval
2. Dense retrieval from existing run output / FAISS-style ranked order
3. Hybrid BM25 + Dense
4. Full system ranker output

Outputs:
- results/retriever_comparison.json
- LaTeX table printed to terminal

Expected project files:
- sample_papers.json
- relevance_labels.json
- results/run_01.json
- results/run_02.json
- results/run_03.json

Optional files:
- results/baseline_run_01.json, baseline_run_02.json, baseline_run_03.json
  If present, these are used as Dense/raw similarity order.
  If absent, the script falls back to run metadata candidates/top-ranked papers.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
CORPUS_FILE = "sample_papers.json"
LABELS_FILE = "relevance_labels.json"
RUN_FILES = ["run_01.json", "run_02.json", "run_03.json"]
BASELINE_FILES = {
    "run_01.json": "baseline_run_01.json",
    "run_02.json": "baseline_run_02.json",
    "run_03.json": "baseline_run_03.json",
}
CUTS = [5, 10]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus() -> Dict[str, Dict]:
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"Missing {CORPUS_FILE}")
    papers = load_json(CORPUS_FILE)
    return {p["paper_id"]: p for p in papers}


def paper_text(p: Dict) -> str:
    parts = [
        p.get("title", ""),
        " ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors", "")),
        str(p.get("year", "")),
        p.get("abstract", ""),
        p.get("introduction", ""),
        p.get("conclusion", ""),
        p.get("chunk_text", ""),
    ]
    # Some corpora store three chunks as separate records; some store only title + chunk_text.
    return "\n".join(x for x in parts if x)


def load_run(run_file: str) -> Tuple[str, List[str]]:
    path = os.path.join(RESULTS_DIR, run_file)
    data = load_json(path)
    report = data.get("report", data)
    query = report.get("query", data.get("metadata", {}).get("query", ""))
    ranked = [p["paper_id"] for p in report.get("per_paper_summaries", [])]
    return query, ranked


def load_labels() -> Dict[str, Dict]:
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Missing {LABELS_FILE}. Run ir_metrics.py --init and fill labels first.")
    return load_json(LABELS_FILE).get("queries", {})


# ── Tokenization / BM25 ──────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25:
    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = {pid: tokenize(txt) for pid, txt in docs.items()}
        self.doc_len = {pid: len(toks) for pid, toks in self.doc_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(len(self.doc_len), 1)
        self.tf = {pid: Counter(toks) for pid, toks in self.doc_tokens.items()}
        self.df = defaultdict(int)
        for toks in self.doc_tokens.values():
            for t in set(toks):
                self.df[t] += 1
        self.N = len(self.doc_tokens)

    def idf(self, term: str) -> float:
        # Standard BM25 idf with smoothing.
        return math.log(1 + (self.N - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))

    def score(self, query: str, pid: str) -> float:
        q_terms = tokenize(query)
        score = 0.0
        dl = self.doc_len.get(pid, 0)
        if dl == 0:
            return 0.0
        for term in q_terms:
            f = self.tf[pid].get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, topn: int = 10) -> List[str]:
        scored = [(pid, self.score(query, pid)) for pid in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored[:topn]]


# ── Dense / hybrid / full system ranking ─────────────────────────────────────

def load_baseline_dense_order(run_file: str) -> Optional[List[str]]:
    baseline_name = BASELINE_FILES.get(run_file)
    if not baseline_name:
        return None
    path = os.path.join(RESULTS_DIR, baseline_name)
    if not os.path.exists(path):
        return None
    data = load_json(path)
    if "papers_retrieved" in data:
        return data["papers_retrieved"]
    report = data.get("report", data)
    if "per_paper_summaries" in report:
        return [p["paper_id"] for p in report["per_paper_summaries"]]
    return None


def full_ranker_order(run_file: str) -> List[str]:
    _, ranked = load_run(run_file)
    return ranked


def dense_order(run_file: str, fallback_full: List[str]) -> List[str]:
    order = load_baseline_dense_order(run_file)
    if order:
        return order
    return fallback_full


def normalize_scores(order: List[str]) -> Dict[str, float]:
    # Higher score for earlier rank. Top=1.0, lower ranks decay linearly.
    if not order:
        return {}
    n = len(order)
    return {pid: (n - i) / n for i, pid in enumerate(order)}


def hybrid_order(bm25_order: List[str], dense_order_: List[str], all_pids: List[str], topn: int = 10) -> List[str]:
    b = normalize_scores(bm25_order)
    d = normalize_scores(dense_order_)
    scored = []
    for pid in all_pids:
        score = 0.5 * b.get(pid, 0.0) + 0.5 * d.get(pid, 0.0)
        scored.append((pid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:topn]]


# ── Metrics ──────────────────────────────────────────────────────────────────

def dcg(grades: List[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def precision_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    topk = ranked[:k]
    if len(topk) < k:
        topk = topk + ["__missing__"] * (k - len(topk))
    return sum(1 for pid in topk if rel.get(pid, 0) > 0) / k


def recall_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> Optional[float]:
    total = sum(1 for g in rel.values() if int(g) > 0)
    if total == 0:
        return None
    hits = sum(1 for pid in ranked[:k] if rel.get(pid, 0) > 0)
    return hits / total


def ndcg_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> Optional[float]:
    gains = [int(rel.get(pid, 0)) for pid in ranked[:k]]
    if len(gains) < k:
        gains += [0] * (k - len(gains))
    ideal = sorted([int(v) for v in rel.values()], reverse=True)[:k]
    if len(ideal) < k:
        ideal += [0] * (k - len(ideal))
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg(gains) / idcg


def mrr(ranked: List[str], rel: Dict[str, int]) -> float:
    for i, pid in enumerate(ranked):
        if rel.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(order: List[str], rel: Dict[str, int]) -> Dict[str, Optional[float]]:
    out = {}
    for k in CUTS:
        out[f"P@{k}"] = precision_at_k(order, rel, k)
        out[f"nDCG@{k}"] = ndcg_at_k(order, rel, k)
        out[f"Recall@{k}"] = recall_at_k(order, rel, k)
    out["MRR"] = mrr(order, rel)
    return out


def avg(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.3f}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    corpus = load_corpus()
    docs = {pid: paper_text(p) for pid, p in corpus.items()}
    bm25 = BM25(docs)
    labels = load_labels()
    all_pids = list(corpus.keys())

    systems = ["BM25", "Dense", "Hybrid", "Full ranker"]
    per_query = []
    aggregate = {s: defaultdict(list) for s in systems}

    for run_file in RUN_FILES:
        query, full_order = load_run(run_file)
        if run_file not in labels:
            print(f"WARNING: no labels for {run_file}; skipping")
            continue
        rel = {pid: int(g) for pid, g in labels[run_file].get("relevance", {}).items()}

        bm25_ranked = bm25.rank(query, topn=10)
        dense_ranked = dense_order(run_file, full_order)[:10]
        full_ranked = full_order[:10]
        hybrid_ranked = hybrid_order(bm25_ranked, dense_ranked, all_pids, topn=10)

        orders = {
            "BM25": bm25_ranked,
            "Dense": dense_ranked,
            "Hybrid": hybrid_ranked,
            "Full ranker": full_ranked,
        }

        row = {"run": run_file, "query": query, "systems": {}}
        for system, order in orders.items():
            metrics = compute_metrics(order, rel)
            row["systems"][system] = {"order": order, "metrics": metrics}
            for m, v in metrics.items():
                aggregate[system][m].append(v)
        per_query.append(row)

    mean = {}
    for system in systems:
        mean[system] = {m: avg(vals) for m, vals in aggregate[system].items()}

    out = {"per_query": per_query, "mean": mean}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "retriever_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved →", out_path)
    print("\n" + "=" * 92)
    print("RETRIEVER COMPARISON — MEAN OVER THREE QUERIES")
    print("=" * 92)
    header = f"{'System':18}{'P@5':>8}{'P@10':>8}{'nDCG@5':>10}{'nDCG@10':>11}{'Recall@5':>11}{'Recall@10':>12}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for system in systems:
        m = mean[system]
        print(
            f"{system:18}"
            f"{fmt(m.get('P@5')):>8}"
            f"{fmt(m.get('P@10')):>8}"
            f"{fmt(m.get('nDCG@5')):>10}"
            f"{fmt(m.get('nDCG@10')):>11}"
            f"{fmt(m.get('Recall@5')):>11}"
            f"{fmt(m.get('Recall@10')):>12}"
            f"{fmt(m.get('MRR')):>8}"
        )

    print("\n" + "=" * 92)
    print("LATEX TABLE")
    print("=" * 92)
    print(r"\begin{table}[t]")
    print(r"\caption{Retriever comparison across BM25, dense retrieval, hybrid retrieval, and the full ranker. Metrics are averaged over the three main queries using author-assigned graded relevance.}")
    print(r"\label{tab:retriever-comparison}")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"\textbf{System} & \textbf{P@5} & \textbf{P@10} & \textbf{nDCG@5} & \textbf{nDCG@10} & \textbf{R@5} & \textbf{R@10} \\")
    print(r"\midrule")
    latex_names = {
        "BM25": "BM25",
        "Dense": "Dense",
        "Hybrid": "Hybrid",
        "Full ranker": "Full ranker",
    }
    for system in systems:
        m = mean[system]
        print(
            f"{latex_names[system]} & {fmt(m.get('P@5'))} & {fmt(m.get('P@10'))} & "
            f"{fmt(m.get('nDCG@5'))} & {fmt(m.get('nDCG@10'))} & "
            f"{fmt(m.get('Recall@5'))} & {fmt(m.get('Recall@10'))} \\\\" 
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()