#!/usr/bin/env python3
"""
run_7_queries.py — extend evaluation from 3 to 10 queries (R4–R10).

Wired to the real repo interface:
  - pipeline.NoveltyAwareResearchAgent(corpus_path=...).run(query, top_k_retrieve, top_n_rank)
  - relevance_labels.json schema (queries -> run_file -> {query, relevance})
  - ir_metrics defs match exp2_metrics_at_k.py (graded nDCG: (2^g - 1)/log2(i+2))

USAGE
-----
    export OPENAI_API_KEY="sk-..."
    python run_7_queries.py

WHAT YOU MUST DO FIRST
----------------------
Fill in graded relevance labels for the 7 new queries in
relevance_labels_new7.json (a template is auto-created on first run).
Grades: 3=highly relevant, 2=relevant, 1=marginal, 0=not relevant.
Use the SAME paper-id strings as the corpus (e.g. "rewoo_2023") — see the
_corpus_reference block in relevance_labels.json for the full id list.

Until a query is labeled, it still reports top-5 / schema / overlap / gap /
runtime, but its P/nDCG/Recall/MRR are shown as n/a.

OUTPUT
------
    results/run_R4.json ... run_R10.json   (full metadata per query)
    results/metrics_10q.json               (machine-readable)
    paste-ready table to stdout
"""

import os, json, math, statistics
from pipeline import NoveltyAwareResearchAgent

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus")
NEW_LABELS_FILE = "relevance_labels_new7.json"
TOP_K_RETRIEVE = 15      # same default as pipeline.run
TOP_N_RANK     = 5       # paper uses top-5

# ── The 7 new queries (R4–R10). Edit freely; keep them answerable by the corpus.
NEW_QUERIES = {
    "R4":  "Compare tool-use and function-calling approaches in LLM agents",
    "R5":  "What memory mechanisms do LLM agents use for long-horizon tasks?",
    "R6":  "Compare planning strategies in LLM-based autonomous agents",
    "R7":  "How is retrieval-augmented generation evaluated for factuality?",
    "R8":  "Compare single-agent and multi-agent problem-solving architectures",
    "R9":  "What methods reduce hallucination in retrieval-augmented systems?",
    "R10": "Compare reflection and self-critique mechanisms in LLM agents",
}

# ── metric defs (identical to exp2_metrics_at_k.py) ───────────────────────────
def precision_at_k(ranked, rel, k):
    topk = ranked[:k]
    return sum(1 for pid in topk if rel.get(pid, 0) > 0) / k if topk else 0.0

def recall_at_k(ranked, rel, k):
    total = sum(1 for g in rel.values() if g > 0)
    if total == 0: return None
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

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0

# ── load or scaffold the new-query labels ─────────────────────────────────────
def load_new_labels():
    if os.path.exists(NEW_LABELS_FILE):
        with open(NEW_LABELS_FILE) as f:
            return json.load(f)
    template = {
        "_instructions": "Grade each paper per query. 3=highly,2=relevant,1=marginal,0=not. Use corpus paper-id strings (see relevance_labels.json _corpus_reference).",
        "queries": {qid: {"query": q, "relevance": {}} for qid, q in NEW_QUERIES.items()},
    }
    with open(NEW_LABELS_FILE, "w") as f:
        json.dump(template, f, indent=2)
    print(f"\n>>> Created {NEW_LABELS_FILE}. Fill in 'relevance' for each query, then re-run.")
    print(">>> (Script will still run now and show top-5 so you know what to grade.)\n")
    return template

def main():
    new_labels = load_new_labels()
    agent = NoveltyAwareResearchAgent(corpus_path=CORPUS_PATH)

    os.makedirs("results", exist_ok=True)
    rows = []
    top5_by_q = {}

    for qid, query in NEW_QUERIES.items():
        report, meta = agent.run(query, top_k_retrieve=TOP_K_RETRIEVE, top_n_rank=TOP_N_RANK)

        with open(f"results/run_{qid}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        ranked_ids = meta["stages"]["ranking"]["selected"]
        top5_by_q[qid] = ranked_ids
        compliance = meta["stages"]["extraction"]["compliance"]["compliance_rate"]
        cmp = meta["stages"]["comparison"]
        runtime = meta["elapsed_seconds"]

        rel = {k: int(v) for k, v in
            new_labels["queries"].get(qid, {}).get("relevance", {}).items()
            if v is not None}
        if rel:
            p5  = precision_at_k(ranked_ids, rel, 5)
            nd5 = ndcg_at_k(ranked_ids, rel, 5)
            r5  = recall_at_k(ranked_ids, rel, 5)
            rr  = mrr(ranked_ids, rel)
            ungraded = [pid for pid in ranked_ids if pid not in rel]
        else:
            p5 = nd5 = r5 = rr = None
            ungraded = ranked_ids

        rows.append(dict(qid=qid, query=query, top5=ranked_ids,
                         P5=p5, nDCG5=nd5, R5=r5, MRR=rr,
                         schema=compliance, overlaps=cmp["n_overlaps"],
                         diffs=cmp["n_differences"], gaps=cmp["n_gaps"],
                         runtime=runtime, ungraded=ungraded))

    # ── paste-ready table ─────────────────────────────────────────────────────
    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else " n/a "
    print("\n" + "="*92)
    print("PASTE THIS BACK  —  new queries R4–R10 (top-5 ranking, author labels)")
    print("="*92)
    hdr = f"{'Q':<5}{'P@5':>7}{'nDCG@5':>9}{'R@5':>7}{'MRR':>7}{'Schema':>9}{'Ov':>4}{'Df':>4}{'Gp':>4}{'sec':>7}  top5"
    print(hdr); print("-"*len(hdr))
    for r in rows:
        print(f"{r['qid']:<5}{fmt(r['P5']):>7}{fmt(r['nDCG5']):>9}{fmt(r['R5']):>7}"
              f"{fmt(r['MRR']):>7}{r['schema']*100:>8.1f}%{r['overlaps']:>4}{r['diffs']:>4}"
              f"{r['gaps']:>4}{r['runtime']:>7.1f}  {r['top5']}")

    labeled = [r for r in rows if r["P5"] is not None]
    if labeled:
        m = lambda k: statistics.mean(r[k] for r in labeled)
        print("-"*len(hdr))
        print(f"{'MEAN':<5}{m('P5'):>7.3f}{m('nDCG5'):>9.3f}{m('R5'):>7.3f}{m('MRR'):>7.3f}"
              f"  (over {len(labeled)} labeled new queries)")

    # ── ungraded warnings ─────────────────────────────────────────────────────
    ung = [(r['qid'], pid) for r in rows for pid in r['ungraded']]
    if ung:
        print("\nUNGRADED papers in a top-5 (add to relevance_labels_new7.json, treated as 0):")
        for qid, pid in ung:
            print(f"   {qid}: {pid}")

    # ── full 10-query Jaccard (R1–R3 hardcoded from your existing labels) ──────
    R1 = ["agentverse_2023","autogen_2023","agentsurvey_2023","agentbench_2023","openagents_2023"]
    R2 = ["agentbench_2023","agentsurvey_2023","art_2023","cot_2022","rap_2023"]
    R3 = ["reflexion_2023","camel_2023","agentverse_2023","generativeagents_2023","innermono_2022"]
    all_q = {"R1":R1,"R2":R2,"R3":R3, **top5_by_q}
    qids = list(all_q)
    pairs = [(qids[i],qids[j]) for i in range(len(qids)) for j in range(i+1,len(qids))]
    jvals = [jaccard(all_q[a], all_q[b]) for a,b in pairs]
    from collections import Counter
    paper_counts = Counter(pid for ids in all_q.values() for pid in ids)
    in_all = [p for p,c in paper_counts.items() if c == len(all_q)]
    n_slots = sum(len(ids) for ids in all_q.values())
    distinct = len(paper_counts)
    print("\n" + "="*60)
    print("FULL 10-QUERY QUERY-SENSITIVITY")
    print("="*60)
    print(f"  mean pairwise Jaccard (all 10) = {statistics.mean(jvals):.3f}")
    print(f"  distinct papers across {n_slots} slots = {distinct}")
    print(f"  query-exclusive papers = {sum(1 for c in paper_counts.values() if c==1)}")
    print(f"  papers appearing in ALL 10 = {len(in_all)}  {in_all}")

    with open("results/metrics_10q.json", "w") as f:
        json.dump({"rows": rows, "jaccard_mean_10q": statistics.mean(jvals),
                   "distinct_papers": distinct}, f, indent=2, default=str)
    print("\nSaved → results/metrics_10q.json")

if __name__ == "__main__":
    main()
