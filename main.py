"""
main.py — CLI entry point for the Novelty-Aware Research Agent.

Usage examples:

  # Step 1: create sample corpus
  python create_sample_corpus.py

  # Step 2: build FAISS index from papers JSON
  python main.py --build-corpus sample_papers.json

  # Step 3: run a query
  python main.py --query "Compare multi-agent LLM frameworks for collaborative reasoning"

  # Run with custom settings and save output
  python main.py \\
      --query  "What methods exist for verbal reinforcement in LLM agents?" \\
      --top-k  20 \\
      --top-n  5 \\
      --output results/run_01.json

  # Export human evaluation bundle alongside results
  python main.py --query "..." --eval-bundle eval_bundle.json
"""

import os
import json
import argparse
from pathlib import Path

from pipeline        import NoveltyAwareResearchAgent
from corpus_builder  import CorpusBuilder
from evaluate        import PipelineEvaluator


# ── Corpus building ───────────────────────────────────────────────────────────

def build_corpus(papers_file: str, corpus_path: str = "corpus") -> None:
    with open(papers_file) as f:
        papers = json.load(f)

    builder = CorpusBuilder()
    builder.build_index(papers, save_path=corpus_path)

    # Save full texts for the extractor
    texts = {}
    for p in papers:
        texts[p["paper_id"]] = " ".join(filter(None, [
            p.get("abstract",     ""),
            p.get("introduction", ""),
            p.get("conclusion",   ""),
        ]))

    with open(os.path.join(corpus_path, "paper_texts.json"), "w") as f:
        json.dump(texts, f, indent=2)

    print(f"\nCorpus ready: {len(papers)} papers indexed at '{corpus_path}/'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novelty-Aware Research Agent — CSC 792 Final Project",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--build-corpus",  type=str,            help="Path to papers JSON file; builds FAISS index")
    parser.add_argument("--corpus-path",   type=str, default="corpus", help="Directory for FAISS index (default: corpus/)")
    parser.add_argument("--query",         type=str,            help="Research comparison query")
    parser.add_argument("--top-k",         type=int, default=15, help="Candidate papers to retrieve (default: 15)")
    parser.add_argument("--top-n",         type=int, default=5,  help="Papers to analyze after ranking (default: 5)")
    parser.add_argument("--model",         type=str, default="gpt-4o", help="LLM model (default: gpt-4o)")
    parser.add_argument("--output",        type=str,            help="Save JSON report to this path")
    parser.add_argument("--eval-bundle",   type=str,            help="Export human eval bundle to this path")
    args = parser.parse_args()

    # ── Build corpus mode ─────────────────────────────────────────────────────
    if args.build_corpus:
        build_corpus(args.build_corpus, args.corpus_path)
        return

    # ── Query mode ────────────────────────────────────────────────────────────
    query = args.query or "Compare recent multi-agent LLM frameworks for collaborative reasoning"

    agent = NoveltyAwareResearchAgent(
            corpus_path    = args.corpus_path,
            openai_api_key = os.getenv("OPENAI_API_KEY"),
            model          = args.model,
        )

    report, metadata = agent.run(
        query          = query,
        top_k_retrieve = args.top_k,
        top_n_rank     = args.top_n,
    )

    agent.print_report(report)

    # ── Save JSON output ──────────────────────────────────────────────────────
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {"report": report.model_dump(), "metadata": metadata},
                f, indent=2,
            )
        print(f"Report saved → {args.output}")

    # ── Export human eval bundle ──────────────────────────────────────────────
    if args.eval_bundle:
        from schemas import PaperContribution, ComparisonResult, OverlapRecord, DifferenceRecord, GapRecord

        # Reconstruct contribution and comparison objects from report for evaluator
        # (In a full run these would be passed directly through the pipeline)
        evaluator = PipelineEvaluator()

        # Build minimal contribution objects from per-paper summaries
        # (full objects available if pipeline is instrumented to return them)
        print(f"\nNote: For full extraction rubrics, instrument pipeline.py to "
              f"return contributions and comparison objects directly.")
        print(f"Exporting comparison rubric and gap list to {args.eval_bundle}…")

        bundle = {
            "query":            query,
            "per_paper_summaries": report.per_paper_summaries,
            "comparison_rubric": {
                "overlaps":    [o.model_dump() for o in report.overlaps],
                "differences": [d.model_dump() for d in report.differences],
                "gaps":        [g.model_dump() for g in report.gaps],
            },
            "instructions": {
                "scale": "1=Poor  2=Fair  3=Good  4=Very Good  5=Excellent",
            },
        }
        with open(args.eval_bundle, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"Eval bundle saved → {args.eval_bundle}")


if __name__ == "__main__":
    main()
