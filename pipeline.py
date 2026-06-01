"""
pipeline.py — End-to-end orchestration of all six components.

Architecture (proposal §3):
  Query Analyzer
    → [ReAct] Retriever          ← ReAct loop (Agent Frameworks slides)
    → Ranker
    → [Constrained] Extractor    ← JSON schema decoding (LLM slides)
    → Comparison Agent (3 passes)
    → Answer Generator           ← T=0.7 prose (LLM slides)
"""

import json
import time
import os
from typing import Optional, Tuple, Dict

from llm_client       import LLMClient
from corpus_builder   import CorpusBuilder
from query_analyzer   import QueryAnalyzer
from retriever        import Retriever
from ranker           import Ranker
from extractor        import ContributionExtractor
from comparison_agent import ComparisonAgent
from answer_generator import AnswerGenerator
from schemas          import FinalReport


class NoveltyAwareResearchAgent:
    """
    Novelty-Aware Research Agent
    Proposal: CSC 792 Final Project — Shou-Tzu Han

    Run with:
        agent = NoveltyAwareResearchAgent(corpus_path="corpus")
        report, meta = agent.run("Compare recent multi-agent LLM papers")
        agent.print_report(report)
    """

    def __init__(
        self,
        corpus_path:    str = "corpus",
        openai_api_key: Optional[str] = None,
        model:          str = "gpt-4o",
    ):
        print("="*60)
        print("  Novelty-Aware Research Agent — initializing")
        print("="*60)

        self.llm    = LLMClient(api_key=openai_api_key, model=model)
        self.corpus = CorpusBuilder()
        self.corpus.load_index(corpus_path)

        self.query_analyzer   = QueryAnalyzer(self.llm)
        self.retriever        = Retriever(self.corpus, self.llm)
        self.ranker           = Ranker(self.llm)
        self.extractor        = ContributionExtractor(self.llm)
        self.comparison_agent = ComparisonAgent(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)

        self.paper_texts: Dict[str, str] = self._load_paper_texts(corpus_path)
        print("\nAgent ready.\n")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        query:          str,
        top_k_retrieve: int = 15,
        top_n_rank:     int = 5,
    ) -> Tuple[FinalReport, dict]:
        """Run all six pipeline stages and return (report, metadata)."""

        t0   = time.time()
        meta: dict = {"query": query, "model": self.llm.model, "stages": {}}

        # ── 1. Query Analyzer ─────────────────────────────────────────────────
        print(f"\n{'─'*55}")
        print(f"[1/6] Query Analyzer  (T={0.3})")
        print(f"  Input: \"{query}\"")
        analysis = self.query_analyzer.analyze(query)
        print(f"  → {len(analysis.intents)} intent(s), "
              f"{len(analysis.reformulated_queries)} reformulated queries")
        meta["stages"]["query_analysis"] = analysis.model_dump()

        # ── 2. Retriever (ReAct) ──────────────────────────────────────────────
        print(f"\n[2/6] Retriever  (ReAct loop, max 3 iterations)")
        retrieved, react_log = self.retriever.retrieve(
            query                = query,
            reformulated_queries = analysis.reformulated_queries,
            top_k                = top_k_retrieve,
        )
        print(f"\n  → {len(retrieved)} candidate papers after ReAct")
        meta["stages"]["retrieval"] = {
            "n_candidates":   len(retrieved),
            "react_steps":    len(react_log),
            "react_log":      [s.model_dump() for s in react_log],
        }

        if not retrieved:
            raise ValueError(
                "No papers retrieved. Verify corpus path and query terms."
            )

        # ── 3. Ranker ─────────────────────────────────────────────────────────
        print(f"\n[3/6] Ranker  (T={0.1})")
        ranked = self.ranker.rank(query, retrieved, top_n=top_n_rank)
        print(f"  → Top {len(ranked)} papers selected:")
        for p in ranked:
            print(f"     [{p.relevance_score:4.1f}] {p.title[:60]}")
        meta["stages"]["ranking"] = {"selected": [p.paper_id for p in ranked]}

        # ── 4. Contribution Extractor ─────────────────────────────────────────
        print(f"\n[4/6] Contribution Extractor  (constrained JSON schema, T={0.1})")
        contributions = self.extractor.extract(ranked, self.paper_texts)
        compliance    = self.extractor.validate_schema_compliance(contributions)
        print(f"  → Schema compliance: "
              f"{compliance['compliance_rate']:.0%} "
              f"({compliance['compliant']}/{compliance['total']})")
        meta["stages"]["extraction"] = {"compliance": compliance}

        # ── 5. Comparison Agent ───────────────────────────────────────────────
        print(f"\n[5/6] Comparison Agent  (three-pass, T={0.2})")
        comparison = self.comparison_agent.compare(contributions)
        print(f"  → Overlaps: {len(comparison.overlaps)}"
              f"  |  Differences: {len(comparison.differences)}"
              f"  |  Gaps: {len(comparison.gaps)}")
        meta["stages"]["comparison"] = {
            "n_overlaps":     len(comparison.overlaps),
            "n_differences":  len(comparison.differences),
            "n_gaps":         len(comparison.gaps),
        }

        # ── 6. Answer Generator ───────────────────────────────────────────────
        print(f"\n[6/6] Answer Generator  (T={0.7})")
        report = self.answer_generator.generate(query, contributions, comparison)

        elapsed = round(time.time() - t0, 2)
        meta["elapsed_seconds"] = elapsed
        print(f"  → Report generated in {elapsed}s")

        return report, meta

    # ── Formatted output ──────────────────────────────────────────────────────

    def print_report(self, report: FinalReport) -> None:
        W = 68
        print(f"\n{'='*W}")
        print("  NOVELTY-AWARE RESEARCH AGENT — FINAL REPORT")
        print(f"{'='*W}")
        print(f"  Query:           {report.query}")
        print(f"  Papers analyzed: {report.papers_analyzed}")

        print(f"\n{'─'*W}")
        print("  PER-PAPER SUMMARIES")
        print(f"{'─'*W}")
        for s in report.per_paper_summaries:
            print(f"\n  [{s['paper_id']}] {s['title']}  ({int(s['year'])})")
            print(f"    {s['one_sentence_summary']}")

        print(f"\n{'─'*W}")
        print("  OVERLAPS")
        print(f"{'─'*W}")
        if report.overlaps:
            for o in report.overlaps:
                print(f"\n  Shared: {o.shared_element}")
                print(f"    {o.description}")
                print(f"    Papers: {', '.join(o.paper_ids)}")
        else:
            print("  None identified.")

        print(f"\n{'─'*W}")
        print("  DIFFERENTIATING ASPECTS")
        print(f"{'─'*W}")
        if report.differences:
            for d in report.differences:
                print(f"\n  [{d.paper_id}] {d.differentiating_aspect}")
                print(f"    {d.description}")
        else:
            print("  None identified.")

        print(f"\n{'─'*W}")
        print("  POTENTIAL GAPS  (within this corpus only)")
        print(f"{'─'*W}")
        if report.gaps:
            for g in report.gaps:
                print(f"\n  Problem:        {g.problem_formulation}")
                print(f"  Missing method: {g.missing_method}")
                print(f"  Note:           {g.description}")
                print(f"  Evidence:       {g.supporting_evidence}")
        else:
            print("  None identified.")

        print(f"\n{'─'*W}")
        print("  SYNTHESIS")
        print(f"{'─'*W}")
        print(f"\n  {report.synthesis}")

        print(f"\n{'─'*W}")
        print("  CITATIONS")
        print(f"{'─'*W}")
        for c in report.citations:
            print(f"  {c}")

        print(f"\n{'='*W}\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_paper_texts(corpus_path: str) -> Dict[str, str]:
        path = os.path.join(corpus_path, "paper_texts.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}
