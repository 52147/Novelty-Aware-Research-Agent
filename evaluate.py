"""
evaluate.py — All five evaluation metrics from proposal §5.

Metric 1: Retrieval Relevance   → Precision@5 (automated)
Metric 2: Extraction Quality    → Human rubric template (JSON export)
Metric 3: Schema Compliance     → Automated % (run via extractor)
Metric 4: Comparison Accuracy   → Human rubric template (JSON export)
Metric 5: Report Usefulness     → Blind Likert side-by-side template

Usage:
    evaluator = PipelineEvaluator()

    # Automated metrics
    p5 = evaluator.precision_at_k(retrieved_ids, relevant_ids, k=5)
    sc = evaluator.schema_compliance(contributions)

    # Human eval bundle (export to JSON for manual scoring)
    evaluator.export_human_eval_bundle(contributions, comparison, query, "eval_bundle.json")
"""

import json
from typing import List, Dict, Optional

from schemas import PaperContribution, ComparisonResult, FinalReport


class PipelineEvaluator:

    # ── Metric 1: Retrieval Relevance ─────────────────────────────────────────

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids:  List[str],
        k:             int = 5,
    ) -> float:
        """
        Precision@k: fraction of top-k retrieved papers that are relevant.
        Relevant set must be manually labeled for each test query.
        """
        top_k = retrieved_ids[:k]
        hits  = sum(1 for pid in top_k if pid in relevant_ids)
        return round(hits / k, 4)

    # ── Metric 3: Schema Compliance (automated) ───────────────────────────────

    def schema_compliance(
        self,
        contributions: List[PaperContribution],
    ) -> Dict:
        """
        Automated check: all four fields are non-empty and not placeholder.
        Also surfaced via ContributionExtractor.validate_schema_compliance().
        """
        total = len(contributions)
        ok    = 0
        bad   = []
        PLACEHOLDER = "not specified in provided text."

        for c in contributions:
            fields = [
                c.problem_statement,
                c.proposed_method,
                c.key_contribution,
                c.claimed_novelty,
            ]
            if all(f and len(f.strip()) > 10 and f.strip().lower() != PLACEHOLDER for f in fields):
                ok += 1
            else:
                bad.append(c.paper_id)

        return {
            "total":             total,
            "compliant":         ok,
            "compliance_rate":   round(ok / total, 3) if total else 0.0,
            "non_compliant_ids": bad,
        }

    # ── Metric 2: Extraction Quality rubric ───────────────────────────────────

    def extraction_rubric(self, contribution: PaperContribution) -> Dict:
        """Human-evaluation rubric template for one paper (1-5 scale)."""
        return {
            "paper_id": contribution.paper_id,
            "title":    contribution.title,
            "fields": {
                "problem_statement": {
                    "extracted": contribution.problem_statement,
                    "score_1_5": None,
                    "criterion": "Does it accurately capture the problem the paper solves?",
                    "notes":     "",
                },
                "proposed_method": {
                    "extracted": contribution.proposed_method,
                    "score_1_5": None,
                    "criterion": "Does it accurately describe the technical approach?",
                    "notes":     "",
                },
                "key_contribution": {
                    "extracted": contribution.key_contribution,
                    "score_1_5": None,
                    "criterion": "Does it correctly identify the main contribution?",
                    "notes":     "",
                },
                "claimed_novelty": {
                    "extracted": contribution.claimed_novelty,
                    "score_1_5": None,
                    "criterion": "Does it faithfully represent what the authors claim is new?",
                    "notes":     "",
                },
            },
            "overall_score_1_5": None,
            "overall_notes":     "",
        }

    # ── Metric 4: Comparison Accuracy rubric ─────────────────────────────────

    def comparison_rubric(
        self,
        comparison: ComparisonResult,
        query:      str,
    ) -> Dict:
        """Human-evaluation rubric for comparison quality (1-5 scale)."""
        return {
            "query": query,
            "overlaps": {
                "identified":           [o.model_dump() for o in comparison.overlaps],
                "correctness_1_5":      None,
                "completeness_1_5":     None,
                "notes":                "",
            },
            "differences": {
                "identified":           [d.model_dump() for d in comparison.differences],
                "correctness_1_5":      None,
                "completeness_1_5":     None,
                "notes":                "",
            },
            "gaps": {
                "identified":           [g.model_dump() for g in comparison.gaps],
                "plausibility_1_5":     None,
                "evidence_quality_1_5": None,
                "notes":                "",
            },
        }

    # ── Metric 5: Report Usefulness — blind side-by-side ─────────────────────

    def likert_comparison_template(
        self,
        query:           str,
        system_report:   str,
        baseline_report: str,
        randomize:       bool = True,
    ) -> Dict:
        """
        Blind side-by-side preference template (5-point Likert).
        Evaluator should not know which is system vs. baseline.
        Set randomize=True to swap A/B assignment randomly.
        """
        import random
        if randomize and random.random() < 0.5:
            a, b = baseline_report, system_report
            labels = {"A": "baseline", "B": "system"}
        else:
            a, b   = system_report, baseline_report
            labels = {"A": "system", "B": "baseline"}

        return {
            "query":           query,
            "report_A":        a,
            "report_B":        b,
            "_key":            labels,   # reveal only after scoring
            "preference":      None,     # "A" or "B"
            "usefulness_A_1_5": None,
            "usefulness_B_1_5": None,
            "notes":           "",
            "scale":           "1=Poor  2=Fair  3=Good  4=Very Good  5=Excellent",
        }

    # ── Bundle export ─────────────────────────────────────────────────────────

    def export_human_eval_bundle(
        self,
        contributions: List[PaperContribution],
        comparison:    ComparisonResult,
        query:         str,
        output_path:   str = "human_eval_bundle.json",
    ) -> None:
        bundle = {
            "query":               query,
            "instructions": {
                "scale": "1=Poor  2=Fair  3=Good  4=Very Good  5=Excellent",
                "note":  "Score each field independently then give an overall score.",
            },
            "extraction_rubrics":  [self.extraction_rubric(c)   for c in contributions],
            "comparison_rubric":    self.comparison_rubric(comparison, query),
        }
        with open(output_path, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"Human evaluation bundle saved → {output_path}")

    # ── Aggregate automated metrics ───────────────────────────────────────────

    def automated_summary(
        self,
        retrieved_ids:     List[str],
        relevant_ids:      List[str],
        contributions:     List[PaperContribution],
    ) -> Dict:
        return {
            "precision_at_5":    self.precision_at_k(retrieved_ids, relevant_ids, k=5),
            "schema_compliance": self.schema_compliance(contributions),
        }
