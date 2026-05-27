"""
comparison_agent.py — Three-pass reasoning over structured extraction records.

The three passes operate ONLY on PaperContribution records, not raw text.
This constrains the reasoning space and makes each pass auditable.

  Pass 1 — Overlap:         shared problem / dataset / method family
  Pass 2 — Differentiation: what each paper does differently
  Pass 3 — Gap matrix:      problem × method combinations absent from the corpus

Temperature: 0.2 for all three passes (structured reasoning, low variance).
Each pass uses its own constrained JSON schema (course slide technique).
"""

from llm_client import LLMClient
from schemas import (
    PaperContribution,
    ComparisonResult,
    OverlapRecord,
    DifferenceRecord,
    GapRecord,
)
from typing import List

# ── JSON schemas for each pass ────────────────────────────────────────────────

OVERLAP_SCHEMA = {
    "type": "object",
    "properties": {
        "overlaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_ids":      {"type": "array", "items": {"type": "string"}},
                    "shared_element": {"type": "string"},
                    "description":    {"type": "string"},
                },
                "required": ["paper_ids", "shared_element", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["overlaps"],
    "additionalProperties": False,
}

DIFFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "differences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id":               {"type": "string"},
                    "title":                  {"type": "string"},
                    "differentiating_aspect": {"type": "string"},
                    "description":            {"type": "string"},
                },
                "required": ["paper_id", "title", "differentiating_aspect", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["differences"],
    "additionalProperties": False,
}

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "gap_matrix": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem_formulation": {"type": "string"},
                    "missing_method":      {"type": "string"},
                    "description":         {"type": "string"},
                    "supporting_evidence": {"type": "string"},
                },
                "required": [
                    "problem_formulation", "missing_method",
                    "description", "supporting_evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["gap_matrix"],
    "additionalProperties": False,
}


class ComparisonAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ── Public interface ──────────────────────────────────────────────────────

    def compare(self, contributions: List[PaperContribution]) -> ComparisonResult:
        records = self._format_records(contributions)

        print("    Pass 1 — Overlap detection…")
        overlaps     = self._overlap_pass(records)

        print("    Pass 2 — Differentiation analysis…")
        differences  = self._differentiation_pass(records)

        print("    Pass 3 — Gap matrix construction…")
        gaps         = self._gap_pass(records)

        return ComparisonResult(
            overlaps=overlaps,
            differences=differences,
            gaps=gaps,
        )

    # ── Record formatting ─────────────────────────────────────────────────────

    @staticmethod
    def _format_records(contributions: List[PaperContribution]) -> str:
        blocks = []
        for c in contributions:
            blocks.append(
                f"--- [{c.paper_id}] {c.title} ({c.year}) ---\n"
                f"Problem:      {c.problem_statement}\n"
                f"Method:       {c.proposed_method}\n"
                f"Contribution: {c.key_contribution}\n"
                f"Novelty:      {c.claimed_novelty}"
            )
        return "\n\n".join(blocks)

    # ── Pass 1: Overlap ───────────────────────────────────────────────────────

    def _overlap_pass(self, records: str) -> List[OverlapRecord]:
        system = (
            "You are performing an Overlap Pass on structured paper contribution records.\n"
            "Identify groups of papers sharing the same problem formulation, dataset, or method family.\n"
            "Operate ONLY on the structured records — do not infer beyond what is stated."
        )
        user = (
            f"Structured records:\n{records}\n\n"
            "Identify all meaningful overlaps. For each overlap: list the paper IDs, "
            "the shared element (problem / dataset / method family), and a brief description."
        )
        result = self.llm.complete_json(system, user, component="comparison", schema=OVERLAP_SCHEMA)
        return [OverlapRecord(**o) for o in result["overlaps"]]

    # ── Pass 2: Differentiation ───────────────────────────────────────────────

    def _differentiation_pass(self, records: str) -> List[DifferenceRecord]:
        system = (
            "You are performing a Differentiation Pass on structured paper contribution records.\n"
            "For each paper identify what makes it distinctly different from the others.\n"
            "Focus on: method, scope, evaluation approach, or claimed contribution.\n"
            "Operate ONLY on the structured records."
        )
        user = (
            f"Structured records:\n{records}\n\n"
            "For each paper, identify its key differentiating aspect and describe it briefly."
        )
        result = self.llm.complete_json(system, user, component="comparison", schema=DIFFERENCE_SCHEMA)
        return [DifferenceRecord(**d) for d in result["differences"]]

    # ── Pass 3: Gap matrix ────────────────────────────────────────────────────

    def _gap_pass(self, records: str) -> List[GapRecord]:
        system = (
            "You are performing a Gap Pass on structured paper contribution records.\n"
            "Construct a conceptual problem × method matrix from the records.\n"
            "Identify combinations of problem formulation and methodological approach "
            "that are NOT represented in the retrieved set.\n"
            "Frame gaps as observations within THIS CORPUS ONLY — "
            "not claims about the broader research landscape."
        )
        user = (
            f"Structured records:\n{records}\n\n"
            "Step 1: List distinct problem formulations across all papers.\n"
            "Step 2: List distinct methodological approaches across all papers.\n"
            "Step 3: Identify problem × method combinations MISSING from the corpus.\n"
            "Step 4: For each gap, state the problem, the missing method type, "
            "a description, and the evidence from the records that supports this gap.\n\n"
            "Only report gaps clearly evidenced by the records."
        )
        result = self.llm.complete_json(system, user, component="comparison", schema=GAP_SCHEMA)
        return [GapRecord(**g) for g in result["gap_matrix"]]
