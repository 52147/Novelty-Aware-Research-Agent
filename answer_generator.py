"""
answer_generator.py — Synthesizes the final citation-grounded report.

Temperature: 0.7 (moderate — natural, readable prose for the synthesis section).
Course slide rule of thumb (p.38): "Want both accuracy and creativity? ~0.7"
Still uses constrained JSON schema for per-paper summaries and citations
so the structured fields remain machine-parseable.
"""

from llm_client import LLMClient
from schemas import PaperContribution, ComparisonResult, FinalReport
from typing import List

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "per_paper_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id":             {"type": "string"},
                    "title":                {"type": "string"},
                    "year":                 {"type": "number"},
                    "one_sentence_summary": {"type": "string"},
                },
                "required": ["paper_id", "title", "year", "one_sentence_summary"],
                "additionalProperties": False,
            },
        },
        "synthesis":  {"type": "string"},
        "citations":  {"type": "array", "items": {"type": "string"}},
    },
    "required": ["per_paper_summaries", "synthesis", "citations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a research synthesis assistant.
Write clearly and concisely. Ground every claim in the provided records.
The synthesis paragraph should be 3-5 sentences of natural academic prose —
summarising what the paper set collectively shows, where they converge,
and what the gap analysis suggests within this corpus."""


class AnswerGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        query: str,
        contributions: List[PaperContribution],
        comparison: ComparisonResult,
    ) -> FinalReport:
        overlaps_txt = "\n".join(
            f"  • {o.shared_element}: {o.description}  (papers: {', '.join(o.paper_ids)})"
            for o in comparison.overlaps
        ) or "  None identified."

        diffs_txt = "\n".join(
            f"  • [{d.paper_id}] {d.differentiating_aspect}: {d.description}"
            for d in comparison.differences
        ) or "  None identified."

        gaps_txt = "\n".join(
            f"  • Problem '{g.problem_formulation}' × Missing '{g.missing_method}': {g.description}"
            for g in comparison.gaps
        ) or "  None identified."

        papers_txt = "\n".join(
            f"  [{c.paper_id}] {', '.join(c.authors[:2])} et al. {c.title} ({c.year}): {c.key_contribution}"
            for c in contributions
        )

        user_prompt = f"""Generate a structured research comparison report.

QUERY: "{query}"

PAPERS ANALYZED ({len(contributions)}):
{papers_txt}

OVERLAPS:
{overlaps_txt}

DIFFERENTIATING ASPECTS:
{diffs_txt}

POTENTIAL GAPS (within this corpus only):
{gaps_txt}

Produce:
1. A one-sentence summary for each paper.
2. A synthesis paragraph (3-5 sentences) grounded in the records.
3. A citation list in format: [paper_id] Authors (year). Title."""

        result = self.llm.complete_json(
            SYSTEM_PROMPT,
            user_prompt,
            component="generator",
            schema=REPORT_SCHEMA,
        )

        return FinalReport(
            query               = query,
            papers_analyzed     = len(contributions),
            per_paper_summaries = result["per_paper_summaries"],
            overlaps            = comparison.overlaps,
            differences         = comparison.differences,
            gaps                = comparison.gaps,
            synthesis           = result["synthesis"],
            citations           = result["citations"],
        )
