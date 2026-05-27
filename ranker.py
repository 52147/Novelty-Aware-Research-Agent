"""
ranker.py — LLM-based relevance scoring and top-N selection.

Temperature: 0.1 (very low — this is factual scoring, not generation).
Uses constrained JSON output to guarantee well-formed ranking records.
"""

from llm_client import LLMClient
from schemas import RetrievedPaper
from typing import List

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id":        {"type": "string"},
                    "relevance_score": {"type": "number"},
                    "reason":          {"type": "string"},
                },
                "required": ["paper_id", "relevance_score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a relevance scoring system for academic papers.
Score each paper 0-10 based on how closely it matches the research query.
Criteria: topical alignment, methodological relevance, recency.
Be consistent — use the full 0-10 range."""


class Ranker:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rank(
        self,
        query: str,
        papers: List[RetrievedPaper],
        top_n: int = 5,
    ) -> List[RetrievedPaper]:
        if not papers:
            return []

        papers_text = "\n\n".join(
            f"Paper ID: {p.paper_id}\n"
            f"Title:    {p.title}\n"
            f"Year:     {p.year}\n"
            f"Abstract: {p.abstract[:400]}"
            for p in papers
        )

        user_prompt = (
            f'Query: "{query}"\n\n'
            f"Papers to score (0-10 relevance):\n{papers_text}"
        )

        result = self.llm.complete_json(
            SYSTEM_PROMPT,
            user_prompt,
            component="ranker",
            schema=RANKING_SCHEMA,
        )

        score_map = {r["paper_id"]: r["relevance_score"] for r in result["rankings"]}
        for p in papers:
            p.relevance_score = score_map.get(p.paper_id, 0.0)

        ranked = sorted(papers, key=lambda p: p.relevance_score or 0.0, reverse=True)
        return ranked[:top_n]
