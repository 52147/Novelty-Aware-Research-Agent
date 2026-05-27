"""
query_analyzer.py — Decomposes a user query into focused retrieval intents.

Temperature: 0.3 (moderate-low — needs some creativity for decomposition,
but outputs are structured via JSON schema constrained decoding).
"""

from llm_client import LLMClient
from schemas import QueryAnalysis, RetrievalIntent

# ── JSON schema for constrained decoding ─────────────────────────────────────
QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "original_query": {"type": "string"},
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent":          {"type": "string"},
                    "domain":          {"type": "string"},
                    "comparison_axis": {"type": "string"},
                },
                "required": ["intent", "domain", "comparison_axis"],
                "additionalProperties": False,
            },
        },
        "reformulated_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["original_query", "intents", "reformulated_queries"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a research query analyzer for an academic paper comparison system.
Decompose the user's question into specific retrieval intents and 2-4 reformulated
search queries optimized for dense vector retrieval.
Be precise and extract only the core technical concepts."""


class QueryAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
        user_prompt = f"""Analyze this research query:

"{query}"

Identify:
1. Core retrieval intents (what to search for)
2. Domain (e.g., multi-agent LLMs, RAG, reasoning)
3. Comparison axis (methods / benchmarks / architectures / results)
4. 2-4 reformulated queries for vector similarity search

Return as JSON following the schema."""

        result = self.llm.complete_json(
            SYSTEM_PROMPT,
            user_prompt,
            component="query_analyzer",
            schema=QUERY_ANALYSIS_SCHEMA,
        )

        return QueryAnalysis(
            original_query=result["original_query"],
            intents=[RetrievalIntent(**i) for i in result["intents"]],
            reformulated_queries=result["reformulated_queries"],
        )
