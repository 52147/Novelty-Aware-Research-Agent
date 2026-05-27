"""
schemas.py — Pydantic data models for every pipeline stage.
All inter-component contracts are typed here.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Query Analyzer ────────────────────────────────────────────────────────────

class RetrievalIntent(BaseModel):
    intent: str
    domain: str
    comparison_axis: str


class QueryAnalysis(BaseModel):
    original_query: str
    intents: List[RetrievalIntent]
    reformulated_queries: List[str]


# ── Corpus / Retriever ────────────────────────────────────────────────────────

class PaperChunk(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    chunk_text: str
    chunk_type: str  # "abstract" | "introduction" | "conclusion"


class RetrievedPaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    similarity_score: float
    relevance_score: Optional[float] = None


# ── ReAct ─────────────────────────────────────────────────────────────────────

class ReActStep(BaseModel):
    thought: str
    action: str       # "CONTINUE" | "REFINE" | "STOP"
    observation: str
    iteration: int


# ── Extractor ─────────────────────────────────────────────────────────────────

class PaperContribution(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    problem_statement: str
    proposed_method: str
    key_contribution: str
    claimed_novelty: str


# ── Comparison Agent ──────────────────────────────────────────────────────────

class OverlapRecord(BaseModel):
    paper_ids: List[str]
    shared_element: str   # "problem" | "dataset" | "method family"
    description: str


class DifferenceRecord(BaseModel):
    paper_id: str
    title: str
    differentiating_aspect: str
    description: str


class GapRecord(BaseModel):
    problem_formulation: str
    missing_method: str
    description: str
    supporting_evidence: str


class ComparisonResult(BaseModel):
    overlaps: List[OverlapRecord]
    differences: List[DifferenceRecord]
    gaps: List[GapRecord]


# ── Final Report ──────────────────────────────────────────────────────────────

class FinalReport(BaseModel):
    query: str
    papers_analyzed: int
    per_paper_summaries: List[Dict[str, Any]]
    overlaps: List[OverlapRecord]
    differences: List[DifferenceRecord]
    gaps: List[GapRecord]
    synthesis: str
    citations: List[str]
