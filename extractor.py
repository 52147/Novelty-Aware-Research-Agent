"""
extractor.py — Extracts four structured fields from each paper.

Course technique: JSON / Schema-Guided Decoding (LLM slides, pp. 44-52)
  "The decoder only allows tokens that keep the structure valid."
  Temperature 0.1 → precision, deterministic, no hallucinated fields.

The four-field schema enforces:
  problem_statement  | proposed_method | key_contribution | claimed_novelty
All fields are required and non-nullable — schema compliance is enforced
at the token level via OpenAI's response_format: json_schema.

Evaluation metric this supports: Schema Compliance (automated %)
"""

from llm_client import LLMClient
from schemas import RetrievedPaper, PaperContribution
from typing import List, Dict

# ── Constrained decoding schema ───────────────────────────────────────────────
# Course slide p.50: define schema → model output constrained to match it.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_id": {"type": "string"},
        "title":    {"type": "string"},
        "problem_statement": {
            "type":        "string",
            "description": "The specific research problem this paper addresses.",
        },
        "proposed_method": {
            "type":        "string",
            "description": "The technical approach or method the authors propose.",
        },
        "key_contribution": {
            "type":        "string",
            "description": "The main contribution this paper makes to the field.",
        },
        "claimed_novelty": {
            "type":        "string",
            "description": "What the authors explicitly claim is new vs. prior work.",
        },
    },
    "required": [
        "paper_id", "title",
        "problem_statement", "proposed_method",
        "key_contribution",  "claimed_novelty",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a research paper contribution extractor.
Extract exactly four fields from the provided paper text.
Be precise and faithful — do not invent claims beyond what the text states.
If a field truly cannot be determined, write: "Not specified in provided text." """


class ContributionExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ── Batch extraction ──────────────────────────────────────────────────────

    def extract(
        self,
        papers: List[RetrievedPaper],
        paper_texts: Dict[str, str],
    ) -> List[PaperContribution]:
        """
        Extract contributions for every paper.
        Uses constrained decoding — every output is guaranteed to have all four fields.
        """
        contributions = []
        for paper in papers:
            full_text = paper_texts.get(paper.paper_id, paper.abstract)
            contrib   = self._extract_single(paper, full_text)
            contributions.append(contrib)
            print(f"    ✓ [{paper.paper_id}] {paper.title[:65]}")
        return contributions

    # ── Single extraction ─────────────────────────────────────────────────────

    def _extract_single(self, paper: RetrievedPaper, text: str) -> PaperContribution:
        user_prompt = f"""Paper ID:  {paper.paper_id}
Title:     {paper.title}
Authors:   {', '.join(paper.authors)}
Year:      {paper.year}

Text:
{text[:3500]}

Extract the four contribution fields:
1. problem_statement  — what specific problem does this paper address?
2. proposed_method    — what technical approach do the authors propose?
3. key_contribution   — what is the main contribution to the field?
4. claimed_novelty    — what do the authors explicitly claim is new?"""

        result = self.llm.complete_json(
            SYSTEM_PROMPT,
            user_prompt,
            component="extractor",
            schema=EXTRACTION_SCHEMA,
        )

        return PaperContribution(
            paper_id          = paper.paper_id,
            title             = paper.title,
            authors           = paper.authors,
            year              = paper.year,
            problem_statement = result["problem_statement"],
            proposed_method   = result["proposed_method"],
            key_contribution  = result["key_contribution"],
            claimed_novelty   = result["claimed_novelty"],
        )

    # ── Automated Schema Compliance metric ────────────────────────────────────
    # Evaluation section 5 of proposal: "percentage of papers where all four
    # fields are non-empty and pass a format validation check."

    def validate_schema_compliance(
        self,
        contributions: List[PaperContribution],
    ) -> Dict:
        total     = len(contributions)
        compliant = 0
        issues    = []
        PLACEHOLDER = "not specified in provided text."

        for c in contributions:
            fields = [
                c.problem_statement,
                c.proposed_method,
                c.key_contribution,
                c.claimed_novelty,
            ]
            ok = all(
                f and len(f.strip()) > 10 and f.strip().lower() != PLACEHOLDER
                for f in fields
            )
            if ok:
                compliant += 1
            else:
                issues.append(c.paper_id)

        return {
            "total":            total,
            "compliant":        compliant,
            "compliance_rate":  round(compliant / total, 3) if total else 0.0,
            "non_compliant_ids": issues,
        }
