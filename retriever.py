"""
retriever.py — ReAct-style retrieval with iterative query refinement.

Course technique: ReAct (Agent Frameworks slides, pp. 2-10)
  Loop: Thought → Action → Observation → Thought → …
  Here: the agent evaluates whether retrieved papers are sufficient;
  if not, it reasons about why and rewrites the query before retrying.

Course technique: RAG Retrieval (RAG slides, pp. 7, 14-15)
  - Embeds query, runs FAISS top-k search
  - Deduplicates by paper_id, keeps best score per paper
"""

from llm_client import LLMClient
from corpus_builder import CorpusBuilder
from schemas import RetrievedPaper, ReActStep, PaperChunk
from typing import List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
L2_RELEVANCE_THRESHOLD = 400.0   # max L2 distance to consider a chunk relevant
MIN_PAPERS_REQUIRED    = 3       # minimum unique papers to accept the result
MAX_REACT_ITERATIONS   = 3       # hard cap on retrieval iterations

REACT_SYSTEM = """You are a retrieval reasoning agent.
Evaluate search results and decide whether to refine the query or stop.
Respond in this EXACT format (three lines):
Thought: <your reasoning about why results are insufficient>
Action: <REFINE or STOP>
Refinement: <a new search query if Action is REFINE, else leave empty>"""


class Retriever:
    def __init__(self, corpus: CorpusBuilder, llm: LLMClient):
        self.corpus = corpus
        self.llm    = llm

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        reformulated_queries: List[str],
        top_k: int = 15,
    ) -> Tuple[List[RetrievedPaper], List[ReActStep]]:
        """
        ReAct retrieval loop.

        Thought: Are the retrieved papers sufficient for comparison?
        Action:  STOP (if yes) | REFINE (if no)
        Observation: {n} unique papers retrieved with query '{q}'
        """
        react_log: List[ReActStep] = []
        query_pool = [query] + reformulated_queries
        query_idx  = 0
        best_candidates: List[Tuple[PaperChunk, float]] = []

        for iteration in range(MAX_REACT_ITERATIONS):
            current_query = query_pool[min(query_idx, len(query_pool) - 1)]

            # ── Action: vector search ─────────────────────────────────────────
            raw = self.corpus.search(current_query, top_k=top_k)

            # Filter by distance and deduplicate by paper_id
            candidates = self._deduplicate(
                [(c, d) for c, d in raw if d < L2_RELEVANCE_THRESHOLD]
            )

            # ── Thought + Action (via LLM if insufficient) ────────────────────
            thought, action, refinement = self._react_step(
                original_query=query,
                current_query=current_query,
                n_candidates=len(candidates),
                iteration=iteration,
            )

            observation = (
                f"Retrieved {len(candidates)} unique papers "
                f"(threshold L2<{L2_RELEVANCE_THRESHOLD}) "
                f"using query: '{current_query}'"
            )

            step = ReActStep(
                thought=thought,
                action=action,
                observation=observation,
                iteration=iteration + 1,
            )
            react_log.append(step)

            print(f"\n  [ReAct iter {iteration+1}]")
            print(f"    Thought:     {thought}")
            print(f"    Action:      {action}")
            print(f"    Observation: {observation}")

            # Keep best results seen so far
            if len(candidates) > len(best_candidates):
                best_candidates = candidates

            if action in ("STOP", "CONTINUE"):
                break

            # REFINE: add the new query to the pool and advance
            if refinement:
                query_pool.append(refinement)
            query_idx += 1

        return self._to_retrieved_papers(best_candidates), react_log

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _deduplicate(
        self, pairs: List[Tuple[PaperChunk, float]]
    ) -> List[Tuple[PaperChunk, float]]:
        """Keep the chunk with the lowest L2 distance per paper_id."""
        best: dict = {}
        for chunk, dist in pairs:
            if chunk.paper_id not in best or dist < best[chunk.paper_id][1]:
                best[chunk.paper_id] = (chunk, dist)
        return list(best.values())

    def _react_step(
        self,
        original_query: str,
        current_query: str,
        n_candidates: int,
        iteration: int,
    ) -> Tuple[str, str, str]:
        """Return (thought, action, refinement)."""

        # Fast path: enough papers — no LLM call needed
        if n_candidates >= MIN_PAPERS_REQUIRED:
            return (
                f"Retrieved {n_candidates} relevant papers ≥ minimum ({MIN_PAPERS_REQUIRED}). Sufficient.",
                "STOP",
                "",
            )

        # Last iteration: stop regardless
        if iteration >= MAX_REACT_ITERATIONS - 1:
            return (
                f"Max iterations reached. Proceeding with {n_candidates} papers.",
                "STOP",
                "",
            )

        # LLM reasoning about why retrieval was insufficient
        user_prompt = f"""Original query:  "{original_query}"
Current query:   "{current_query}"
Papers found:    {n_candidates}  (minimum required: {MIN_PAPERS_REQUIRED})
Iteration:       {iteration + 1} of {MAX_REACT_ITERATIONS}

The retrieval is insufficient. Reason about what went wrong and suggest a better query."""

        response = self.llm.complete(
            REACT_SYSTEM,
            user_prompt,
            component="react_reasoning",
        )
        return self._parse_react(response)

    @staticmethod
    def _parse_react(response: str) -> Tuple[str, str, str]:
        thought = refinement = ""
        action = "STOP"
        for line in response.splitlines():
            if line.startswith("Thought:"):
                thought    = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                raw_action = line[len("Action:"):].strip().upper()
                action     = raw_action if raw_action in ("REFINE", "STOP", "CONTINUE") else "STOP"
            elif line.startswith("Refinement:"):
                refinement = line[len("Refinement:"):].strip()
        return thought, action, refinement

    @staticmethod
    def _to_retrieved_papers(
        candidates: List[Tuple[PaperChunk, float]]
    ) -> List[RetrievedPaper]:
        papers = []
        for chunk, dist in candidates:
            papers.append(
                RetrievedPaper(
                    paper_id         = chunk.paper_id,
                    title            = chunk.title,
                    authors          = chunk.authors,
                    year             = chunk.year,
                    abstract         = chunk.chunk_text if chunk.chunk_type == "abstract" else "",
                    similarity_score = dist,
                )
            )
        return papers
