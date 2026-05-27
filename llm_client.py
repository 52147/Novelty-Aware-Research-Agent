"""
llm_client.py — Mock LLM client for testing without an API key.

Returns realistic hardcoded responses for each pipeline component.
Set USE_MOCK=True (default) to run without OpenAI.
Set USE_MOCK=False and provide OPENAI_API_KEY to use the real API.
"""

import json
import os
from typing import Optional, Dict, Any

USE_MOCK = False   # flip to False when you have a real API key

COMPONENT_TEMPERATURES: Dict[str, float] = {
    "query_analyzer":  0.3,
    "react_reasoning": 0.3,
    "ranker":          0.1,
    "extractor":       0.1,
    "comparison":      0.2,
    "generator":       0.7,
}

# ── Mock responses per component ──────────────────────────────────────────────

MOCK_RESPONSES: Dict[str, Any] = {

    "query_analyzer": {
        "original_query": "Compare multi-agent LLM frameworks for collaborative reasoning",
        "intents": [
            {
                "intent": "Compare architectures of multi-agent LLM systems",
                "domain": "multi-agent LLMs",
                "comparison_axis": "architecture and coordination mechanism"
            },
            {
                "intent": "Evaluate collaborative reasoning methods across frameworks",
                "domain": "LLM reasoning",
                "comparison_axis": "reasoning strategy and task performance"
            }
        ],
        "reformulated_queries": [
            "multi-agent LLM collaboration framework architecture",
            "LLM agent coordination reasoning benchmark",
            "role-playing conversational agents task solving"
        ]
    },

    "ranker": {
        "rankings": [
            {"paper_id": "autogen_2023",    "relevance_score": 9.5, "reason": "Directly proposes a multi-agent conversation framework."},
            {"paper_id": "metagpt_2023",    "relevance_score": 9.2, "reason": "Role-based multi-agent coordination for software tasks."},
            {"paper_id": "camel_2023",      "relevance_score": 8.8, "reason": "Role-playing dialogue between two collaborative agents."},
            {"paper_id": "reflexion_2023",  "relevance_score": 7.4, "reason": "Single-agent verbal RL, adjacent to multi-agent collaboration."},
            {"paper_id": "agentbench_2023", "relevance_score": 7.0, "reason": "Benchmark evaluating agents including multi-agent setups."},
        ]
    },

    "extractor_autogen_2023": {
        "paper_id": "autogen_2023",
        "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "problem_statement": "Single-agent LLM systems are insufficient for complex, multi-step tasks that require diverse capabilities and iterative refinement.",
        "proposed_method": "A multi-agent conversation framework where customizable agents communicate via structured message passing to solve tasks collaboratively.",
        "key_contribution": "AutoGen introduces a flexible conversation pattern abstraction supporting hierarchical, nested, and group chat agent topologies without custom orchestration code.",
        "claimed_novelty": "Unlike prior work, AutoGen unifies diverse agent interaction patterns under a single programmable interface, enabling both automated and human-in-the-loop workflows."
    },

    "extractor_metagpt_2023": {
        "paper_id": "metagpt_2023",
        "title": "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework",
        "problem_statement": "Multi-agent systems lack structured coordination protocols, leading to inconsistent outputs and role confusion on complex engineering tasks.",
        "proposed_method": "Role-specialized GPT agents (product manager, architect, engineer, QA) coordinate through standardized deliverables and structured communication channels inspired by software engineering workflows.",
        "key_contribution": "MetaGPT encodes human organizational conventions into agent interactions, reducing hallucination and improving task completion on software development benchmarks.",
        "claimed_novelty": "First framework to embed software engineering SOPs into multi-agent LLM coordination, resulting in coherent end-to-end software generation."
    },

    "extractor_camel_2023": {
        "paper_id": "camel_2023",
        "title": "CAMEL: Communicative Agents for Mind Exploration of Large Language Model Society",
        "problem_statement": "Multi-agent LLM collaboration requires human intervention to maintain role coherence and prevent conversation degradation.",
        "proposed_method": "Inception prompting assigns complementary roles to two agents, embedding task context into system prompts to guide autonomous dialogue without human oversight.",
        "key_contribution": "CAMEL demonstrates that role-playing via inception prompting enables autonomous, coherent multi-agent task completion across diverse domains.",
        "claimed_novelty": "Introduces inception prompting as a mechanism for maintaining agent role alignment throughout extended conversations without human intervention."
    },

    "extractor_reflexion_2023": {
        "paper_id": "reflexion_2023",
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "problem_statement": "Reinforcing language agents traditionally requires expensive gradient-based weight updates that are impractical for rapid iteration.",
        "proposed_method": "Verbal reinforcement: agents generate natural language reflections on failed attempts, storing them in episodic memory to improve future attempts without modifying model weights.",
        "key_contribution": "Achieves state-of-the-art on HumanEval (91%), WebArena, and AlfWorld using verbal self-reflection as a lightweight alternative to RL fine-tuning.",
        "claimed_novelty": "First to use natural language self-critique as a reinforcement signal stored in episodic memory, enabling interpretable learning without gradient updates."
    },

    "extractor_agentbench_2023": {
        "paper_id": "agentbench_2023",
        "title": "AgentBench: Evaluating LLMs as Agents",
        "problem_statement": "There is no standardized benchmark for evaluating LLMs as autonomous agents across diverse, realistic environments.",
        "proposed_method": "A benchmark spanning eight environments with standardized protocols measuring multi-step reasoning, tool use, and long-horizon task completion across 27 LLMs.",
        "key_contribution": "Reveals a significant performance gap between commercial and open-source LLMs on agent tasks, providing a reproducible evaluation framework.",
        "claimed_novelty": "First comprehensive agent benchmark covering eight distinct environment types with unified evaluation protocols."
    },

    "comparison_overlaps": {
        "overlaps": [
            {
                "paper_ids": ["autogen_2023", "metagpt_2023", "camel_2023"],
                "shared_element": "problem formulation",
                "description": "All three frame the core problem as enabling effective agent collaboration through structured communication — they differ in how structure is imposed."
            },
            {
                "paper_ids": ["autogen_2023", "metagpt_2023"],
                "shared_element": "method family",
                "description": "Both use role-specialization as the coordination primitive, though AutoGen is general-purpose while MetaGPT is domain-specific to software engineering."
            }
        ]
    },

    "comparison_differences": {
        "differences": [
            {
                "paper_id": "autogen_2023",
                "title": "AutoGen",
                "differentiating_aspect": "General-purpose conversation topology",
                "description": "AutoGen supports arbitrary agent graph topologies (hierarchical, nested, group), making it the most flexible framework but with less built-in task structure."
            },
            {
                "paper_id": "metagpt_2023",
                "title": "MetaGPT",
                "differentiating_aspect": "SOP-driven structured coordination",
                "description": "MetaGPT is the only framework that embeds domain-specific SOPs as first-class coordination primitives."
            },
            {
                "paper_id": "camel_2023",
                "title": "CAMEL",
                "differentiating_aspect": "Two-agent inception prompting",
                "description": "CAMEL is the only framework focusing on a strict two-agent dyad with role coherence maintained via inception prompting."
            },
            {
                "paper_id": "reflexion_2023",
                "title": "Reflexion",
                "differentiating_aspect": "Verbal reinforcement without weight updates",
                "description": "Reflexion is the only system treating self-critique as a learning signal, operating in a single-agent loop rather than multi-agent collaboration."
            },
            {
                "paper_id": "agentbench_2023",
                "title": "AgentBench",
                "differentiating_aspect": "Evaluation framework rather than system",
                "description": "AgentBench contributes a benchmark, not a new agent architecture."
            }
        ]
    },

    "comparison_gaps": {
        "gap_matrix": [
            {
                "problem_formulation": "Multi-agent collaboration for complex reasoning tasks",
                "missing_method": "Verbal reinforcement applied to multi-agent settings",
                "description": "No paper applies verbal reinforcement learning (Reflexion-style) to a multi-agent topology. Multi-agent verbal RL is absent from this corpus.",
                "supporting_evidence": "Reflexion operates single-agent only. AutoGen, MetaGPT, and CAMEL use no learning signal between attempts."
            },
            {
                "problem_formulation": "Standardized evaluation of collaborative reasoning",
                "missing_method": "Benchmark targeting inter-agent coordination quality",
                "description": "AgentBench evaluates individual LLM capabilities, not the coordination quality between agents in a multi-agent system.",
                "supporting_evidence": "AgentBench evaluates 27 individual LLMs; no paper benchmarks agent-to-agent coordination protocols."
            }
        ]
    },

    "generator": {
        "per_paper_summaries": [
            {"paper_id": "autogen_2023",    "title": "AutoGen", "year": 2023, "one_sentence_summary": "AutoGen provides a general-purpose framework for building multi-agent LLM applications through flexible, programmable conversation topologies."},
            {"paper_id": "metagpt_2023",    "title": "MetaGPT", "year": 2023, "one_sentence_summary": "MetaGPT encodes software engineering SOPs into role-specialized agents for structured multi-agent coordination."},
            {"paper_id": "camel_2023",      "title": "CAMEL",   "year": 2023, "one_sentence_summary": "CAMEL introduces inception prompting to enable autonomous two-agent role-playing for task completion without human intervention."},
            {"paper_id": "reflexion_2023",  "title": "Reflexion","year": 2023, "one_sentence_summary": "Reflexion enables lightweight agent improvement through natural language self-reflection stored in episodic memory."},
            {"paper_id": "agentbench_2023", "title": "AgentBench","year": 2023, "one_sentence_summary": "AgentBench provides the first standardized multi-environment benchmark for evaluating LLMs as autonomous agents."},
        ],
        "synthesis": (
            "The retrieved papers establish that multi-agent LLM collaboration is a productive paradigm "
            "for complex task completion, with three distinct coordination approaches: general topology "
            "(AutoGen), SOP-driven role specialization (MetaGPT), and dyadic role-playing (CAMEL). "
            "Reflexion contributes an orthogonal finding that verbal self-reflection is a viable "
            "lightweight learning signal, but remains confined to single-agent settings. "
            "A notable gap within this corpus is the absence of any system combining verbal reinforcement "
            "with multi-agent architectures, and the lack of a benchmark targeting inter-agent "
            "coordination quality rather than individual agent capability."
        ),
        "citations": [
            "[autogen_2023] Wu et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.",
            "[metagpt_2023] Hong et al. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework.",
            "[camel_2023] Li et al. (2023). CAMEL: Communicative Agents for Mind Exploration of Large Language Model Society.",
            "[reflexion_2023] Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.",
            "[agentbench_2023] Liu et al. (2023). AgentBench: Evaluating LLMs as Agents.",
        ]
    }
}


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.model     = model
        self._use_mock = USE_MOCK or not (api_key or os.getenv("OPENAI_API_KEY"))

        if self._use_mock:
            print("  [MockLLM] Running in mock mode — no API calls will be made.")
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self._extraction_order = [
            "autogen_2023", "metagpt_2023", "camel_2023",
            "reflexion_2023", "agentbench_2023"
        ]
        self._extract_idx = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        component: str,
        json_schema: Optional[Dict] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self._use_mock:
            return json.dumps(self._mock_response(component, user_prompt))

        temp = temperature if temperature is not None else COMPONENT_TEMPERATURES.get(component, 0.3)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temp,
        }
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": json_schema, "strict": True},
            }
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        component: str,
        schema: Dict,
    ) -> Dict:
        raw = self.complete(system_prompt, user_prompt, component, json_schema=schema)
        return json.loads(raw)

    # ── Mock routing ──────────────────────────────────────────────────────────

    def _mock_response(self, component: str, user_prompt: str) -> Dict:
        if component == "query_analyzer":
            return MOCK_RESPONSES["query_analyzer"]

        if component == "ranker":
            return MOCK_RESPONSES["ranker"]

        if component == "extractor":
            paper_id = self._detect_paper_id(user_prompt)
            key = f"extractor_{paper_id}"
            if key in MOCK_RESPONSES:
                return MOCK_RESPONSES[key]
            pid = self._extraction_order[self._extract_idx % len(self._extraction_order)]
            self._extract_idx += 1
            return MOCK_RESPONSES[f"extractor_{pid}"]

        if component == "comparison":
            prompt_lower = user_prompt.lower()
            if "overlap" in prompt_lower:
                return MOCK_RESPONSES["comparison_overlaps"]
            if "differenti" in prompt_lower:
                return MOCK_RESPONSES["comparison_differences"]
            if "gap" in prompt_lower:
                return MOCK_RESPONSES["comparison_gaps"]
            return MOCK_RESPONSES["comparison_overlaps"]

        if component == "generator":
            return MOCK_RESPONSES["generator"]

        if component == "react_reasoning":
            return {"thought": "Sufficient papers retrieved.", "action": "STOP", "refinement": ""}

        return {"result": f"[mock response for component={component}]"}

    @staticmethod
    def _detect_paper_id(prompt: str) -> str:
        for pid in ["autogen_2023", "metagpt_2023", "camel_2023", "reflexion_2023", "agentbench_2023"]:
            if pid in prompt:
                return pid
        return "autogen_2023"
