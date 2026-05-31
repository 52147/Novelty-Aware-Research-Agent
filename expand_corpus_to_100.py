#!/usr/bin/env python3
"""
expand_corpus_to_100.py
=======================
Extends the existing 55-paper corpus to 100 by appending 45 additional REAL
papers in the agentic-AI / reasoning / retrieval / IR space (weighted toward
retrieval and IR, which suits the venue).

IMPORTANT
---------
- The 45 papers are real (real authors, venues, years). The abstract/intro/
  conclusion text here is CONDENSED, REPRESENTATIVE summary text written to
  populate a prototype retrieval corpus. It is not the verbatim official
  abstract. If you need precision, replace the text with the official abstracts.
- Adding papers changes EVERY downstream number. After running this you must
  re-run all experiments and re-thread the numbers (see RERUN_CHECKLIST.md).
  The paper may not claim "100 papers" until those experiments actually run.

Usage
-----
    python expand_corpus_to_100.py
    # backs up sample_papers.json -> sample_papers_55.json.bak
    # writes the 100-paper sample_papers.json
    # then REBUILD the index (your corpus_builder build step) so FAISS has ~300 vectors
"""

import json
import os
import shutil

CORPUS_FILE = "sample_papers.json"
BACKUP_FILE = "sample_papers_55.json.bak"


def P(pid, title, authors, year, abstract, intro, concl):
    return {
        "paper_id": pid, "title": title, "authors": authors, "year": year,
        "abstract": abstract, "introduction": intro, "conclusion": concl,
    }


NEW_PAPERS = [
    # ── Tool use / agents ────────────────────────────────────────────────────
    P("toolformer_2023", "Toolformer: Language Models Can Teach Themselves to Use Tools",
      ["Timo Schick", "et al."], 2023,
      "Toolformer is a language model trained to decide which APIs to call, when, and how to incorporate the results, learning tool use in a self-supervised way from a handful of demonstrations.",
      "We study how language models can teach themselves to use external tools such as search engines, calculators, and translation systems via simple API calls.",
      "Self-supervised tool learning improves zero-shot performance across tasks while preserving core language modeling ability."),
    P("hugginggpt_2023", "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
      ["Yongliang Shen", "et al."], 2023,
      "HuggingGPT uses an LLM as a controller to plan tasks, select expert models from a model hub, execute subtasks, and aggregate responses, enabling multimodal task solving.",
      "We connect a language-model controller to a community of specialist models to solve complex AI tasks spanning language, vision, and speech.",
      "LLM-as-controller orchestration enables compositional multimodal problem solving across many expert models."),
    P("tot_2023", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
      ["Shunyu Yao", "et al."], 2023,
      "Tree of Thoughts generalizes chain-of-thought by exploring a tree of intermediate reasoning states with lookahead and backtracking, improving performance on tasks requiring search and planning.",
      "We frame reasoning as search over a tree of partial solutions, allowing deliberate exploration, evaluation, and backtracking.",
      "Deliberate tree search over thoughts substantially improves performance on planning and search problems."),
    P("plansolve_2023", "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning",
      ["Lei Wang", "et al."], 2023,
      "Plan-and-Solve prompting first devises a plan to divide a task into subtasks, then carries out the subtasks, reducing calculation and missing-step errors in zero-shot reasoning.",
      "We address common error types in zero-shot chain-of-thought by explicitly prompting the model to plan before solving.",
      "Planning before solving reduces reasoning errors without exemplars."),
    P("selfrefine_2023", "Self-Refine: Iterative Refinement with Self-Feedback",
      ["Aman Madaan", "et al."], 2023,
      "Self-Refine improves LLM outputs by having the same model generate feedback on its own output and revise iteratively, without additional training.",
      "We propose iterative self-feedback and revision as a test-time mechanism to improve generation quality.",
      "Self-generated feedback and refinement improve outputs across diverse tasks without extra supervision."),
    P("art_2023", "ART: Automatic Multi-Step Reasoning and Tool-Use for Large Language Models",
      ["Bhargavi Paranjape", "et al."], 2023,
      "ART automatically selects reasoning demonstrations and interleaves tool calls during multi-step reasoning, allowing frozen LLMs to use external tools without hand-crafted prompts.",
      "We automate program-style multi-step reasoning with tool use for frozen language models.",
      "Automatic reasoning programs with tool use generalize across tasks and tools."),
    P("cove_2023", "Chain-of-Verification Reduces Hallucination in Large Language Models",
      ["Shehzaad Dhuliawala", "et al."], 2023,
      "Chain-of-Verification drafts an initial response, plans verification questions, answers them independently, and produces a verified final response, reducing factual hallucination.",
      "We reduce hallucination by having the model verify its own claims through independently answered checking questions.",
      "Self-verification through planned questions reduces hallucinations in generated text."),
    P("rest_2023", "Reinforced Self-Training (ReST) for Language Modeling",
      ["Caglar Gulcehre", "et al."], 2023,
      "ReST alternates generating a dataset from the policy (Grow) and fine-tuning on filtered high-reward samples (Improve), an offline RL approach for aligning language models.",
      "We align language models with reward signals through an offline grow-and-improve self-training loop.",
      "Reinforced self-training improves quality with reduced compute relative to online RL."),

    # ── Embodied / GUI / web agents ──────────────────────────────────────────
    P("webgpt_2021", "WebGPT: Browser-Assisted Question-Answering with Human Feedback",
      ["Reiichiro Nakano", "et al."], 2021,
      "WebGPT fine-tunes a model to answer long-form questions using a text-based web browser, navigating and citing sources, trained with human feedback and rejection sampling.",
      "We teach a model to search and browse the web to answer open-ended questions with citations.",
      "Browser-assisted QA with human feedback yields more factual, well-cited answers."),
    P("webshop_2022", "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents",
      ["Shunyu Yao", "et al."], 2022,
      "WebShop is a simulated e-commerce environment with millions of products where agents must navigate and purchase items from natural-language instructions, testing grounded web interaction.",
      "We build a large-scale web environment to train and evaluate language agents on realistic shopping tasks.",
      "Grounded web agents show progress but remain far below human performance on realistic tasks."),
    P("mind2web_2023", "Mind2Web: Towards a Generalist Agent for the Web",
      ["Xiang Deng", "et al."], 2023,
      "Mind2Web is a dataset for developing generalist web agents that follow language instructions to complete tasks across many real websites and domains.",
      "We collect diverse real-website tasks to train and evaluate generalist web-browsing agents.",
      "Generalist web agents require broad transfer across sites, domains, and interaction patterns."),
    P("cogagent_2023", "CogAgent: A Visual Language Model for GUI Agents",
      ["Wenyi Hong", "et al."], 2023,
      "CogAgent is a vision-language model specialized for GUI understanding and navigation, recognizing screen elements and predicting actions from screenshots.",
      "We build a visual language model that operates graphical user interfaces directly from pixels.",
      "Screen-grounded visual agents can navigate GUIs without relying on accessibility trees."),
    P("appagent_2023", "AppAgent: Multimodal Agents as Smartphone Users",
      ["Chi Zhang", "et al."], 2023,
      "AppAgent is a multimodal agent that learns to operate smartphone applications by exploration and by observing demonstrations, acting through taps and swipes.",
      "We develop a multimodal agent that uses mobile apps the way a human user would.",
      "Exploration-based learning enables agents to operate unfamiliar smartphone applications."),
    P("gitm_2023", "Ghost in the Minecraft: Generally Capable Agents for Open-World Environments",
      ["Xizhou Zhu", "et al."], 2023,
      "Ghost in the Minecraft combines LLM-based planning with structured action interfaces to achieve long-horizon goals in the open-world game Minecraft.",
      "We pair language-model planning with a structured action layer for open-world embodied tasks.",
      "Structured LLM planning unlocks long-horizon achievement in open-world environments."),
    P("deps_2023", "Describe, Explain, Plan and Select: Interactive Planning with LLMs for Open-World Agents",
      ["Zihao Wang", "et al."], 2023,
      "DEPS is an interactive planning approach where an LLM describes, explains failures, plans, and selects goals, improving long-horizon task completion for open-world agents.",
      "We propose an interactive describe-explain-plan-select loop for robust open-world planning.",
      "Failure-aware interactive planning improves long-horizon success rates."),
    P("saycan_2022", "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)",
      ["Michael Ahn", "et al."], 2022,
      "SayCan grounds language-model instructions in robotic affordances by combining LLM task likelihoods with learned value functions, so a robot performs feasible useful actions.",
      "We ground language-model plans in what a robot can actually do via affordance value functions.",
      "Combining language priors with affordances yields feasible real-robot behavior."),
    P("innermono_2022", "Inner Monologue: Embodied Reasoning through Planning with Language Models",
      ["Wenlong Huang", "et al."], 2022,
      "Inner Monologue incorporates environment feedback into an LLM planning loop, letting embodied agents replan from success detection, scene description, and human feedback.",
      "We close the loop between language-model planning and embodied feedback.",
      "Feedback-grounded inner monologue improves embodied task completion."),
    P("codeaspolicies_2022", "Code as Policies: Language Model Programs for Embodied Control",
      ["Jacky Liang", "et al."], 2022,
      "Code as Policies uses LLMs to write executable robot policy code from language commands, composing perception and control APIs into runnable programs.",
      "We treat robot control as program synthesis driven by language models.",
      "Language-model-generated policy code generalizes to new instructions and objects."),
    P("lats_2023", "Language Agent Tree Search Unifies Reasoning, Acting, and Planning",
      ["Andy Zhou", "et al."], 2023,
      "LATS uses Monte Carlo tree search over language-agent actions with environment feedback and self-reflection, unifying reasoning, acting, and planning.",
      "We bring principled search to language agents via Monte Carlo tree search with reflection.",
      "Search over agent actions improves decision-making on reasoning and tool-use tasks."),
    P("rap_2023", "Reasoning with Language Model is Planning with World Model (RAP)",
      ["Shibo Hao", "et al."], 2023,
      "RAP repurposes an LLM as both a reasoning agent and a world model, using Monte Carlo tree search to plan over predicted states for multi-step reasoning.",
      "We cast reasoning as planning by treating the language model as a world model.",
      "Planning with an LLM world model improves multi-step reasoning."),
    P("adapt_2023", "ADaPT: As-Needed Decomposition and Planning with Language Models",
      ["Archiki Prasad", "et al."], 2023,
      "ADaPT recursively decomposes tasks only when the executor fails, adapting plan granularity to task and model capability for interactive decision-making.",
      "We decompose tasks as needed rather than all at once, adapting to failures.",
      "As-needed recursive decomposition improves success on complex interactive tasks."),
    P("agenttuning_2023", "AgentTuning: Enabling Generalized Agent Abilities for LLMs",
      ["Aohan Zeng", "et al."], 2023,
      "AgentTuning instruction-tunes open LLMs on a mixture of agent trajectories and general instructions, improving agent abilities without sacrificing general capability.",
      "We fine-tune open models on agent interaction data to build generalist agents.",
      "Mixed agent and general tuning yields broad agent skills while preserving general ability."),
    P("lemur_2023", "Lemur: Harmonizing Natural Language and Code for Language Agents",
      ["Yiheng Xu", "et al."], 2023,
      "Lemur is an open model pretrained and instruction-tuned to balance language and coding ability, targeting the needs of language agents that act in environments.",
      "We build open models that harmonize natural language and code for agentic use.",
      "Balanced language-code ability improves agent performance across environments."),

    # ── Retrieval-augmented reasoning ────────────────────────────────────────
    P("ircot_2023", "Interleaving Retrieval with Chain-of-Thought Reasoning for Multi-Step Questions (IRCoT)",
      ["Harsh Trivedi", "et al."], 2023,
      "IRCoT interleaves retrieval with chain-of-thought, using each reasoning step to guide what to retrieve next, improving multi-hop question answering.",
      "We couple retrieval and reasoning so each guides the other for multi-hop QA.",
      "Interleaving retrieval with reasoning outperforms one-shot retrieval on multi-hop questions."),
    P("dsp_2022", "Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP",
      ["Omar Khattab", "et al."], 2022,
      "DSP is a framework that composes retrieval models and language models through programs that demonstrate, search, and predict, for knowledge-intensive tasks.",
      "We compose retrievers and language models with a programming abstraction for knowledge tasks.",
      "Program-level composition of retrieval and generation improves knowledge-intensive QA."),
    P("dspy_2023", "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines",
      ["Omar Khattab", "et al."], 2023,
      "DSPy expresses LLM pipelines as declarative modules and compiles them by optimizing prompts and demonstrations, replacing brittle hand-crafted prompting.",
      "We treat LLM pipelines as programs that can be optimized and compiled.",
      "Declarative, compilable pipelines outperform hand-tuned prompting."),
    P("chainofnote_2023", "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models",
      ["Wenhao Yu", "et al."], 2023,
      "Chain-of-Note generates sequential reading notes for retrieved documents to assess relevance before answering, improving robustness to noisy or irrelevant retrieval.",
      "We add a note-taking step over retrieved documents to improve RAG robustness.",
      "Reading notes improve robustness to irrelevant or noisy retrieved passages."),
    P("recomp_2023", "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation",
      ["Fangyuan Xu", "et al."], 2023,
      "RECOMP compresses retrieved documents into concise summaries before in-context augmentation, reducing cost and filtering irrelevant content.",
      "We compress retrieved evidence into summaries to make RAG cheaper and more focused.",
      "Compressing retrieved context improves efficiency without harming task performance."),
    P("query2doc_2023", "Query2doc: Query Expansion with Large Language Models",
      ["Liang Wang", "et al."], 2023,
      "Query2doc expands a query by prompting an LLM to generate a pseudo-document, which improves both sparse and dense retrieval.",
      "We expand queries with LLM-generated pseudo-documents to improve retrieval.",
      "LLM-based query expansion consistently improves retrieval effectiveness."),
    P("genread_2022", "Generate rather than Retrieve: Large Language Models are Strong Context Generators",
      ["Wenhao Yu", "et al."], 2022,
      "GenRead prompts an LLM to generate contextual documents and reads them to answer, sometimes matching or surpassing retrieve-then-read pipelines.",
      "We replace retrieval with model-generated context for knowledge-intensive QA.",
      "Generated context can rival retrieved context for some knowledge tasks."),
    P("raptor_2024", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
      ["Parth Sarthi", "et al."], 2024,
      "RAPTOR builds a hierarchical tree of recursively summarized text chunks, enabling retrieval at multiple levels of abstraction for long-document QA.",
      "We organize a corpus as a recursive summary tree for multi-scale retrieval.",
      "Tree-organized abstractive retrieval improves long-context question answering."),
    P("lostmiddle_2023", "Lost in the Middle: How Language Models Use Long Contexts",
      ["Nelson F. Liu", "et al."], 2023,
      "This study shows language models use information best at the beginning and end of long contexts and degrade when relevant content is in the middle, with implications for retrieval order.",
      "We analyze how position within long contexts affects language-model performance.",
      "Position strongly affects long-context use, motivating careful ordering of retrieved evidence."),
    P("knnlm_2020", "Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)",
      ["Urvashi Khandelwal", "et al."], 2020,
      "kNN-LM interpolates a language model with a nearest-neighbor retrieval over a datastore of cached representations, improving perplexity without further training.",
      "We augment language modeling with explicit nearest-neighbor retrieval over stored examples.",
      "Retrieval over a datastore improves language modeling, especially for rare patterns."),
    P("promptagator_2022", "Promptagator: Few-Shot Dense Retrieval from 8 Examples",
      ["Zhuyun Dai", "et al."], 2022,
      "Promptagator generates synthetic queries from a few examples using a large LM to train task-specific dense retrievers with minimal supervision.",
      "We bootstrap task-specific dense retrievers from only a handful of examples.",
      "Few-shot synthetic query generation yields strong task-specific retrievers."),

    # ── Dense / sparse retrieval & IR benchmarks ─────────────────────────────
    P("colbert_2020", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
      ["Omar Khattab", "Matei Zaharia"], 2020,
      "ColBERT introduces late interaction, encoding queries and documents into token-level embeddings and scoring via efficient maximum-similarity, balancing effectiveness and efficiency.",
      "We propose late interaction to retain fine-grained matching while remaining scalable.",
      "Late interaction delivers strong retrieval quality at practical efficiency."),
    P("colbertv2_2022", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
      ["Keshav Santhanam", "et al."], 2022,
      "ColBERTv2 improves late-interaction retrieval with residual compression and denoised supervision, reducing storage while improving quality.",
      "We make late-interaction retrieval more accurate and far more storage-efficient.",
      "Compression and better supervision improve both quality and footprint."),
    P("splade_2021", "SPLADE: Sparse Lexical and Expansion Model for First-Stage Ranking",
      ["Thibault Formal", "et al."], 2021,
      "SPLADE learns sparse lexical representations with term expansion via masked language modeling, combining inverted-index efficiency with learned semantics.",
      "We learn sparse, expansion-aware representations for efficient first-stage retrieval.",
      "Learned sparse retrieval is competitive with dense methods while staying index-friendly."),
    P("contriever_2021", "Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever)",
      ["Gautier Izacard", "et al."], 2021,
      "Contriever trains dense retrievers with contrastive learning on unlabeled text, achieving strong zero-shot and transfer retrieval without supervised query-document pairs.",
      "We learn dense retrievers without labeled data using contrastive pretraining.",
      "Unsupervised contrastive training yields strong transferable dense retrievers."),
    P("gtr_2021", "Large Dual Encoders Are Generalizable Retrievers (GTR)",
      ["Jianmo Ni", "et al."], 2021,
      "GTR shows that scaling dual-encoder retrievers improves out-of-domain generalization, with a single model transferring across diverse retrieval tasks.",
      "We study how scaling dual encoders affects retrieval generalization.",
      "Scaling dual-encoder retrievers improves out-of-domain transfer."),
    P("e5_2022", "Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)",
      ["Liang Wang", "et al."], 2022,
      "E5 trains general-purpose text embeddings via weakly-supervised contrastive pretraining on curated text pairs, performing well across retrieval and embedding benchmarks.",
      "We pretrain general text embeddings from large weakly-supervised pair collections.",
      "Weakly-supervised contrastive embeddings generalize across many tasks."),
    P("bge_2023", "C-Pack: Packed Resources for General Chinese and Multilingual Text Embeddings (BGE)",
      ["Shitao Xiao", "et al."], 2023,
      "BGE / C-Pack provides training data, models, and recipes for general text embeddings, achieving strong results on embedding and retrieval benchmarks.",
      "We release packed resources and recipes for training strong general text embeddings.",
      "Open embedding resources enable competitive general-purpose retrieval models."),
    P("sbert_2019", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
      ["Nils Reimers", "Iryna Gurevych"], 2019,
      "Sentence-BERT fine-tunes BERT in a siamese structure to produce semantically meaningful sentence embeddings comparable with cosine similarity, enabling efficient semantic search.",
      "We produce fixed-size sentence embeddings suitable for fast semantic similarity.",
      "Siamese fine-tuning yields embeddings well-suited to retrieval and clustering."),
    P("mteb_2022", "MTEB: Massive Text Embedding Benchmark",
      ["Niklas Muennighoff", "et al."], 2022,
      "MTEB is a benchmark spanning many embedding tasks and datasets, providing a standardized comparison of text embedding models across retrieval, clustering, and classification.",
      "We standardize evaluation of text embeddings across a broad set of tasks.",
      "A unified embedding benchmark reveals that no single model dominates all tasks."),
    P("beir_2021", "BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models",
      ["Nandan Thakur", "et al."], 2021,
      "BEIR is a heterogeneous zero-shot retrieval benchmark across diverse domains and tasks, used to evaluate the generalization of retrieval models.",
      "We assemble a diverse zero-shot benchmark to test retrieval generalization.",
      "Zero-shot evaluation across domains exposes large gaps between retrieval methods."),
    P("msmarco_2016", "MS MARCO: A Human Generated Machine Reading Comprehension Dataset",
      ["Tri Nguyen", "et al."], 2016,
      "MS MARCO is a large-scale dataset of real user queries with human-generated answers and passage relevance, widely used to train and evaluate retrieval and ranking models.",
      "We release a large dataset of real queries and passages for reading and ranking.",
      "Large-scale real-query data enables training of modern neural retrievers and rankers."),
]


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"ERROR: {CORPUS_FILE} not found in this directory.")
        return

    with open(CORPUS_FILE) as f:
        existing = json.load(f)
    print(f"Existing corpus: {len(existing)} papers")

    if not os.path.exists(BACKUP_FILE):
        shutil.copy(CORPUS_FILE, BACKUP_FILE)
        print(f"Backed up -> {BACKUP_FILE}")
    else:
        print(f"Backup {BACKUP_FILE} already exists; not overwriting.")

    have = {p["paper_id"] for p in existing}
    added, skipped = 0, []
    for p in NEW_PAPERS:
        if p["paper_id"] in have:
            skipped.append(p["paper_id"])
            continue
        existing.append(p)
        have.add(p["paper_id"])
        added += 1

    with open(CORPUS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Added {added} new papers. Skipped {len(skipped)} duplicates: {skipped}")
    print(f"New corpus size: {len(existing)} papers")
    print(f"Expected FAISS vectors after rebuild: ~{len(existing)*3} "
          f"(3 chunks/paper)")
    print("\nNEXT: rebuild the FAISS index, then follow RERUN_CHECKLIST.md.")
    if len(existing) != 100:
        print(f"\nNOTE: corpus is {len(existing)}, not exactly 100. "
              f"Adjust NEW_PAPERS if you need precisely 100.")


if __name__ == "__main__":
    main()
