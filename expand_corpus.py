#!/usr/bin/env python3
"""
expand_corpus.py — adds 35 real papers to the existing 20-paper corpus,
bringing the total to 55. Weighted toward RAG / retrieval papers for IR relevance.

Usage:
    python expand_corpus.py
    python main.py --build-corpus sample_papers.json   # rebuild FAISS index
"""

import json
import os

# ── 35 additional real papers (no overlap with existing 20) ───────────────────
NEW_PAPERS = [
    # ── RAG / Retrieval (IR-focused — 12 papers) ──────────────────────────────
    {
        "paper_id": "selfrag_2023",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": ["Akari Asai", "Zeqiu Wu", "Yizhong Wang", "Avirup Sil", "Hannaneh Hajishirzi"],
        "year": 2023,
        "abstract": "Self-RAG is a framework that improves a language model's quality and factuality through retrieval and self-reflection. Rather than retrieving a fixed number of passages, the model learns to adaptively retrieve on demand and to critique both retrieved passages and its own generations using special reflection tokens.",
        "introduction": "Retrieval-augmented generation reduces hallucination but indiscriminate retrieval can introduce irrelevant passages and degrade output. We address whether and when to retrieve, and how to assess retrieved evidence. Self-RAG trains a single model to decide retrieval timing, evaluate passage relevance, and verify whether its output is supported, using reflection tokens generated inline.",
        "conclusion": "Self-RAG outperforms standard RAG and instruction-tuned baselines on open-domain QA, reasoning, and fact verification. Adaptive retrieval with self-critique improves both factuality and citation accuracy. Future work includes extending reflection to longer-form generation and multi-hop retrieval."
    },
    {
        "paper_id": "flare_2023",
        "title": "Active Retrieval Augmented Generation",
        "authors": ["Zhengbao Jiang", "Frank F. Xu", "Luyu Gao", "Zhiqing Sun", "Graham Neubig"],
        "year": 2023,
        "abstract": "FLARE is a generation method that actively decides when and what to retrieve during long-form generation. Instead of retrieving once at the start, it anticipates upcoming content by generating a temporary next sentence and retrieving relevant documents whenever the model expresses low confidence.",
        "introduction": "Most RAG systems retrieve once based on the initial query, which is insufficient for long-form generation where information needs shift across the output. We propose forward-looking active retrieval, where the system iteratively predicts the upcoming sentence, uses it as a query when confidence is low, and regenerates with retrieved context.",
        "conclusion": "FLARE achieves strong performance across long-form knowledge-intensive generation tasks. Actively triggering retrieval based on generation confidence outperforms single-shot and fixed-interval retrieval. The method generalizes across multiple datasets without task-specific training."
    },
    {
        "paper_id": "ragsurvey_2023",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": ["Yunfan Gao", "Yun Xiong", "Xinyu Gao", "Kangxiang Jia", "Haofen Wang"],
        "year": 2023,
        "abstract": "This survey reviews the development of retrieval-augmented generation for large language models, organizing the field into naive RAG, advanced RAG, and modular RAG paradigms. It examines the retrieval, generation, and augmentation components and discusses evaluation frameworks and open challenges.",
        "introduction": "Large language models exhibit hallucination and outdated knowledge, which retrieval augmentation mitigates by grounding generation in external sources. We survey the rapidly growing RAG literature, categorize architectures, and analyze how retrieval quality, document chunking, and fusion strategies affect downstream performance.",
        "conclusion": "RAG has evolved from simple retrieve-then-read pipelines to modular systems with iterative and adaptive retrieval. Key open problems include retrieval robustness, evaluation methodology, and integration with long-context models. The survey provides a roadmap for future RAG research."
    },
    {
        "paper_id": "replug_2023",
        "title": "REPLUG: Retrieval-Augmented Black-Box Language Models",
        "authors": ["Weijia Shi", "Sewon Min", "Michihiro Yasunaga", "Minjoon Seo", "Mike Lewis"],
        "year": 2023,
        "abstract": "REPLUG treats the language model as a frozen black box and prepends retrieved documents to the input. A tunable retriever is trained using the language model's own output probabilities as a supervision signal, requiring no access to model internals.",
        "introduction": "Many strong language models are available only through APIs, preventing fine-tuning for retrieval augmentation. We ask whether retrieval can improve such black-box models. REPLUG trains the retriever to find documents that minimize the language model's perplexity on the target, aligning retrieval with the frozen model.",
        "conclusion": "REPLUG improves black-box language model performance on language modeling and downstream tasks without modifying the model. Training the retriever against the model's output distribution is an effective supervision signal. The approach is compatible with arbitrary frozen models."
    },
    {
        "paper_id": "hyde_2022",
        "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
        "authors": ["Luyu Gao", "Xueguang Ma", "Jimmy Lin", "Jamie Callan"],
        "year": 2022,
        "abstract": "HyDE performs zero-shot dense retrieval by first generating a hypothetical document that answers the query using a language model, then encoding that hypothetical document to retrieve real documents. This sidesteps the need for relevance-labeled training data.",
        "introduction": "Dense retrieval typically requires relevance labels to train query and document encoders, which are unavailable in many settings. We propose generating a hypothetical answer document from the query and using its embedding for retrieval, leveraging the language model to bridge the query-document gap.",
        "conclusion": "HyDE matches or exceeds strong supervised dense retrievers across diverse tasks and languages without any relevance labels. Hypothetical document embeddings capture relevance patterns effectively. The method is a practical zero-shot retrieval approach."
    },
    {
        "paper_id": "crag_2024",
        "title": "Corrective Retrieval Augmented Generation",
        "authors": ["Shi-Qi Yan", "Jia-Chen Gu", "Yun Zhu", "Zhen-Hua Ling"],
        "year": 2024,
        "abstract": "Corrective RAG introduces a lightweight retrieval evaluator that assesses the quality of retrieved documents and triggers corrective actions. When retrieval is judged incorrect or ambiguous, the system falls back to web search and decomposes documents to filter irrelevant content.",
        "introduction": "RAG performance degrades sharply when retrieval returns irrelevant documents. We propose evaluating retrieval quality before generation and applying corrective strategies. A retrieval evaluator assigns confidence scores that route the system toward using, discarding, or supplementing retrieved knowledge.",
        "conclusion": "Corrective RAG improves robustness to retrieval errors across short- and long-form generation tasks. Decoupling retrieval evaluation from generation lets the system recover from poor retrieval. The approach is plug-and-play with existing RAG pipelines."
    },
    {
        "paper_id": "dpr_2020",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis", "Wen-tau Yih"],
        "year": 2020,
        "abstract": "Dense Passage Retrieval learns dense vector representations for questions and passages using a dual-encoder trained on question-passage pairs. It replaces sparse term-matching retrieval such as BM25 with learned dense embeddings for open-domain question answering.",
        "introduction": "Open-domain QA requires retrieving relevant passages from a large corpus. Traditional sparse methods rely on lexical overlap and miss semantic matches. We train a dual-encoder to embed questions and passages into a shared space where relevant pairs are close, enabling efficient maximum inner-product search.",
        "conclusion": "Dense Passage Retrieval substantially outperforms BM25 on open-domain QA benchmarks and improves end-to-end answer accuracy. Learned dense representations capture semantic relevance beyond lexical matching. DPR became a foundation for subsequent retrieval-augmented systems."
    },
    {
        "paper_id": "realm_2020",
        "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
        "authors": ["Kelvin Guu", "Kenton Lee", "Zora Tung", "Panupong Pasupat", "Ming-Wei Chang"],
        "year": 2020,
        "abstract": "REALM augments language model pre-training with a learned neural retriever over a textual knowledge corpus. The retriever is trained end-to-end with the language model using masked language modeling as the signal, allowing the model to attend to retrieved documents during prediction.",
        "introduction": "Language models store knowledge implicitly in parameters, making it hard to interpret or update. We propose learning a knowledge retriever jointly with the language model during pre-training, so the model explicitly retrieves and conditions on documents from a corpus when predicting masked tokens.",
        "conclusion": "REALM improves open-domain QA over models that store knowledge only in parameters. End-to-end training of retrieval with language modeling produces a retriever aligned with the model's needs. Explicit retrieval also improves interpretability and updatability."
    },
    {
        "paper_id": "atlas_2022",
        "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models",
        "authors": ["Gautier Izacard", "Patrick Lewis", "Maria Lomeli", "Lucas Hosseini", "Edouard Grave"],
        "year": 2022,
        "abstract": "Atlas is a retrieval-augmented language model designed for few-shot learning on knowledge-intensive tasks. It jointly trains a dense retriever and a sequence-to-sequence model, achieving strong performance with far fewer parameters than comparable closed-book models.",
        "introduction": "Large closed-book models memorize knowledge but require enormous scale. We investigate whether retrieval augmentation enables strong few-shot performance at smaller scale. Atlas couples a Contriever-style retriever with a Fusion-in-Decoder reader, training both on few examples.",
        "conclusion": "Atlas matches much larger closed-book models on knowledge-intensive tasks using orders of magnitude fewer parameters. Retrieval augmentation is especially effective in few-shot regimes. Decoupling knowledge storage from reasoning improves parameter efficiency."
    },
    {
        "paper_id": "incontextralm_2023",
        "title": "In-Context Retrieval-Augmented Language Models",
        "authors": ["Ori Ram", "Yoav Levine", "Itay Dalmedigos", "Dor Muhlgay", "Yoav Shoham"],
        "year": 2023,
        "abstract": "In-Context RALM prepends retrieved documents to the input of an off-the-shelf language model without any further training. The work shows that even simple BM25 retrieval combined with frozen models yields substantial language modeling improvements.",
        "introduction": "Retrieval augmentation often requires specialized training. We study how far one can go by simply prepending retrieved documents to a frozen language model's context. We analyze retrieval frequency, document ranking, and query construction to maximize gains without changing model weights.",
        "conclusion": "In-context retrieval augmentation improves frozen language models across model sizes with no training. Retrieval stride and reranking matter substantially. The simplicity makes the approach broadly applicable to existing models."
    },
    {
        "paper_id": "retro_2022",
        "title": "Improving Language Models by Retrieving from Trillions of Tokens",
        "authors": ["Sebastian Borgeaud", "Arthur Mensch", "Jordan Hoffmann", "Trevor Cai", "Laurent Sifre"],
        "year": 2022,
        "abstract": "RETRO enhances language models by retrieving from a database of trillions of tokens via chunked cross-attention. A relatively small model augmented with retrieval matches the performance of much larger models on language modeling.",
        "introduction": "Scaling parameters is an expensive route to better language modeling. We explore scaling retrieval instead. RETRO retrieves nearest-neighbor chunks from a massive text database and integrates them through a chunked cross-attention mechanism interleaved with standard layers.",
        "conclusion": "RETRO with retrieval from trillions of tokens matches models 25 times larger on language modeling. Retrieval is a competitive alternative to parameter scaling. The chunked cross-attention design integrates retrieved context efficiently."
    },
    {
        "paper_id": "fid_2021",
        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        "authors": ["Gautier Izacard", "Edouard Grave"],
        "year": 2021,
        "abstract": "Fusion-in-Decoder retrieves multiple passages and processes them independently in the encoder, then fuses their representations in the decoder to generate answers. This allows the model to aggregate evidence across many retrieved passages efficiently.",
        "introduction": "Generative open-domain QA must combine evidence from many retrieved passages. Concatenating passages is limited by sequence length. We encode each passage separately and let the decoder attend jointly across all encoded passages, scaling to large numbers of retrieved documents.",
        "conclusion": "Fusion-in-Decoder achieves strong open-domain QA performance and scales with the number of retrieved passages. Independent encoding with joint decoding is an efficient evidence aggregation strategy. The architecture became widely adopted in retrieval-augmented QA."
    },

    # ── Reasoning techniques (8 papers) ──────────────────────────────────────
    {
        "paper_id": "selfconsistency_2022",
        "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "authors": ["Xuezhi Wang", "Jason Wei", "Dale Schuurmans", "Quoc Le", "Ed Chi"],
        "year": 2022,
        "abstract": "Self-consistency replaces greedy decoding in chain-of-thought prompting with sampling multiple diverse reasoning paths and selecting the most consistent final answer by majority vote. This simple decoding strategy substantially improves reasoning accuracy.",
        "introduction": "Chain-of-thought prompting elicits step-by-step reasoning but greedy decoding commits to a single path that may contain errors. We hypothesize that complex problems admit multiple valid reasoning paths converging on the same answer. Sampling diverse paths and marginalizing over them should improve robustness.",
        "conclusion": "Self-consistency yields large gains across arithmetic and commonsense reasoning benchmarks. Aggregating multiple sampled reasoning paths is more reliable than any single path. The method requires no additional training or supervision."
    },
    {
        "paper_id": "leasttomost_2022",
        "title": "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
        "authors": ["Denny Zhou", "Nathanael Scharli", "Le Hou", "Jason Wei", "Ed Chi"],
        "year": 2022,
        "abstract": "Least-to-most prompting decomposes a complex problem into a sequence of simpler subproblems and solves them in order, using the answers to earlier subproblems as context for later ones. This enables generalization to harder problems than seen in the prompt.",
        "introduction": "Chain-of-thought struggles to generalize from easy demonstrations to harder problems. We propose explicitly decomposing problems into ordered subproblems. The model first reduces a problem to subproblems, then sequentially solves each, building toward the final answer.",
        "conclusion": "Least-to-most prompting solves problems substantially harder than the prompt examples, improving compositional generalization. Explicit decomposition outperforms standard chain-of-thought on symbolic and mathematical reasoning. The two-stage approach is broadly applicable."
    },
    {
        "paper_id": "got_2023",
        "title": "Graph of Thoughts: Solving Elaborate Problems with Large Language Models",
        "authors": ["Maciej Besta", "Nils Blach", "Ales Kubicek", "Robert Gerstenberger", "Torsten Hoefler"],
        "year": 2023,
        "abstract": "Graph of Thoughts models reasoning as a graph where vertices are intermediate thoughts and edges are dependencies, enabling aggregation, refinement, and feedback loops over thoughts. This generalizes chain and tree structures to arbitrary graphs.",
        "introduction": "Tree-of-thoughts allows branching reasoning but cannot merge or refine thoughts arbitrarily. We represent reasoning as a graph, where thoughts can be combined, looped, and distilled. This richer structure supports operations like aggregating partial solutions and iteratively improving them.",
        "conclusion": "Graph of Thoughts improves solution quality and reduces cost on tasks like sorting and set operations compared to tree-of-thoughts. Graph-structured reasoning enables aggregation and refinement beyond tree branching. The framework is extensible to new thought transformations."
    },
    {
        "paper_id": "pal_2022",
        "title": "PAL: Program-aided Language Models",
        "authors": ["Luyu Gao", "Aman Madaan", "Shuyan Zhou", "Uri Alon", "Graham Neubig"],
        "year": 2022,
        "abstract": "Program-aided language models generate reasoning steps as executable program code and offload the actual computation to a Python interpreter. This separates reasoning from calculation, eliminating arithmetic errors in the model's output.",
        "introduction": "Language models reason in natural language but make arithmetic and logical execution errors. We propose having the model write programs that express the reasoning steps, then execute them with an interpreter. The model focuses on decomposition while the interpreter handles exact computation.",
        "conclusion": "PAL outperforms chain-of-thought on mathematical and symbolic reasoning by delegating computation to an interpreter. Separating reasoning from execution removes a major error source. The approach generalizes across reasoning tasks requiring precise computation."
    },
    {
        "paper_id": "rewoo_2023",
        "title": "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models",
        "authors": ["Binfeng Xu", "Zhiyuan Peng", "Bowen Lei", "Subhabrata Mukherjee", "Dongkuan Xu"],
        "year": 2023,
        "abstract": "ReWOO decouples the reasoning process from external tool observations by generating a complete plan of tool calls upfront, then executing them, then combining results. This reduces redundant LLM calls compared to interleaved reasoning-acting approaches.",
        "introduction": "Interleaved reasoning-acting agents like ReAct repeatedly feed observations back into the model, incurring high token cost. We propose separating planning from execution: the model produces a full reasoning plan with placeholder evidence, tools fill the placeholders, and a solver combines them.",
        "conclusion": "ReWOO achieves comparable or better accuracy than ReAct while using substantially fewer tokens. Decoupling reasoning from observation reduces redundant context. The modular design also improves robustness to tool failures."
    },
    {
        "paper_id": "autocot_2022",
        "title": "Automatic Chain of Thought Prompting in Large Language Models",
        "authors": ["Zhuosheng Zhang", "Aston Zhang", "Mu Li", "Alex Smola"],
        "year": 2022,
        "abstract": "Auto-CoT automatically constructs chain-of-thought demonstrations by clustering questions and generating reasoning chains for representative examples, removing the need for manually written demonstrations.",
        "introduction": "Chain-of-thought prompting requires hand-crafted reasoning demonstrations, which is labor-intensive. We automate demonstration construction by clustering questions for diversity and using zero-shot prompting to generate reasoning chains for cluster representatives, then using these as demonstrations.",
        "conclusion": "Auto-CoT matches manually designed chain-of-thought prompts across reasoning benchmarks. Diversity-based sampling of demonstrations is key to performance. The method removes manual prompt engineering for chain-of-thought."
    },
    {
        "paper_id": "selfask_2022",
        "title": "Measuring and Narrowing the Compositionality Gap in Language Models",
        "authors": ["Ofir Press", "Muru Zhang", "Sewon Min", "Ludwig Schmidt", "Mike Lewis"],
        "year": 2022,
        "abstract": "This work introduces the self-ask prompting method, where the model explicitly asks and answers follow-up subquestions before producing a final answer. Self-ask narrows the compositionality gap and integrates naturally with a search engine.",
        "introduction": "Language models often know the facts needed for a multi-hop question yet fail to compose them. We measure this compositionality gap and propose self-ask, where the model decomposes a question into explicit follow-up questions, optionally answered by a search engine, before composing the final answer.",
        "conclusion": "Self-ask improves multi-hop question answering and narrows the compositionality gap. Explicit subquestion decomposition makes reasoning steps inspectable and supports tool integration. The gap persists even as model scale increases."
    },
    {
        "paper_id": "decomposed_2022",
        "title": "Decomposed Prompting: A Modular Approach for Solving Complex Tasks",
        "authors": ["Tushar Khot", "Harsh Trivedi", "Matthew Finlayson", "Yao Fu", "Ashish Sabharwal"],
        "year": 2022,
        "abstract": "Decomposed prompting solves complex tasks by breaking them into subtasks, each handled by a dedicated prompt that can be further decomposed recursively or delegated to specialized handlers including retrieval and symbolic operations.",
        "introduction": "Single monolithic prompts struggle with complex compositional tasks. We propose a modular library of decomposition prompts, where a controller routes subtasks to specialized handlers. Handlers can recurse, call other handlers, or invoke external operations, enabling flexible task solving.",
        "conclusion": "Decomposed prompting outperforms chain-of-thought on tasks requiring compositional and recursive structure. Modular subtask handlers improve reuse and debugging. The approach supports integrating symbolic and retrieval operations into prompting."
    },

    # ── Tool use & deployment (7 papers) ─────────────────────────────────────
    {
        "paper_id": "gorilla_2023",
        "title": "Gorilla: Large Language Model Connected with Massive APIs",
        "authors": ["Shishir G. Patil", "Tianjun Zhang", "Xin Wang", "Joseph E. Gonzalez"],
        "year": 2023,
        "abstract": "Gorilla is a finetuned language model that generates accurate API calls across large, changing API collections. It uses retrieval-aware training so the model can adapt to documentation changes at inference time and reduce hallucinated API usage.",
        "introduction": "Language models struggle to invoke the correct API among thousands of options and hallucinate arguments. We finetune a model with a document retriever in the loop, teaching it to consult retrieved API documentation when generating calls. This handles overlapping and evolving APIs.",
        "conclusion": "Gorilla outperforms general models including GPT-4 at writing correct API calls and adapts to documentation changes via retrieval. Retrieval-aware training reduces hallucination. The approach scales to large, dynamic tool collections."
    },
    {
        "paper_id": "critic_2023",
        "title": "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing",
        "authors": ["Zhibin Gou", "Zhihong Shao", "Yeyun Gong", "Yelong Shen", "Weizhu Chen"],
        "year": 2023,
        "abstract": "CRITIC enables language models to verify and correct their own outputs by interacting with external tools such as search engines and code interpreters. The model critiques its output using tool feedback, then revises accordingly.",
        "introduction": "Language models cannot reliably self-correct using only their internal knowledge. We propose grounding self-correction in external tools: the model generates an output, queries tools to validate it, and uses the returned evidence to critique and revise, mirroring how humans verify claims.",
        "conclusion": "CRITIC improves accuracy across question answering, mathematical reasoning, and toxicity reduction by grounding self-correction in tools. Tool feedback is more reliable than purely internal critique. External verification is essential for trustworthy self-correction."
    },
    {
        "paper_id": "chameleon_2023",
        "title": "Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models",
        "authors": ["Pan Lu", "Baolin Peng", "Hao Cheng", "Michel Galley", "Jianfeng Gao"],
        "year": 2023,
        "abstract": "Chameleon augments language models with a set of plug-and-play modules including vision models, search engines, and Python functions, composing them into programs to solve multimodal and knowledge-intensive reasoning tasks.",
        "introduction": "Language models lack access to up-to-date information, precise computation, and visual perception. We build a compositional reasoning system where the LLM acts as a planner that assembles sequences of modules from a toolbox to address each query's specific needs.",
        "conclusion": "Chameleon achieves strong results on science question answering and tabular reasoning by composing heterogeneous tools. LLM-driven program synthesis over a module library is flexible and interpretable. Plug-and-play design eases extension with new tools."
    },
    {
        "paper_id": "restgpt_2023",
        "title": "RestGPT: Connecting Large Language Models with Real-World RESTful APIs",
        "authors": ["Yifan Song", "Weimin Xiong", "Dawei Zhu", "Wenhao Wu", "Sujian Li"],
        "year": 2023,
        "abstract": "RestGPT connects language models to real-world RESTful APIs through a coarse-to-fine planning module and an API executor that handles authentication, parameter binding, and response parsing for complex multi-step tasks.",
        "introduction": "Real-world APIs are numerous, complex, and return large structured responses that challenge language model agents. We propose a hierarchical planner that decomposes user instructions into API calls and an executor that manages the practical details of invoking RESTful services.",
        "conclusion": "RestGPT completes complex tasks over real APIs such as movie databases and music services. Coarse-to-fine planning handles API complexity better than flat planning. Robust response parsing is critical for real-world tool use."
    },
    {
        "paper_id": "memgpt_2023",
        "title": "MemGPT: Towards LLMs as Operating Systems",
        "authors": ["Charles Packer", "Sarah Wooders", "Kevin Lin", "Vivian Fang", "Joseph E. Gonzalez"],
        "year": 2023,
        "abstract": "MemGPT manages a tiered memory hierarchy analogous to an operating system, allowing a language model to page information between a limited context window and external storage. This enables unbounded effective context for long conversations and document analysis.",
        "introduction": "Fixed context windows limit language models on long conversations and large documents. Inspired by operating system memory management, we give the model functions to move data between its context and external storage, letting it decide what to keep in working memory and what to retrieve.",
        "conclusion": "MemGPT handles long-term conversation and large-document analysis beyond the native context window. OS-inspired memory paging lets the model manage its own context. Self-directed memory management is a promising direction for long-horizon agents."
    },
    {
        "paper_id": "expel_2023",
        "title": "ExpeL: LLM Agents Are Experiential Learners",
        "authors": ["Andrew Zhao", "Daniel Huang", "Quentin Xu", "Matthieu Lin", "Gao Huang"],
        "year": 2023,
        "abstract": "ExpeL enables language model agents to learn from experience without parameter updates by collecting trajectories across training tasks, extracting natural-language insights, and retrieving relevant experiences at test time.",
        "introduction": "Fine-tuning agents is costly and proprietary models cannot be updated. We ask whether agents can improve from experience using only their context. ExpeL gathers successful and failed trajectories, distills cross-task insights, and recalls similar past experiences when facing new tasks.",
        "conclusion": "ExpeL improves agent performance across tasks without any gradient updates. Extracting and retrieving natural-language experience is an effective learning mechanism for frozen models. Experiential learning complements in-context and tool-use approaches."
    },
    {
        "paper_id": "fireact_2023",
        "title": "FireAct: Toward Language Agent Fine-tuning",
        "authors": ["Baian Chen", "Chang Shu", "Ehsan Shareghi", "Nigel Collier", "Yu Su"],
        "year": 2023,
        "abstract": "FireAct studies fine-tuning language models for agentic tasks using trajectories generated by stronger models across multiple prompting methods. It shows that fine-tuning on diverse agent trajectories improves smaller models substantially.",
        "introduction": "Most language agents rely on prompting frozen models, leaving fine-tuning underexplored. We investigate fine-tuning agents on trajectories collected from multiple tasks and prompting methods including ReAct and chain-of-thought. We analyze how trajectory diversity affects agent robustness.",
        "conclusion": "Fine-tuning on diverse agent trajectories improves smaller models' performance and robustness over prompting alone. Mixing prompting methods during fine-tuning helps generalization. Agent fine-tuning is a promising complement to prompting-based agents."
    },

    # ── Multi-agent (4 papers) ────────────────────────────────────────────────
    {
        "paper_id": "agentverse_2023",
        "title": "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors",
        "authors": ["Weize Chen", "Yusheng Su", "Jingwei Zuo", "Cheng Yang", "Zhiyuan Liu"],
        "year": 2023,
        "abstract": "AgentVerse is a framework for orchestrating multiple language model agents through expert recruitment, collaborative decision-making, action execution, and evaluation phases. It studies emergent social behaviors among collaborating agents.",
        "introduction": "Coordinating multiple agents toward a shared goal requires structured collaboration. We propose a framework that dynamically recruits expert agents, conducts collaborative decision-making, executes actions, and evaluates outcomes in iterative rounds, while observing emergent cooperative and competitive behaviors.",
        "conclusion": "AgentVerse enables multi-agent groups to outperform individual agents on reasoning and coding tasks. Structured collaboration phases improve coordination. The framework surfaces emergent behaviors relevant to multi-agent system design."
    },
    {
        "paper_id": "multidebate_2023",
        "title": "Improving Factuality and Reasoning in Language Models through Multiagent Debate",
        "authors": ["Yilun Du", "Shuang Li", "Antonio Torralba", "Joshua B. Tenenbaum", "Igor Mordatch"],
        "year": 2023,
        "abstract": "This work has multiple language model instances propose and debate their individual responses over several rounds, converging on a shared answer. Multiagent debate improves factual accuracy and reasoning over single-model baselines.",
        "introduction": "A single language model may commit to flawed reasoning. We propose having multiple model instances independently answer, then critique and revise their answers in light of others' responses across debate rounds. This mirrors deliberative human reasoning and surfaces errors.",
        "conclusion": "Multiagent debate improves factuality and reasoning across diverse tasks. Iterative cross-examination among model instances reduces errors and hallucination. Debate is a simple, training-free way to improve output quality."
    },
    {
        "paper_id": "metaagents_2023",
        "title": "MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination",
        "authors": ["Yuan Li", "Yixuan Zhang", "Lichao Sun"],
        "year": 2023,
        "abstract": "MetaAgents studies coordination among language model agents in simulated task-oriented social environments, examining how agents with different roles negotiate, plan, and assign tasks to achieve collective goals.",
        "introduction": "Coordinating language model agents in realistic social settings remains challenging. We construct a simulated environment where agents with distinct roles must communicate and divide labor. We analyze how reasoning and coordination capabilities affect collective task success.",
        "conclusion": "MetaAgents reveals both capabilities and limitations of language model agents in coordinated task-oriented settings. Role specialization and communication structure strongly affect outcomes. Coordination remains a bottleneck for complex multi-agent tasks."
    },
    {
        "paper_id": "roco_2023",
        "title": "RoCo: Dialectic Multi-Robot Collaboration with Large Language Models",
        "authors": ["Zhao Mandi", "Shreeya Jain", "Shuran Song"],
        "year": 2023,
        "abstract": "RoCo uses language models for multi-robot collaboration, where each robot agent reasons and dialogues to jointly plan, then validates the plan against motion constraints. It combines high-level language reasoning with low-level motion planning.",
        "introduction": "Coordinating multiple robots requires both high-level task allocation and low-level feasibility. We assign each robot a language model agent that proposes and discusses subtask plans through dialogue, then validate the agreed plan with a motion planner, iterating when validation fails.",
        "conclusion": "RoCo achieves effective multi-robot coordination across collaborative manipulation tasks. Combining dialectic language reasoning with motion validation bridges high- and low-level planning. Language-based negotiation is a viable coordination mechanism for embodied multi-agent systems."
    },

    # ── Evaluation & benchmarks (4 papers) ───────────────────────────────────
    {
        "paper_id": "gaia_2023",
        "title": "GAIA: A Benchmark for General AI Assistants",
        "authors": ["Gregoire Mialon", "Clementine Fourrier", "Craig Swift", "Thomas Wolf", "Yann LeCun"],
        "year": 2023,
        "abstract": "GAIA is a benchmark of real-world questions requiring reasoning, multimodality, web browsing, and tool use. The questions are conceptually simple for humans yet challenging for AI assistants, testing fundamental assistant capabilities.",
        "introduction": "Existing benchmarks saturate or test narrow skills. We propose questions that humans answer easily but that require AI assistants to combine reasoning, tool use, and multimodal understanding. The benchmark resists memorization and measures general assistant competence.",
        "conclusion": "GAIA exposes a large gap between human and AI-assistant performance despite question simplicity for humans. Tool use and multi-step reasoning remain bottlenecks. The benchmark provides a target for general-purpose assistant development."
    },
    {
        "paper_id": "webarena_2023",
        "title": "WebArena: A Realistic Web Environment for Building Autonomous Agents",
        "authors": ["Shuyan Zhou", "Frank F. Xu", "Hao Zhu", "Xuhui Zhou", "Graham Neubig"],
        "year": 2023,
        "abstract": "WebArena is a realistic, reproducible web environment with functional websites for e-commerce, forums, and collaborative development, paired with tasks that require agents to perform multi-step actions to achieve goals.",
        "introduction": "Evaluating web agents requires realistic environments rather than simplified simulations. We build self-hosted functional websites and define long-horizon tasks with execution-based evaluation. This enables reproducible measurement of agent capabilities in web navigation and interaction.",
        "conclusion": "WebArena shows that current language model agents complete only a small fraction of realistic web tasks. Long-horizon planning and robust action execution are key gaps. The environment supports reproducible web-agent research."
    },
    {
        "paper_id": "swebench_2023",
        "title": "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        "authors": ["Carlos E. Jimenez", "John Yang", "Alexander Wettig", "Shunyu Yao", "Karthik Narasimhan"],
        "year": 2023,
        "abstract": "SWE-bench evaluates language models on resolving real GitHub issues by generating code patches that must pass the repository's test suite. The tasks require understanding large codebases and coordinating changes across files.",
        "introduction": "Code generation benchmarks use isolated functions, not realistic software engineering. We collect real issue-and-pull-request pairs from popular repositories, requiring models to produce patches validated by the project's own tests. This measures practical software engineering ability.",
        "conclusion": "SWE-bench reveals that even strong models resolve only a small percentage of real issues. Repository-scale context and cross-file coordination are major challenges. The benchmark drives research on practical coding agents."
    },
    {
        "paper_id": "mint_2023",
        "title": "MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback",
        "authors": ["Xingyao Wang", "Zihan Wang", "Jiateng Liu", "Yangyi Chen", "Heng Ji"],
        "year": 2023,
        "abstract": "MINT evaluates language models on multi-turn tasks where they can use tools and receive natural-language feedback. It measures how well models leverage tool calls and incorporate user feedback across interaction turns.",
        "introduction": "Real assistant use involves multiple turns, tool use, and iterative feedback, but most benchmarks are single-turn. We construct tasks that allow tool interaction and simulated user feedback, measuring how performance changes as models use more turns and feedback.",
        "conclusion": "MINT shows that models vary widely in their ability to use tools and feedback across turns, and that some improvements from feedback do not generalize. Multi-turn tool interaction is an important and under-measured capability. The benchmark guides development of interactive agents."
    },
]


def main():
    base = "sample_papers.json"
    if not os.path.exists(base):
        print(f"ERROR: {base} not found. Run this from the novelty_agent/ directory.")
        return

    with open(base) as f:
        existing = json.load(f)

    existing_ids = {p["paper_id"] for p in existing}
    added = 0
    for p in NEW_PAPERS:
        if p["paper_id"] not in existing_ids:
            existing.append(p)
            added += 1

    # Backup original
    with open("sample_papers_20.json.bak", "w") as f:
        json.dump(existing[:20], f, indent=2)

    with open(base, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Added {added} papers.")
    print(f"Corpus now has {len(existing)} papers.")
    print("Original 20 backed up to sample_papers_20.json.bak")
    print()
    print("Next step — rebuild the FAISS index:")
    print("    python main.py --build-corpus sample_papers.json")


if __name__ == "__main__":
    main()
