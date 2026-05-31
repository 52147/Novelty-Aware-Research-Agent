#!/usr/bin/env python3
"""
topup_to_100.py
===============
Your corpus is at 95 (5 of the earlier batch were already present). This adds 5
more REAL papers to reach exactly 100. Run AFTER expand_corpus_to_100.py.

Same caveat: papers are real; abstract text is condensed representative summary
text for the prototype corpus, not verbatim official abstracts.

    python topup_to_100.py
    # then rebuild the FAISS index (~300 vectors) and follow RERUN_CHECKLIST.md
"""

import json
import os

CORPUS_FILE = "sample_papers.json"


def P(pid, title, authors, year, abstract, intro, concl):
    return {"paper_id": pid, "title": title, "authors": authors, "year": year,
            "abstract": abstract, "introduction": intro, "conclusion": concl}


# 5 more real papers, chosen to avoid the IDs already in your corpus and to keep
# the retrieval/IR + agent balance.
TOPUP = [
    P("monot5_2020", "Document Ranking with a Pretrained Sequence-to-Sequence Model (monoT5)",
      ["Rodrigo Nogueira", "et al."], 2020,
      "monoT5 reranks passages by framing relevance as a sequence-to-sequence task, fine-tuning T5 to output a relevance token, a strong and simple neural reranker.",
      "We cast passage reranking as a text-to-text relevance prediction with a pretrained encoder-decoder.",
      "Sequence-to-sequence reranking is a simple, strong baseline for neural ranking."),
    P("rankgpt_2023", "Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents (RankGPT)",
      ["Weiwei Sun", "et al."], 2023,
      "RankGPT studies using LLMs as zero-shot listwise rerankers via permutation generation and distills their ability into smaller specialized rerankers.",
      "We investigate large language models as zero-shot listwise passage rerankers.",
      "LLMs are strong zero-shot rerankers and can be distilled into efficient models."),
    P("hotpotqa_2018", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering",
      ["Zhilin Yang", "et al."], 2018,
      "HotpotQA provides multi-hop questions requiring reasoning over multiple supporting documents, with sentence-level supporting facts for explainability, widely used to test multi-hop retrieval.",
      "We introduce a dataset requiring reasoning across multiple documents with supporting-fact supervision.",
      "Multi-hop QA with supporting facts drives research on retrieval-and-reasoning systems."),
    P("rarr_2023", "RARR: Researching and Revising What Language Models Say, Using Retrieval",
      ["Luyu Gao", "et al."], 2023,
      "RARR attributes and revises language-model outputs after generation by retrieving evidence and editing unsupported claims, improving attribution without retraining.",
      "We post-hoc attribute and revise model outputs using retrieved evidence.",
      "Retrieval-based post-editing improves factual attribution of generated text."),
    P("activerag_2023", "Active Retrieval Augmented Generation for Knowledge-Intensive Generation",
      ["Zhengbao Jiang", "et al."], 2023,
      "This work triggers retrieval actively during generation whenever the model is uncertain, anticipating future content to decide what to retrieve next for long-form generation.",
      "We retrieve actively and on-demand during generation rather than only once up front.",
      "Uncertainty-triggered active retrieval improves long-form knowledge-intensive generation."),
]


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"ERROR: {CORPUS_FILE} not found.")
        return
    with open(CORPUS_FILE) as f:
        corpus = json.load(f)
    print(f"Current corpus: {len(corpus)} papers")

    have = {p["paper_id"] for p in corpus}
    added, skipped = 0, []
    for p in TOPUP:
        if p["paper_id"] in have:
            skipped.append(p["paper_id"]); continue
        corpus.append(p); have.add(p["paper_id"]); added += 1

    with open(CORPUS_FILE, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Added {added}. Skipped {len(skipped)} duplicates: {skipped}")
    print(f"New corpus size: {len(corpus)} papers (~{len(corpus)*3} vectors after rebuild)")
    if len(corpus) != 100:
        n = 100 - len(corpus)
        print(f"\nStill {'short' if n>0 else 'over'} by {abs(n)}. "
              f"{'Tell me and I will add more.' if n>0 else 'Remove an entry.'}")
    else:
        print("\n✅ Exactly 100. Rebuild the FAISS index, then follow RERUN_CHECKLIST.md.")


if __name__ == "__main__":
    main()
