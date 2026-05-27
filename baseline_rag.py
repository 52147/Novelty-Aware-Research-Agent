"""
baseline_rag.py — Runs a simple baseline RAG system for comparison.

Baseline: retrieve top-5 papers → ask GPT-4o to summarize/compare directly.
No structured extraction, no Comparison Agent, no schema constraints.

Run:
    python baseline_rag.py --query "Compare multi-agent LLM frameworks for collaborative reasoning" --output results/baseline_run_01.json
    python baseline_rag.py --query "What evaluation methods exist for LLM reasoning agents?" --output results/baseline_run_02.json
    python baseline_rag.py --query "Compare verbal reinforcement and role-playing approaches in LLM agents" --output results/baseline_run_03.json
"""

import argparse
import json
import os
import time
from openai import OpenAI
from corpus_builder import CorpusBuilder

def run_baseline(query: str, corpus_path: str = "corpus", top_k: int = 5) -> dict:
    embedder = CorpusBuilder()
    embedder.load_index(corpus_path)

    # Retrieve top-k chunks
    results = embedder.search(query, top_k=top_k * 3)

    # Deduplicate by paper_id
    seen = {}
    for chunk, dist in results:
        if chunk.paper_id not in seen:
            seen[chunk.paper_id] = (chunk, dist)

    top_papers = list(seen.values())[:top_k]

    # Build context from chunks
    context = "\n\n".join([
        f"Title: {chunk.title}\nAbstract: {chunk.chunk_text}"
        for chunk, _ in top_papers
    ])

    paper_ids = [chunk.paper_id for chunk, _ in top_papers]

    # Single GPT-4o call — no structured extraction, no comparison agent
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Summarize and compare the following papers based on the user's query. Provide a cohesive summary paragraph, note similarities and differences where relevant, and mention any apparent gaps."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nPapers:\n{context}\n\nPlease summarize and compare these papers."
            }
        ],
        temperature=0.7
    )
    elapsed = round(time.time() - t0, 2)

    output_text = response.choices[0].message.content

    return {
        "query": query,
        "papers_retrieved": paper_ids,
        "n_papers": len(paper_ids),
        "output": output_text,
        "elapsed_seconds": elapsed,
        "metadata": {
            "model": "gpt-4o",
            "temperature": 0.7,
            "method": "direct_summarization",
            "structured_extraction": False,
            "comparison_agent": False,
            "schema_constraints": False
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Baseline RAG comparison system")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, default="corpus")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print(f"\nBaseline RAG")
    print(f"Query: {args.query}")
    print("Retrieving and summarizing...")

    result = run_baseline(args.query, args.corpus_path)

    print(f"\n{'='*60}")
    print("BASELINE OUTPUT")
    print(f"{'='*60}")
    print(f"Papers: {', '.join(result['papers_retrieved'])}")
    print(f"\n{result['output']}")
    print(f"\nElapsed: {result['elapsed_seconds']}s")
    print(f"{'='*60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
