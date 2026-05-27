"""
corpus_builder.py — Builds and searches a FAISS vector index over paper chunks.

Course technique: RAG Indexing Pipeline (RAG slides, pp. 6-7, 14)
  - Chunk at abstract + introduction + conclusion
    (course rec: 200-500 tokens per chunk, 10-20% overlap handled by section boundaries)
  - Embed with Sentence-Transformers
  - Store in IndexFlatL2 (exact search, no compression — per course code example)
"""

import numpy as np
import faiss
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

from schemas import PaperChunk


EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dimension


class CorpusBuilder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 = None   # type: ignore
        self.chunks: List[PaperChunk] = []

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_paper(self, paper: Dict) -> List[PaperChunk]:
        """
        Split a paper into abstract / introduction / conclusion chunks.
        Using section boundaries keeps each chunk semantically coherent
        (200-500 token range, per course slide recommendation).
        """
        chunks = []
        for section in ("abstract", "introduction", "conclusion"):
            text = paper.get(section, "").strip()
            if text:
                chunks.append(
                    PaperChunk(
                        paper_id  = paper["paper_id"],
                        title     = paper["title"],
                        authors   = paper.get("authors", []),
                        year      = paper.get("year", 2024),
                        chunk_text= text,
                        chunk_type= section,
                    )
                )
        return chunks

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, papers: List[Dict], save_path: str = "corpus") -> None:
        """
        Embed all chunks and build a FAISS IndexFlatL2.

        IndexFlatL2 explanation (course slide p.14):
          "Flat" = no compression, exact search.
          "L2"   = Euclidean distance — good default for MiniLM embeddings.
        """
        Path(save_path).mkdir(exist_ok=True)

        for paper in papers:
            self.chunks.extend(self.chunk_paper(paper))

        texts = [c.chunk_text for c in self.chunks]
        print(f"Embedding {len(texts)} chunks from {len(papers)} papers…")
        embeddings = self.embedder.encode(texts, show_progress_bar=True).astype(np.float32)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embeddings)                       # course slide: index.add(...)

        # Persist
        faiss.write_index(self.index, f"{save_path}/faiss.index")
        with open(f"{save_path}/chunks.json", "w") as f:
            json.dump([c.model_dump() for c in self.chunks], f, indent=2)

        print(f"Index built: {self.index.ntotal} vectors stored.")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_index(self, save_path: str = "corpus") -> None:
        self.index = faiss.read_index(f"{save_path}/faiss.index")
        with open(f"{save_path}/chunks.json") as f:
            self.chunks = [PaperChunk(**c) for c in json.load(f)]
        print(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks.")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Tuple[PaperChunk, float]]:
        """
        Encode query and retrieve top-k chunks by L2 distance.

        Course RAG slide (p.7):
          D_k = TopK( similarity(e_q, e_d) )
        """
        q_emb = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results
