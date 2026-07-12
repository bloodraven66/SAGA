#!/usr/bin/env python3
"""
Retrieval layer — load FAISS indexes and query them.

Usage (CLI test mode):
  python rag/retrieve.py --query "self supervised learning for speech"
  python rag/retrieve.py --query "papers by Watanabe on end-to-end ASR" --collection metadata
  python rag/retrieve.py --query "what sessions were at Interspeech 2023" --collection session
  python rag/retrieve.py --query "noise robust ASR" --top-k 10 --embeds-version v1
"""

import argparse
import json
import math
import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_CURRENT_YEAR = datetime.date.today().year


def rerank(results: list[dict], alpha: float = 0.3, beta: float = 0.1) -> list[dict]:
    """Re-score results by blending FAISS score with citation count and recency."""
    for r in results:
        p    = r["payload"]
        cit  = p.get("citation_count")  # None = not yet fetched, skip citation boost
        year = p.get("year") or _CURRENT_YEAR
        recency = max(0.0, 1.0 - (_CURRENT_YEAR - int(year)) / 15.0)
        cit_factor = (1 + alpha * math.log(cit + 2)) if cit is not None else 1.0
        r["score"] = r["score"] * cit_factor * (1 + beta * recency)
    return sorted(results, key=lambda r: r["score"], reverse=True)

EMBEDS_DIR = Path("data/embeds")

# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, embeds_version: str = "v1",
                 rerank_alpha: float = 0.3, rerank_beta: float = 0.1):
        self.embeds_dir = EMBEDS_DIR / embeds_version
        config_path = self.embeds_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config at {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        print(f"Loading model: {self.config['model']}")
        self.model = SentenceTransformer(self.config["model"])
        self.query_instruction = self.config.get("query_instruction", "")
        self.rerank_alpha = rerank_alpha
        self.rerank_beta  = rerank_beta

        self._indexes   = {}   # collection → faiss.Index
        self._payloads  = {}   # collection → list[dict]
        self._ids       = {}   # collection → list[str]

    def _load_collection(self, collection: str):
        if collection in self._indexes:
            return
        base = self.embeds_dir / collection
        faiss_path   = base.with_suffix(".faiss")
        payload_path = self.embeds_dir / f"{collection}_payloads.json"
        ids_path     = self.embeds_dir / f"{collection}_ids.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"Index not found: {faiss_path}")

        self._indexes[collection]  = faiss.read_index(str(faiss_path))
        with open(payload_path, encoding="ascii") as f:
            self._payloads[collection] = json.load(f)
        with open(ids_path, encoding="ascii") as f:
            self._ids[collection] = json.load(f)

        n = self._indexes[collection].ntotal
        print(f"  [{collection}] loaded  {n} vectors")

    def _embed_query(self, query: str) -> np.ndarray:
        text = self.query_instruction + query if self.query_instruction else query
        vec  = self.model.encode(
            [text],
            normalize_embeddings=self.config.get("normalize", True),
            show_progress_bar=False,
        )
        return vec.astype("float32")

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        freshness: float = 0.5,
        paper_impact: float = 0.5,
    ) -> list[dict]:
        """
        Returns list of dicts:
          {"id": ..., "score": ..., "payload": {...}}
        sorted by descending score.

        freshness:    0.0 = prefer older papers, 1.0 = prefer recent papers (scales beta)
        paper_impact: 0.0 = ignore citations,    1.0 = strongly prefer high-citation papers (scales alpha)
        """
        self._load_collection(collection)
        vec = self._embed_query(query)
        scores, indices = self._indexes[collection].search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "id":      self._ids[collection][idx],
                "score":   float(score),
                "payload": self._payloads[collection][idx],
            })
        alpha = self.rerank_alpha * (paper_impact * 2)  # 0.5 → default, 1.0 → 2×, 0.0 → disabled
        beta  = self.rerank_beta  * (freshness    * 2)
        if alpha > 0 or beta > 0:
            results = rerank(results, alpha, beta)
        return results


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def print_results(results: list[dict], collection: str):
    print(f"\n{'─'*65}")
    print(f"  {collection.upper()}  —  {len(results)} results")
    print(f"{'─'*65}")
    for i, r in enumerate(results, 1):
        p = r["payload"]
        print(f"\n  [{i}]  score={r['score']:.4f}  id={r['id']}")
        if collection in ("metadata", "abstract"):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            print(f"       {p.get('title', '')}")
            print(f"       {p.get('venue', '')} {p.get('year', '')}  |  {p.get('session', '')}")
            print(f"       {authors}")
            if collection == "abstract" and "abstract" in p:
                snippet = p["abstract"][:200].replace("\n", " ")
                print(f"       \"{snippet}...\"")
        elif collection == "session":
            print(f"       {p.get('venue', '')} {p.get('year', '')}  |  {p.get('session_name', '')}")
            print(f"       {p.get('paper_count', 0)} papers")
            for t in p.get("paper_titles", [])[:5]:
                print(f"         - {t}")
            if p.get("paper_count", 0) > 5:
                print(f"         ... ({p['paper_count'] - 5} more)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query RAG indexes.")
    parser.add_argument("--query",          required=True, help="Query string")
    parser.add_argument("--collection",     default=None,
                        choices=["metadata", "abstract", "session"],
                        help="Collection to search (default: all three)")
    parser.add_argument("--top-k",          type=int, default=5,
                        help="Number of results per collection (default: 5)")
    parser.add_argument("--embeds-version", default="v1",
                        help="Embeds version to load (default: v1)")
    args = parser.parse_args()

    retriever   = Retriever(embeds_version=args.embeds_version)
    collections = [args.collection] if args.collection else ["metadata", "abstract", "session"]

    print(f"\nQuery: \"{args.query}\"")
    print(f"Instruction: \"{retriever.query_instruction}\"")

    for coll in collections:
        results = retriever.search(args.query, coll, top_k=args.top_k)
        print_results(results, coll)


if __name__ == "__main__":
    main()
