#!/usr/bin/env python3
"""
RAG retrieval server — FastAPI wrapper around the FAISS indexes.

Loads all three collections on startup and serves search requests.
Run this on an SGE GPU/CPU node with access to data/embeds/.

Usage:
  python rag/server.py                          # defaults
  python rag/server.py --embeds-version v1      # explicit version
  python rag/server.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import os
import re
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retrieve import Retriever, rerank as _rerank
from rag.auth import load_api_key

# ---------------------------------------------------------------------------
# Fuzzy matching (rapidfuzz preferred, difflib fallback)
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import process as _rf_process, fuzz as _fuzz

    def _fuzzy_find(query: str, choices: list[str], score_cutoff: int, limit: int = 5):
        return _rf_process.extract(
            query, choices,
            scorer=_fuzz.WRatio,
            score_cutoff=score_cutoff,
            limit=limit,
        )
except ImportError:
    from difflib import SequenceMatcher, get_close_matches as _gcm

    def _fuzzy_find(query: str, choices: list[str], score_cutoff: int, limit: int = 5):
        cutoff = score_cutoff / 100.0
        matches = _gcm(query, choices, n=limit, cutoff=cutoff)
        return [(m, SequenceMatcher(None, query, m).ratio() * 100, 0) for m in matches]


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower()).strip()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    freshness: float = 0.5
    paper_impact: float = 0.5

class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict

class SearchResponse(BaseModel):
    collection: str
    query: str
    results: list[SearchResult]

class AuthorSearchRequest(BaseModel):
    name: str
    top_k: int = 20
    score_cutoff: int = 80
    freshness: float = 0.5
    paper_impact: float = 0.5

class AuthorMatch(BaseModel):
    name: str
    paper_count: int
    score: float

class AuthorSearchResponse(BaseModel):
    matched_author: Optional[str]
    match_score: Optional[float]
    paper_count: int
    query_name: str
    alternatives: list[AuthorMatch]
    results: list[dict]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

retriever:    Optional[Retriever] = None
author_index: dict = {}
author_names: list[str] = []

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

def make_app(embeds_version: str, api_key: str) -> FastAPI:
    rerank_alpha = float(os.environ.get("RERANK_ALPHA", "0.3"))
    rerank_beta  = float(os.environ.get("RERANK_BETA",  "0.1"))

    def verify_key(key: str = Security(API_KEY_HEADER)):
        if not secrets.compare_digest(key, api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global retriever, author_index, author_names
        print(f"Loading retriever (embeds/{embeds_version}, alpha={rerank_alpha}, beta={rerank_beta})...")
        retriever = Retriever(embeds_version=embeds_version,
                              rerank_alpha=rerank_alpha, rerank_beta=rerank_beta)
        for coll in ["metadata", "abstract", "session"]:
            retriever._load_collection(coll)
        # author_topics is optional — only load if the index exists
        _at_path = Path(__file__).parent.parent / "data" / "embeds" / embeds_version / "author_topics.faiss"
        if _at_path.exists():
            retriever._load_collection("author_topics")
            print("author_topics collection loaded.")
        else:
            print(f"[INFO] No author_topics index at {_at_path} — skipping. Run build_author_topic_index.sh to build it.")
        print("Retriever ready.")

        author_index_path = Path(__file__).parent.parent / "data" / "author_index.json"
        if author_index_path.exists():
            print(f"Loading author index...")
            data = json.loads(author_index_path.read_text(encoding="ascii"))
            author_index = data["authors"]
            author_names = list(author_index.keys())
            print(f"Author index ready: {len(author_names):,} authors")
        else:
            print(f"[WARN] No author index found — run: python scrape_papers/build_author_index.py")

        yield
        print("Shutting down.")

    app = FastAPI(title="Speech RAG Server", lifespan=lifespan)

    @app.get("/health")
    def health():
        # health check is unauthenticated — just confirms server is up
        return {"status": "ok", "embeds_version": embeds_version}

    @app.post("/search/author", response_model=AuthorSearchResponse,
              dependencies=[Depends(verify_key)])
    def search_author(req: AuthorSearchRequest):
        if not author_names:
            raise HTTPException(status_code=503,
                                detail="Author index not loaded. Run build_author_index.py first.")
        query_norm = _normalize_name(req.name)
        if not query_norm:
            raise HTTPException(status_code=400, detail="name must not be empty")

        matches = _fuzzy_find(query_norm, author_names, req.score_cutoff, limit=5)
        if not matches:
            return AuthorSearchResponse(
                matched_author=None, match_score=None, paper_count=0,
                query_name=req.name, alternatives=[], results=[],
            )

        top_key, top_score, _ = matches[0]
        entry = author_index[top_key]

        alternatives = [
            AuthorMatch(
                name=author_index[k]["canonical"],
                paper_count=author_index[k]["paper_count"],
                score=s,
            )
            for k, s, _ in matches[1:]
            if top_score - s <= 5.0
        ]

        # Sort author papers by the dominant dimension.
        # Continuous blending doesn't work here because citation counts span
        # orders of magnitude — a paper with 500 cit always beats a fresh paper
        # with 5 cit in any blend. Instead: whichever score is higher controls
        # the primary sort key; the other score breaks ties.
        f, p = req.freshness, req.paper_impact
        if f >= p:
            # recency-primary: most recent first, citations break ties
            papers = sorted(
                entry["papers"],
                key=lambda x: (-(x.get("year") or 0), -(x.get("citation_count") or 0)),
            )[: req.top_k]
        else:
            # citation-primary: most cited first, year breaks ties
            papers = sorted(
                entry["papers"],
                key=lambda x: (-(x.get("citation_count") or 0), -(x.get("year") or 0)),
            )[: req.top_k]
        return AuthorSearchResponse(
            matched_author=entry["canonical"],
            match_score=top_score,
            paper_count=entry["paper_count"],
            query_name=req.name,
            alternatives=alternatives,
            results=papers,
        )

    # registered after /search/author so the wildcard doesn't swallow it
    @app.post("/search/{collection}", response_model=SearchResponse,
              dependencies=[Depends(verify_key)])
    def search(collection: str, req: SearchRequest):
        if collection not in ("metadata", "abstract", "session", "author_topics"):
            raise HTTPException(status_code=400, detail=f"Unknown collection: {collection}")
        if collection == "author_topics" and collection not in retriever._indexes:
            raise HTTPException(status_code=503, detail="author_topics index not loaded. Run build_author_topic_index.sh first.")
        results = retriever.search(
            req.query, collection,
            top_k=req.top_k,
            freshness=req.freshness,
            paper_impact=req.paper_impact,
        )
        return SearchResponse(
            collection=collection,
            query=req.query,
            results=[SearchResult(**r) for r in results],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="RAG retrieval server.")
    parser.add_argument("--embeds-version", default="v1")
    parser.add_argument("--host",           default="127.0.0.1")
    parser.add_argument("--port",           type=int, default=8000)
    parser.add_argument("--workers",        type=int, default=1)
    args = parser.parse_args()

    api_key = load_api_key()
    app = make_app(args.embeds_version, api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
