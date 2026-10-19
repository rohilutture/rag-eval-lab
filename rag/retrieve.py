"""Retrievers: BM25 (lexical), dense (embeddings), hybrid (reciprocal rank fusion),
and an optional cross-encoder reranker.

Why hybrid: filings are full of exact tokens (tickers, "Item 7A", "$1,234 million",
product names) that BM25 nails and embeddings blur, while paraphrased questions
("how much did they spend on R&D") need semantic match. RRF combines ranked
lists without having to calibrate BM25 scores against cosine similarities:
score(d) = sum over lists of 1 / (k + rank_d).
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk
from .llm import Embedder

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        s = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.chunks[i], float(s[i]), "bm25") for i in top]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None):
        self.chunks, self.embedder = chunks, embedder
        self.vecs = vectors if vectors is not None else embedder(
            [c.title + "\n" + c.text for c in chunks], kind="doc")

    def search(self, query: str, k: int) -> list[Hit]:
        q = self.embedder([query], kind="query")[0]
        s = self.vecs @ q
        top = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.chunks[i], float(s[i]), "dense") for i in top]

    def save(self, path: Path):
        np.save(path, self.vecs)


def rrf(lists: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    scores, keep = {}, {}
    for lst in lists:
        for rank, h in enumerate(lst):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            keep.setdefault(cid, h.chunk)
    best = sorted(scores, key=lambda c: -scores[c])[:k]
    return [Hit(keep[c], scores[c], "hybrid") for c in best]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, fanout: int = 50):
        self.bm25, self.dense, self.fanout = bm25, dense, fanout

    def search(self, query: str, k: int) -> list[Hit]:
        return rrf([self.bm25.search(query, self.fanout), self.dense.search(query, self.fanout)], k)


class CrossEncoderReranker:
    """Scores (query, passage) jointly with full attention across both — far more
    accurate than a dot product of separate embeddings, but too slow to run on the
    whole corpus. Hence: retrieve ~30-50 cheaply, rerank, keep the top few."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return hits
        s = self.ce.predict([(query, h.chunk.text) for h in hits])
        order = np.argsort(-np.asarray(s), kind="stable")[:k]
        return [Hit(hits[i].chunk, float(s[i]), hits[i].source + "+rerank") for i in order]


class OverlapReranker:
    """Dependency-free stand-in for tests: query-term coverage of the passage."""

    def rerank(self, query, hits, k):
        q = set(tokenize(query))
        s = [len(q & set(tokenize(h.chunk.text))) / (len(q) or 1) for h in hits]
        order = np.argsort(-np.asarray(s), kind="stable")[:k]
        return [Hit(hits[i].chunk, float(s[i]), hits[i].source + "+rerank") for i in order]


def save_index(path: Path, chunks, dense: DenseRetriever | None):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    if dense is not None:
        dense.save(path / "vectors.npy")
        (path / "embedder.txt").write_text(dense.embedder.spec)
