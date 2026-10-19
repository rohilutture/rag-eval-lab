"""The RAG pipeline: retrieve -> (rerank) -> generate with citations, fully traced.

Generation contract (enforced by the prompt, checked by evals):
  * answer only from the numbered context passages
  * cite passages inline as [1], [2]
  * if the context doesn't contain the answer, say exactly "I don't know"
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .llm import ChatModel, Embedder
from .retrieve import (BM25Retriever, CrossEncoderReranker, DenseRetriever, Hit,
                       HybridRetriever, OverlapReranker)

SYSTEM = (
    "You answer questions about company filings using ONLY the numbered context passages. "
    "Cite every factual sentence with the passage number in brackets, like [2]. "
    "Copy numbers exactly as written, with units. "
    "If the passages do not contain the answer, reply exactly: I don't know"
)
ABSTAIN = "i don't know"


@dataclass
class Config:
    name: str = "hybrid+rerank"
    retriever: str = "hybrid"        # bm25 | dense | hybrid
    rerank: str | None = "cross-encoder"   # cross-encoder | overlap | None
    fanout: int = 40                 # candidates before rerank
    k: int = 5                       # passages given to the LLM
    llm: str = "ollama:llama3.1:8b"
    embedder: str = "st:BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class Answer:
    question: str
    text: str
    citations: list[int]
    contexts: list[Hit]
    abstained: bool
    trace: dict = field(default_factory=dict)


def parse_citations(text: str, n: int) -> list[int]:
    """[2] / [1, 3] / [1][3] -> 0-based passage indices, deduped, in range."""
    nums = [int(x) for grp in re.findall(r"\[([\d,\s]+)\]", text) for x in re.findall(r"\d+", grp)]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            out.append(x - 1)
    return out


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(hits))


class RAG:
    def __init__(self, cfg: Config, chunks, vectors: np.ndarray | None = None,
                 llm=None, embedder=None, reranker=None):
        self.cfg = cfg
        self.llm = llm or ChatModel(cfg.llm)
        needs_dense = cfg.retriever in {"dense", "hybrid"}
        emb = embedder or (Embedder(cfg.embedder) if needs_dense else None)
        bm25 = BM25Retriever(chunks) if cfg.retriever in {"bm25", "hybrid"} else None
        dense = DenseRetriever(chunks, emb, vectors) if needs_dense else None
        self.dense = dense
        self.retriever = {"bm25": bm25, "dense": dense,
                          "hybrid": HybridRetriever(bm25, dense, cfg.fanout) if bm25 and dense else None
                          }[cfg.retriever]
        if reranker is not None:
            self.reranker = reranker
        elif cfg.rerank == "cross-encoder":
            self.reranker = CrossEncoderReranker(cfg.reranker_model)
        elif cfg.rerank == "overlap":
            self.reranker = OverlapReranker()
        else:
            self.reranker = None

    @classmethod
    def from_index(cls, cfg: Config, index_dir: Path, **kw):
        chunks = pickle.loads((index_dir / "chunks.pkl").read_bytes())
        vec_path = index_dir / "vectors.npy"
        same_embedder = (index_dir / "embedder.txt").exists() and \
            (index_dir / "embedder.txt").read_text() == cfg.embedder
        vectors = np.load(vec_path) if vec_path.exists() and same_embedder else None
        return cls(cfg, chunks, vectors, **kw)

    def retrieve(self, question: str) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        n = self.cfg.fanout if self.reranker else self.cfg.k
        hits = self.retriever.search(question, n)
        t1 = time.perf_counter()
        if self.reranker:
            hits = self.reranker.rerank(question, hits, self.cfg.k)
        t2 = time.perf_counter()
        return hits[:self.cfg.k], {"retrieve_ms": 1000 * (t1 - t0), "rerank_ms": 1000 * (t2 - t1)}

    def answer(self, question: str) -> Answer:
        hits, trace = self.retrieve(question)
        user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
        r = self.llm(SYSTEM, user)
        text = r.text.strip()
        trace.update(generate_ms=1000 * r.seconds, input_tokens=r.input_tokens,
                     output_tokens=r.output_tokens, cost_usd=r.cost_usd)
        trace["total_ms"] = trace["retrieve_ms"] + trace["rerank_ms"] + trace["generate_ms"]
        return Answer(question, text, parse_citations(text, len(hits)), hits,
                      text.lower().strip(" .").startswith(ABSTAIN), trace)
