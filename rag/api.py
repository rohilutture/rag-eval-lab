"""FastAPI service: POST /ask {"question": ...} -> answer, citations, sources, trace.

    RAG_INDEX=indexes/c200 RAG_LLM=ollama:llama3.1:8b uvicorn rag.api:app --port 8090
Each request's trace (stage latencies, tokens, cost) is appended to traces.jsonl,
the same fields the offline evaluation reports, so online and offline line up.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import RAG, Config

app = FastAPI(title="rag-eval-lab")
_rag: RAG | None = None


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def get_rag() -> RAG:
    global _rag
    if _rag is None:
        cfg = Config(llm=os.environ.get("RAG_LLM", Config.llm),
                     embedder=os.environ.get("RAG_EMBEDDER", Config.embedder),
                     rerank=os.environ.get("RAG_RERANK", "cross-encoder") or None)
        _rag = RAG.from_index(cfg, Path(os.environ.get("RAG_INDEX", "indexes/c200")))
    return _rag


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    a = get_rag().answer(body.question)
    sources = [{"n": i + 1, "doc": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id,
                "snippet": h.chunk.text[:300]} for i, h in enumerate(a.contexts)]
    rec = {"ts": time.time(), "question": body.question, "abstained": a.abstained, **a.trace}
    with open(os.environ.get("RAG_TRACES", "traces.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"answer": a.text, "abstained": a.abstained, "citations": [c + 1 for c in a.citations],
            "sources": sources, "trace": a.trace}
