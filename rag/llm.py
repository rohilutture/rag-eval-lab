"""Provider-agnostic chat + embedding clients over plain HTTP (no vendor SDKs).

Pick with a spec string, e.g.
  LLM:    "ollama:llama3.1:8b"  "openai:gpt-4o-mini"  "anthropic:claude-haiku-4-5"
  Embed:  "ollama:nomic-embed-text"  "openai:text-embedding-3-small"
          "st:BAAI/bge-small-en-v1.5" (local sentence-transformers)  "hash:384" (tests)
Env vars: OLLAMA_HOST, OPENAI_API_KEY, ANTHROPIC_API_KEY.
Every call returns token usage so cost and latency can be traced per stage.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass

import httpx
import numpy as np

# USD per 1M tokens (input, output). Edit to current list prices; Ollama is local = 0.
PRICES = {
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4.1-mini": (0.40, 1.60),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float


class ChatModel:
    def __init__(self, spec: str, temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 120):
        self.spec = spec
        self.provider, self.model = spec.split(":", 1)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.http = httpx.Client(timeout=timeout)

    def __call__(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        fn = getattr(self, f"_{self.provider}")
        for attempt in range(4):
            try:
                text, i, o = fn(system, user)
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                # Retry transient failures (local runner crash / GPU memory pressure, 429, 5xx); fail fast on 4xx.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if attempt == 3 or (status is not None and status < 500 and status != 429):
                    raise
                time.sleep(5 * 2 ** attempt)
        pin, pout = PRICES.get(self.spec, (0.0, 0.0))
        return LLMResponse(text, i, o, time.perf_counter() - t0, (i * pin + o * pout) / 1e6)

    def _ollama(self, system, user):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = self.http.post(f"{host}/api/chat", json={
            "model": self.model, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"], j.get("prompt_eval_count", 0), j.get("eval_count", 0)

    def _openai(self, system, user):
        r = self.http.post("https://api.openai.com/v1/chat/completions",
                           headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                           json={"model": self.model, "temperature": self.temperature,
                                 "max_tokens": self.max_tokens,
                                 "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        return (j["choices"][0]["message"]["content"], j["usage"]["prompt_tokens"],
                j["usage"]["completion_tokens"])

    def _anthropic(self, system, user):
        r = self.http.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                    "anthropic-version": "2023-06-01"},
                           json={"model": self.model, "max_tokens": self.max_tokens,
                                 "temperature": self.temperature, "system": system,
                                 "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        text = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
        return text, j["usage"]["input_tokens"], j["usage"]["output_tokens"]


class Embedder:
    def __init__(self, spec: str, batch: int = 64):
        self.spec = spec
        self.provider, self.model = spec.split(":", 1)
        if self.provider not in {"hash", "st", "ollama", "openai"}:
            raise ValueError(f"unknown embedding provider in {spec!r}")
        self.batch = batch
        self._st_model = None
        self.http = httpx.Client(timeout=120)

    def __call__(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        """kind='query' lets models with asymmetric prefixes (nomic, bge) use them."""
        fn = getattr(self, f"_emb_{self.provider}")
        out = [fn(texts[s:s + self.batch], kind) for s in range(0, len(texts), self.batch)]
        v = np.vstack(out).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)

    def _emb_hash(self, texts, kind):
        """Deterministic bag-of-words hashing embedder: no downloads, good enough for
        tests (shared words -> similar vectors)."""
        d = int(self.model)
        m = np.zeros((len(texts), d), np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                m[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % d] += 1
        return m

    def _emb_st(self, texts, kind):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model)
        prefix = ("Represent this sentence for searching relevant passages: "
                  if kind == "query" and "bge" in self.model else "")
        return self._st_model.encode([prefix + t for t in texts], normalize_embeddings=True)

    def _emb_ollama(self, texts, kind):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        prefix = ("search_query: " if kind == "query" else "search_document: ") \
            if "nomic" in self.model else ""
        r = self.http.post(f"{host}/api/embed", json={"model": self.model,
                                                      "input": [prefix + t for t in texts]})
        r.raise_for_status()
        return np.array(r.json()["embeddings"])

    def _emb_openai(self, texts, kind):
        r = self.http.post("https://api.openai.com/v1/embeddings",
                           headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                           json={"model": self.model, "input": texts})
        r.raise_for_status()
        return np.array([d["embedding"] for d in r.json()["data"]])
