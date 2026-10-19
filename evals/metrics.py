"""Deterministic metrics (no LLM needed): retrieval hits, citations, numbers, abstention."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GoldenItem:
    id: str
    question: str
    answer: str                       # reference answer ("" for unanswerable)
    evidence: list[str] = field(default_factory=list)  # verbatim quotes from the corpus
    type: str = "factoid"             # factoid | numeric | multi_hop | unanswerable

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem(**json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("—", "-")
    s = re.sub(r"\s*\|\s*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_evidence(text: str, evidence: str) -> bool:
    return norm(evidence) in norm(text)


def retrieval_metrics(contexts: list[str], evidence: list[str]) -> dict:
    """hit = any evidence quote appears in the retrieved passages;
    evidence_recall = share of quotes found (matters for multi-hop);
    rr = 1/rank of the first passage holding any quote."""
    if not evidence:
        return {"hit": None, "evidence_recall": None, "rr": None}
    found = [any(contains_evidence(c, e) for c in contexts) for e in evidence]
    rank = next((i for i, c in enumerate(contexts) if any(contains_evidence(c, e) for e in evidence)),
                None)
    return {"hit": float(any(found)), "evidence_recall": float(np.mean(found)),
            "rr": 1 / (rank + 1) if rank is not None else 0.0}


def citation_precision(cited: list[int], contexts: list[str], evidence: list[str]):
    """Of the passages the answer cites, how many actually hold gold evidence."""
    if not cited or not evidence:
        return None
    return float(np.mean([any(contains_evidence(contexts[i], e) for e in evidence) for i in cited]))


_NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?\s*(?:%|percent|billion|million|thousand|bn|mm|m|k)?",
                  re.I)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0).strip().lower()
        neg = tok.startswith("-") or ("(" in tok and ")" in tok)  # (123) = negative in filings
        unit = re.search(r"(billion|million|thousand|bn|mm|m|k)$", tok)
        digits = re.sub(r"[^\d.]", "", re.sub(r"[a-z%]+$", "", tok))
        if not digits or digits == ".":
            continue
        v = float(digits.rstrip(".")) * (_SCALE[unit.group(1)] if unit else 1)
        out.append(-v if neg else v)
    return out


def numeric_match(pred: str, ref: str, rel_tol: float = 0.005) -> bool | None:
    """True if every number in the reference appears in the prediction (scale-aware:
    '$4.2 billion' == '4,200 million'). None if the reference has no numbers."""
    ref_nums = [n for n in numbers(ref) if abs(n) >= 1 or n == 0]
    if not ref_nums:
        return None
    pred_nums = numbers(pred)
    return all(any(abs(p - r) <= rel_tol * max(abs(r), 1e-9) for p in pred_nums) for r in ref_nums)


def abstention_correct(item: GoldenItem, abstained: bool) -> bool:
    return abstained == (not item.answerable)


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 0):
    """95% CI and two-sided p-value for mean(b - a) on the SAME questions.
    Pairing matters: question difficulty varies a lot, so unpaired CIs are far
    wider and would hide real differences between configs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))
