"""LLM-as-judge for what string metrics can't score: correctness and faithfulness.

Rules that keep a judge trustworthy:
  * binary verdicts with a short reason (Likert scales drift; yes/no calibrates)
  * the judge sees the reference answer for correctness, the passages for faithfulness
  * strict JSON output, parsed defensively; unparseable = recorded as an error, never
    silently counted as pass
  * the judge never learns which pipeline config produced the answer
  * prompts are versioned, and a version is only trusted after calibration against
    human labels (see calibrate.py)
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "v2"

CORRECTNESS = """You grade answers to questions about company filings.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

The candidate is CORRECT if it states the same facts as the reference. Wording may differ.
Numbers must match after unit conversion (e.g. $4.2 billion = $4,200 million); rounding
to the reference's precision is fine. Extra true detail is fine. A candidate that hedges
between several values, or that says it does not know, is INCORRECT.

Reply with JSON only: {{"verdict": "correct" | "incorrect", "reason": "<one sentence>"}}"""

FAITHFULNESS = """You check whether an answer is fully supported by source passages.

Passages:
{context}

Answer: {candidate}

List every factual claim in the answer that is NOT directly supported by the passages.
Ignore citation markers like [1]. An answer that only says "I don't know" has no claims.

Reply with JSON only: {{"unsupported_claims": ["..."], "faithful": true | false}}"""


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Judge:
    def __init__(self, llm):
        self.llm = llm
        self.version = PROMPT_VERSION

    def correctness(self, question: str, reference: str, candidate: str) -> dict:
        r = self.llm("You are a strict, fair grader.",
                     CORRECTNESS.format(question=question, reference=reference, candidate=candidate))
        j = _parse(r.text)
        if not j or j.get("verdict") not in {"correct", "incorrect"}:
            return {"correct": None, "reason": "judge_parse_error", "raw": r.text[:300],
                    "cost_usd": r.cost_usd}
        return {"correct": j["verdict"] == "correct", "reason": j.get("reason", ""),
                "cost_usd": r.cost_usd}

    def faithfulness(self, contexts: list[str], candidate: str) -> dict:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        r = self.llm("You are a meticulous fact checker.",
                     FAITHFULNESS.format(context=ctx, candidate=candidate))
        j = _parse(r.text)
        if not j or "faithful" not in j:
            return {"faithful": None, "unsupported": [], "reason": "judge_parse_error",
                    "cost_usd": r.cost_usd}
        return {"faithful": bool(j["faithful"]), "unsupported": j.get("unsupported_claims", []),
                "cost_usd": r.cost_usd}
