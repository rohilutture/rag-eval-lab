"""Ablation runner: every pipeline config x the same golden set -> one comparison table.

    python -m evals.run --config configs/ablation.yaml

For each config: per-item results (results/<name>/per_item.jsonl) and an
aggregate; then results/ABLATION.md with a paired-bootstrap CI on end-to-end
correctness against the baseline config, so "hybrid+rerank is better" comes with
an error bar rather than a vibe.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from rag.llm import ChatModel
from rag.pipeline import RAG, Config

from .judge import Judge
from .metrics import (GoldenItem, abstention_correct, citation_precision, load_golden,
                      numeric_match, paired_bootstrap, retrieval_metrics)


def evaluate_item(rag: RAG, item: GoldenItem, judge: Judge) -> dict:
    ans = rag.answer(item.question)
    ctx = [h.chunk.text for h in ans.contexts]
    row = {"id": item.id, "type": item.type, "question": item.question,
           "reference": item.answer, "answer": ans.text, "abstained": ans.abstained,
           "citations": ans.citations, "context_ids": [h.chunk.chunk_id for h in ans.contexts],
           **retrieval_metrics(ctx, item.evidence),
           "citation_precision": citation_precision(ans.citations, ctx, item.evidence),
           "numeric_match": numeric_match(ans.text, item.answer) if item.answerable else None,
           "abstention_correct": abstention_correct(item, ans.abstained),
           "judge_correct": None, "faithful": None, "unsupported": [], "judge_cost_usd": 0.0,
           **{f"t_{k}": v for k, v in ans.trace.items()}}
    if item.answerable and not ans.abstained:
        c = judge.correctness(item.question, item.answer, ans.text)
        f = judge.faithfulness(ctx, ans.text)
        row.update(judge_correct=c["correct"], judge_reason=c["reason"], faithful=f["faithful"],
                   unsupported=f["unsupported"], judge_cost_usd=c["cost_usd"] + f["cost_usd"])
    # End-to-end: unanswerable -> must abstain; answerable -> judge says correct.
    row["correct"] = (ans.abstained if not item.answerable
                      else bool(row["judge_correct"]) and not ans.abstained)
    return row


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    lat = [r["t_total_ms"] for r in rows]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["correct"])
    return {
        "n": len(rows), "correct": _mean([r["correct"] for r in rows]),
        "correct_by_type": {k: _mean(v) for k, v in sorted(by_type.items())},
        "retrieval_hit": _mean([r["hit"] for r in ans]),
        "evidence_recall": _mean([r["evidence_recall"] for r in ans]),
        "mrr": _mean([r["rr"] for r in ans]),
        "faithful": _mean([r["faithful"] for r in ans]),
        "citation_precision": _mean([r["citation_precision"] for r in ans]),
        "numeric_match": _mean([r["numeric_match"] for r in ans]),
        "abstain_on_unanswerable": _mean([r["abstained"] for r in una]),
        "false_abstain": _mean([r["abstained"] for r in ans]),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "cost_per_query_usd": round(float(np.mean([r.get("t_cost_usd", 0) for r in rows])), 6),
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in rows), 4),
        "judge_errors": sum(r.get("judge_reason") == "judge_parse_error" for r in rows),
    }


def run_config(cfg: Config, index_dir: Path | None, golden, judge, out: Path, chunks=None,
               **rag_kw) -> tuple[list[dict], dict]:
    rag = RAG.from_index(cfg, index_dir, **rag_kw) if index_dir else RAG(cfg, chunks, **rag_kw)
    rows = [evaluate_item(rag, it, judge) for it in golden]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_item.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = aggregate(rows)
    (out / "summary.json").write_text(json.dumps({"config": cfg.__dict__, **agg}, indent=2))
    return rows, agg


COLS = [("correct", "End-to-end correct"), ("retrieval_hit", "Retrieval hit@k"), ("mrr", "MRR"),
        ("faithful", "Faithful"), ("citation_precision", "Citation prec."),
        ("abstain_on_unanswerable", "Abstains when it should"), ("false_abstain", "False abstain"),
        ("latency_p50_ms", "p50 ms"), ("latency_p95_ms", "p95 ms"),
        ("cost_per_query_usd", "$/query")]


def markdown(results: dict[str, tuple[list, dict]], baseline: str, judge_spec: str,
             judge_version: str) -> str:
    lines = [f"# RAG ablation\n\nJudge: `{judge_spec}` (prompt {judge_version}). "
             f"Δ = paired bootstrap vs `{baseline}` on end-to-end correctness, 95% CI.\n",
             "| Config | " + " | ".join(h for _, h in COLS) + " | Δ correct (95% CI) |",
             "|---" * (len(COLS) + 2) + "|"]
    base_rows = results[baseline][0]
    for name, (rows, agg) in results.items():
        if name == baseline:
            delta = "—"
        else:
            d, lo, hi, p = paired_bootstrap([r["correct"] for r in base_rows],
                                            [r["correct"] for r in rows])
            delta = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}"
        fmt = lambda v: "—" if v is None else str(v)
        lines.append(f"| {name} | " + " | ".join(fmt(agg[k]) for k, _ in COLS) + f" | {delta} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ablation.yaml")
    p.add_argument("--only", nargs="*", help="run a subset of config names")
    p.add_argument("--resume", action="store_true",
                   help="reuse saved results/<name>/per_item.jsonl instead of re-running that config")
    a = p.parse_args(argv)
    spec = yaml.safe_load(Path(a.config).read_text())
    golden = load_golden(Path(spec["golden"]))
    judge = Judge(ChatModel(spec["judge"], max_tokens=300))
    allowed = {f.name for f in fields(Config)}
    results = {}
    for c in spec["configs"]:
        merged = {**spec.get("defaults", {}), **c}
        if a.only and merged["name"] not in a.only:
            continue
        cfg = Config(**{k: v for k, v in merged.items() if k in allowed})
        saved = Path("results") / cfg.name / "per_item.jsonl"
        if a.resume and saved.exists():
            rows = [json.loads(l) for l in saved.read_text(encoding="utf-8").splitlines() if l]
            results[cfg.name] = (rows, aggregate(rows))
            print(f"== {cfg.name} (loaded saved results)")
            continue
        print(f"== {cfg.name}")
        results[cfg.name] = run_config(cfg, Path(merged["index"]), golden, judge,
                                       Path("results") / cfg.name)
        print(json.dumps(results[cfg.name][1]))
    base = spec.get("baseline", next(iter(results)))
    md = markdown(results, base if base in results else next(iter(results)), spec["judge"],
                  judge.version)
    Path("results/ABLATION.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
