"""Calibrate the LLM judge against human labels before trusting it.

Workflow:
  1. python -m evals.calibrate sheet --run results/hybrid+rerank --out data/label_sheet.csv
     -> a CSV with question / reference / candidate and an empty `human_correct` column.
  2. Label 50-100 rows yourself (1 = correct, 0 = incorrect). Mix in hard cases.
  3. python -m evals.calibrate score --run results/hybrid+rerank --labels data/label_sheet.csv
     -> agreement, Cohen's kappa, and the confusion matrix for the judge vs. you.

Rule of thumb used here: kappa >= 0.6 ("substantial") before reporting judge-based
numbers. If it's lower, read the disagreements, fix the rubric, bump PROMPT_VERSION,
and re-score. Report the kappa next to every judged metric.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(judge: list[int], human: list[int]) -> dict:
    tp = sum(j == 1 and h == 1 for j, h in zip(judge, human))
    tn = sum(j == 0 and h == 0 for j, h in zip(judge, human))
    fp = sum(j == 1 and h == 0 for j, h in zip(judge, human))
    fn = sum(j == 0 and h == 1 for j, h in zip(judge, human))
    n = len(judge)
    return {"n": n, "accuracy": round((tp + tn) / n, 4),
            "kappa": round(cohens_kappa(judge, human), 4),
            "judge_too_lenient(fp)": fp, "judge_too_strict(fn)": fn,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "trusted": cohens_kappa(judge, human) >= 0.6}


def _rows(run_dir: Path):
    return [json.loads(l) for l in (run_dir / "per_item.jsonl").read_text(encoding="utf-8").splitlines()]


def make_sheet(run_dir: Path, out: Path):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference", "candidate", "judge_correct", "human_correct"])
        for r in _rows(run_dir):
            if r["type"] != "unanswerable":
                w.writerow([r["id"], r["question"], r["reference"], r["answer"],
                            "" if r["judge_correct"] is None else int(r["judge_correct"]), ""])


def score(run_dir: Path, labels: Path) -> dict:
    with open(labels, encoding="utf-8") as f:
        human = {r["id"]: int(r["human_correct"]) for r in csv.DictReader(f)
                 if r["human_correct"].strip() in {"0", "1"}}
    pairs = [(int(r["judge_correct"]), human[r["id"]]) for r in _rows(run_dir)
             if r["id"] in human and r["judge_correct"] is not None]
    if not pairs:
        raise SystemExit("no overlapping labelled rows")
    res = agreement([j for j, _ in pairs], [h for _, h in pairs])
    (run_dir / "judge_calibration.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sheet"); s1.add_argument("--run", required=True); s1.add_argument("--out", required=True)
    s2 = sub.add_parser("score"); s2.add_argument("--run", required=True); s2.add_argument("--labels", required=True)
    a = p.parse_args(argv)
    if a.cmd == "sheet":
        make_sheet(Path(a.run), Path(a.out))
        print("wrote", a.out)
    else:
        print(json.dumps(score(Path(a.run), Path(a.labels)), indent=2))


if __name__ == "__main__":
    main()
