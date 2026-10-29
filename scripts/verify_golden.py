"""Check that every evidence quote in the golden set appears verbatim in an index's chunks.

Run after re-fetching filings / rebuilding an index on another machine: if the filings differ from the ones
the golden set was built against, this fails instead of silently producing wrong retrieval metrics.

    python scripts/verify_golden.py --index indexes/c200 --golden data/golden/golden.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--index", default="indexes/c200")
p.add_argument("--golden", default="data/golden/golden.jsonl")
a = p.parse_args()

norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
corpus = norm(" ".join(json.loads(l)["text"] for l in Path(a.index, "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l))
bad = []
n = 0
for line in Path(a.golden).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    g = json.loads(line)
    n += 1
    if any(norm(e) not in corpus for e in g["evidence"]):
        bad.append(g["id"])
print(f"{n} golden items checked; {len(bad)} with evidence not found verbatim in {a.index}")
if bad:
    print("failing ids:", ", ".join(bad))
    sys.exit(1)
