"""Draft golden-set candidates with an LLM, for a HUMAN to review.

The LLM proposes (question, answer, verbatim evidence) from a random chunk.
Drafts are only candidates: you check every one against the source, fix or drop
it, and add your own multi-hop and unanswerable questions. A golden set written
entirely by a model tends to test what that model already finds easy.

    python scripts/draft_golden.py --index indexes/c200 --n 150 --llm ollama:qwen2.5:14b
    # review data/golden/draft.jsonl -> save the keepers as data/golden/golden.jsonl
"""
import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.metrics import contains_evidence
from rag.llm import ChatModel

PROMPT = """From this passage of a company filing, write ONE question an analyst might ask
whose answer is stated in the passage. Prefer specific facts and figures.

Passage ({title}):
{text}

Reply with JSON only:
{{"question": "...", "answer": "...", "evidence": "<exact sentence or table row copied from the passage>",
  "type": "factoid" | "numeric"}}"""

p = argparse.ArgumentParser()
p.add_argument("--index", default="indexes/c200")
p.add_argument("--n", type=int, default=150)
p.add_argument("--llm", default="ollama:qwen2.5:14b")
p.add_argument("--out", default="data/golden/draft.jsonl")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

chunks = pickle.loads((Path(a.index) / "chunks.pkl").read_bytes())
# Skip boilerplate-looking chunks (cover pages, tables of contents, signatures).
chunks = [c for c in chunks if len(c.text.split()) > 80 and len(re.findall(r"\d", c.text)) > 3]
random.Random(a.seed).shuffle(chunks)
llm = ChatModel(a.llm, temperature=0.3, max_tokens=400)
kept = 0
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for c in chunks:
        if kept >= a.n:
            break
        r = llm("You write evaluation questions.", PROMPT.format(title=c.title, text=c.text))
        m = re.search(r"\{.*\}", r.text, re.S)
        try:
            j = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            j = None
        # Keep only drafts whose evidence really is verbatim in the source chunk.
        if not j or not contains_evidence(c.text, j.get("evidence", "")):
            continue
        kept += 1
        f.write(json.dumps({"id": f"q{kept:04d}", "question": j["question"], "answer": j["answer"],
                            "evidence": [j["evidence"]], "type": j.get("type", "factoid"),
                            "_source_chunk": c.chunk_id, "_doc": c.doc_id}) + "\n")
print(f"{kept} drafts -> {a.out}. Review every one by hand before using it.")
