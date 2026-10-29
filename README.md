# rag-eval-lab

> **Status (work in progress).** Index build, golden set and the first three ablation configs (bm25, dense, hybrid) have run; the reranker configs and judge calibration are still to run, so the Results table below is not filled in yet.
>
> **Golden set provenance:** `data/golden/golden.jsonl` (111 items: 71 factoid/numeric, 20 multi-hop, 20 unanswerable) was drafted with `gpt-4o-mini` and then **verified by Claude (an AI), not by a human**: every evidence quote was checked to appear verbatim in the indexed filings, table column headers were checked for numeric rows, and unanswerable questions were checked for absence of the relevant terms across all 8 filings (`scripts/_build_golden.py`). Any judge-agreement number reported here is therefore agreement with **Claude-assigned labels, not human labels**.

A retrieval-augmented QA system over SEC 10-K filings, **built around its evaluation suite**. Every design choice (BM25 vs dense vs hybrid, reranking, chunk size, k) is an ablation row with a confidence interval, not a guess.

```
question ─► BM25 ─┐
                  ├─ RRF fusion ─► cross-encoder rerank (top-40 → 5) ─► LLM with numbered passages ─► answer + [citations] / "I don't know"
question ─► dense ┘
                                   every stage traced: latency · tokens · $ cost
```

**What gets measured, per config, on a hand-checked golden set:**

| Layer | Metric | How |
|---|---|---|
| Retrieval | hit@k, evidence recall, MRR | Does a retrieved passage contain the **verbatim gold evidence**? Deterministic, no LLM. |
| Generation | end-to-end correctness | LLM judge vs. reference answer, **calibrated against my own labels (Cohen's κ)** |
| Grounding | faithfulness, citation precision | Judge lists unsupported claims; do the cited passages actually hold the evidence? |
| Numbers | numeric match | Scale-aware: `$4.2 billion` = `4,200 million`, `(12)` = −12 |
| Safety | abstention | Says "I don't know" on unanswerable questions, and *doesn't* on answerable ones |
| Ops | p50/p95 latency, $/query | Per-stage traces |
| Stats | Δ vs. baseline | **Paired bootstrap 95% CI** on the same questions |

LLM and embeddings are pluggable: **Ollama (local, free) by default**, or OpenAI / Anthropic through an env var.

## Results

> Fill in from `results/ABLATION.md`.

| Config | Correct | Retrieval hit@5 | Faithful | Abstains correctly | p95 ms | Δ correct vs BM25 (95% CI) |
|---|---|---|---|---|---|---|
| bm25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | — |
| dense | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| hybrid | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| hybrid+rerank | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Judge agreement with human labels: κ = _TBD_ on _N_ items.

## Run it

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama pull qwen2.5:14b        # answerer + judge (local)

# 1. Corpus: recent 10-Ks from EDGAR (needs a User-Agent with your email)
python scripts/fetch_sec.py --tickers AAPL MSFT NVDA AMZN GOOGL META TSLA JPM --years 1 \
    --user-agent "Rohil Utture <your email>"

# 2. Indexes (chunk size is an ablation axis)
python scripts/build_index.py --src data/raw --out indexes/c200 --chunk-size 200
python scripts/build_index.py --src data/raw --out indexes/c400 --chunk-size 400 --overlap 80
python scripts/build_index.py --src data/raw --out indexes/c100 --chunk-size 100 --overlap 20

# 3. Golden set: LLM drafts, YOU verify (target ~120: 80 factoid/numeric, 20 multi-hop, 20 unanswerable)
python scripts/draft_golden.py --index indexes/c200 --n 150
#    review data/golden/draft.jsonl -> data/golden/golden.jsonl  (format in data/golden/example.jsonl)

# 4. Ablation
python -m evals.run --config configs/ablation.yaml

# 5. Calibrate the judge, then report κ with every judged number
python -m evals.calibrate sheet --run results/hybrid+rerank --out data/label_sheet.csv
#    fill human_correct (0/1) for ~60 rows
python -m evals.calibrate score --run results/hybrid+rerank --labels data/label_sheet.csv

# 6. Serve
RAG_INDEX=indexes/c200 uvicorn rag.api:app --port 8090
curl -X POST localhost:8090/ask -H "content-type: application/json" \
     -d '{"question": "What was NVIDIA total revenue in its most recent fiscal year?"}'
```

To use OpenAI or Anthropic, set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and change `llm:` / `judge:` in `configs/ablation.yaml` (e.g. `openai:gpt-4o-mini`). Edit `PRICES` in `rag/llm.py` to current list prices.

## Tests
`pytest -q` runs 14 tests offline with a hashing embedder and a fake LLM. They cover:
- chunk size and overlap, sentence splitting, keeping HTML table rows intact
- BM25 exact match and RRF math
- parsing citations
- number parsing, including filing-style negatives
- retrieval and citation metrics
- the paired bootstrap and Cohen's κ
- the judge never counting unparseable output as a pass
- end-to-end ablation, the calibration round trip, and the API

See **[DESIGN.md](DESIGN.md)**.
