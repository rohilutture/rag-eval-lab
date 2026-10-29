# rag-eval-lab

> **Status.** All 8 ablation configs have run on one setup (RTX 4090, `llama3.1:8b` answerer via Ollama, `gpt-4o-mini` judge), and the judge was calibrated. See Results below, including the limits of the evidence.
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

All 8 configs, same 111-question golden set (71 factoid/numeric, 20 multi-hop, 20 unanswerable) over 8 FY2025/26 10-Ks, answerer `llama3.1:8b` (Ollama, temperature 0), judge `gpt-4o-mini`, run on one RTX 4090 (full table with MRR and citation precision: `results/ABLATION.md`). Judge cost was about $0.02 per config.

| Config | Correct | Retrieval hit@k | Faithful | False abstain | p50 / p95 ms | Δ correct vs BM25 (95% CI) |
|---|---|---|---|---|---|---|
| bm25 | 0.766 | 0.813 | 0.726 | 0.198 | 366 / 1019 | — |
| dense | 0.676 | 0.648 | 0.765 | 0.253 | 393 / 918 | -0.090 [-0.171, -0.009] p=0.033 |
| hybrid | 0.766 | 0.747 | 0.811 | 0.187 | 439 / 826 | +0.000 [-0.081, +0.081] |
| hybrid+rerank | 0.811 | 0.791 | 0.756 | 0.143 | 533 / 1094 | +0.045 [-0.036, +0.126] |
| rerank, k=3 | 0.739 | 0.703 | 0.768 | 0.242 | 426 / 969 | -0.027 [-0.117, +0.063] |
| rerank, k=10 | 0.793 | 0.868 | 0.775 | 0.121 | 704 / 1432 | +0.027 [-0.045, +0.108] |
| rerank, 400-word chunks | 0.739 | 0.758 | 0.836 | 0.198 | 669 / 1186 | -0.027 [-0.108, +0.063] |
| rerank, 100-word chunks | 0.802 | 0.802 | 0.779 | 0.154 | 433 / 955 | +0.036 [-0.045, +0.117] |

**How to read this (honestly):**
- With 111 questions, most differences between configs are **within noise**. Only dense < BM25 has an interval that excludes zero (p = 0.033), and that p-value is not corrected for the 7 comparisons, so treat it as suggestive. A plausible reason is that filings are full of exact figures and names, which keyword search handles well.
- Reranking is the best config (+4.5 points correct, fewest false abstains), but its interval spans zero, so this run does not prove it helps.
- More passages (k=10) gave the best retrieval hit rate (0.868) but not better correctness than k=5. k=3 was worse.
- Abstention on the 20 unanswerable questions was perfect in 7 of 8 configs (0.95 for the 400-word-chunk config), but every config also wrongly refuses 12-25% of answerable questions.
- Latency is on a 4090 with the reranker on the GPU; it will differ on other hardware.

**Judge calibration (not a human-label result).** I compared the judge to correctness labels on the answerable questions of the `hybrid+rerank` run (91 items). **The labels were assigned by Claude (an AI), blind to the judge's verdicts, not by a human**, using a stated rubric (`scripts/_claude_labels.py`, `data/label_sheet_claude.csv`). On the 78 items the judge actually scored (13 "I don't know" answers are never judged), agreement is 94.9% with Cohen's κ = 0.72 (bootstrap 95% CI 0.37 to 0.94, wide because there are only 6 negatives); counting the 13 abstentions as incorrect, 95.6% and κ = 0.88 over all 91. The 4 disagreements are borderline calls (an answer with extra factors, a figure given in a table without being stated, a related but different statement, and two numbers given without their difference). A human-labeled κ has not been measured. Calibration was done on one config's answers only.

**Golden set provenance.** Drafted with `gpt-4o-mini`, then verified by Claude, not a human: every evidence quote appears verbatim in the indexed filings (`scripts/verify_golden.py`), table column headers were checked for numeric rows, and unanswerable questions were checked for absence of the relevant terms. The multi-hop and unanswerable questions were written by Claude.

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
