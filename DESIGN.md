# Design notes

## Evaluation comes first
A RAG demo that answers five hand-picked questions proves nothing. The question that matters is: **when I change X, do answers get better, and how sure am I?** So the eval harness is the core of this repo, and the pipeline is the thing under test.

## Golden set
- **Each item has a question, a reference answer, *verbatim* evidence quotes, and a type**: factoid, numeric, multi_hop, or unanswerable.
- **Verbatim evidence makes retrieval metrics deterministic.** A retrieved passage "hits" if it contains the quote (after normalizing whitespace and table pipes). No LLM is involved, so retrieval and generation failures can be told apart.
- **The LLM drafts and a human verifies.**
  - The drafter's evidence must literally appear in its source chunk, or the draft is dropped.
  - I then check every item against the filing and add multi-hop and unanswerable questions myself. A set written only by a model is biased toward what models find easy.
- **Unanswerable questions** (another company, a year not in the corpus, a fact filings don't disclose) test the most important behavior in finance QA: not inventing numbers.

## Hybrid retrieval with RRF
- Filings mix exact tokens ("Item 7A", "$1,234 million", product names) with paraphrase-heavy questions. BM25 wins on the first kind, embeddings on the second.
- **Reciprocal rank fusion**, `Σ 1/(60 + rank)`, combines the two ranked lists without calibrating BM25 scores against cosine similarity, which live on different scales. k=60 is the value from the original RRF paper. It damps the influence of top ranks so that one list can't dominate.

## Cross-encoder reranking
- A bi-encoder embeds the query and the passage separately, so the passage vector can't depend on what was asked.
- A cross-encoder reads them together, with full attention across both. That is much more accurate, but O(candidates) model calls per query.
- So: retrieve 40 cheaply, rerank, keep 5. The ablation shows what this buys in correctness and what it costs in p95 latency.

## Chunking
- Paragraph-first packing to about N words with overlap; long paragraphs split at sentence boundaries.
- HTML table cells are joined with ` | ` so a row's label and its numbers stay in one chunk. Splitting a number from its row label is the most common way RAG over filings goes wrong.
- Chunk size (100/200/400) is an ablation, because the tradeoff is real:
  - small chunks give precise retrieval but lose context
  - large chunks keep context but dilute the embedding and use up the LLM's context budget

## Generation contract
- The model answers **only** from numbered passages and cites `[n]`.
- It copies numbers exactly.
- Otherwise it says exactly "I don't know". The exact string makes abstention machine-checkable.
- **Citation precision** checks whether the cited passages really contain the evidence. A model can cite `[2]` while getting the fact from its pretraining data.

## LLM-as-judge, and why to distrust it until it's calibrated
- **Binary verdicts with a one-line reason.** Yes/no is easier to calibrate than a 1–5 scale.
- **Strict JSON.** Unparseable output is recorded as `judge_parse_error`, never as a pass. A test enforces this.
- **Correctness and faithfulness are separate judgments:**
  - an answer can be correct but unfaithful (right number from model memory, unsupported by the passages)
  - an answer can be faithful but incorrect (wrong passage retrieved)

  Telling the two apart is how you know whether to fix retrieval or generation.
- **Calibration.** I label 50–100 items myself and compute **Cohen's κ** between the judge and me. I trust judged metrics only at κ ≥ 0.6, and report κ next to them. The confusion matrix shows whether the judge is too lenient or too strict, and that decides how to fix the rubric (and bump `PROMPT_VERSION`).
- The judge never sees which config produced an answer. The judge model should be at least as strong as the answering model.

## Statistics
- Configs are compared on the **same questions**, so the comparison is *paired*. Question difficulty varies far more than config quality, and unpaired CIs would bury real differences.
- A paired bootstrap over per-question correctness gives Δ, a 95% CI, and a p-value.
- With ~120 questions, differences under ~5 points usually won't be significant, and the table says so honestly.

## Tracing and cost
- Every answer records retrieve / rerank / generate milliseconds, input and output tokens, and dollar cost. Ollama counts as $0.
- The API logs the same fields per request, so online behavior can be compared with offline eval.

## Next steps
- Query rewriting or decomposition for multi-hop questions.
- Metadata filters (company, fiscal year), since many questions name both.
- Contextual chunk headers ("ACME 10-K 2024, Item 7").
- Long-context models vs. RAG at equal cost.
- Regression-gate the golden set in CI against a cached index.
