import csv
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import calibrate
from evals.judge import Judge
from evals.metrics import (GoldenItem, citation_precision, numbers, numeric_match,
                           paired_bootstrap, retrieval_metrics)
from evals.run import aggregate, markdown, run_config
from rag.ingest import chunk_text, read_document
from rag.llm import Embedder, LLMResponse
from rag.pipeline import RAG, Config, parse_citations
from rag.retrieve import BM25Retriever, Hit, rrf, tokenize

DOCS = {
    "ACME_10-K_2024": (
        "Item 7. Management's Discussion and Analysis\n\n"
        "Total revenue was $4.2 billion in fiscal 2024, up 12% from $3.75 billion in fiscal 2023. "
        "Growth was driven by the cloud segment.\n\n"
        "Research and development expense was $610 million, or 14.5% of revenue.\n\n"
        "Item 1A. Risk Factors\n\nOur largest customer accounted for 18% of revenue. "
        "Loss of this customer could materially harm results."),
    "GLOBEX_10-K_2024": (
        "Item 7. Management's Discussion and Analysis\n\n"
        "Globex net income was $95 million in 2024 compared with a net loss of $(12) million in 2023.\n\n"
        "Item 5. Market\n\nGlobex common stock trades on the NASDAQ under the symbol GBX. "
        "The company paid no dividends."),
}
GOLDEN = [
    GoldenItem("q1", "What was ACME's total revenue in fiscal 2024?", "$4.2 billion",
               ["Total revenue was $4.2 billion in fiscal 2024"], "numeric"),
    GoldenItem("q2", "How much did ACME spend on research and development?", "$610 million",
               ["Research and development expense was $610 million"], "numeric"),
    GoldenItem("q3", "Under what symbol does Globex stock trade?", "GBX",
               ["under the symbol GBX"], "factoid"),
    GoldenItem("q4", "Who is the CEO of Initech?", "", [], "unanswerable"),
]


class FakeLLM:
    """Answers by copying the context sentence that best overlaps the question, with
    a citation; abstains when overlap is weak. Doubles as a judge for JSON prompts."""

    def __call__(self, system, user):
        if "Candidate answer:" in user:  # correctness judge
            ref = re.search(r"Reference answer: (.*)", user).group(1)
            cand = re.search(r"Candidate answer: (.*)", user, re.S).group(1)
            ok = numeric_match(cand, ref) if numeric_match(cand, ref) is not None \
                else ref.lower() in cand.lower()
            return LLMResponse(json.dumps({"verdict": "correct" if ok else "incorrect",
                                           "reason": "fake"}), 10, 5, 0.0, 0.0)
        if "Passages:" in user:  # faithfulness judge
            return LLMResponse('{"unsupported_claims": [], "faithful": true}', 10, 5, 0.0, 0.0)
        q = set(tokenize(user.split("Question:")[-1])) - {"what", "was", "the", "in", "of", "how", "did", "is", "who"}
        best, best_n, best_i = None, 0, 0
        for m in re.finditer(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\]|\n\nQuestion:)", user, re.S):
            for sent in re.split(r"(?<=[.])\s+", m.group(2)):
                n = len(q & set(tokenize(sent)))
                if n > best_n:
                    best, best_n, best_i = sent, n, int(m.group(1))
        if best_n < 2:
            return LLMResponse("I don't know", 50, 3, 0.01, 0.0)
        return LLMResponse(f"{best} [{best_i}]", 50, 20, 0.01, 0.0001)


@pytest.fixture
def chunks():
    out = []
    for doc, text in DOCS.items():
        out += chunk_text(doc, text, size=40, overlap=10, title=doc)
    return out


def make_rag(chunks, retriever="hybrid", rerank="overlap", k=3):
    cfg = Config(name=retriever, retriever=retriever, rerank=rerank, k=k, fanout=10,
                 embedder="hash:256")
    return RAG(cfg, chunks, llm=FakeLLM(), embedder=Embedder("hash:256"))


# ------------------------------------------------------------------ ingestion
def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    cs = chunk_text("d", text, size=80, overlap=35)
    assert all(len(c.text.split()) <= 80 for c in cs)
    assert len(cs) > 3 and len({c.chunk_id for c in cs}) == len(cs)
    # overlap: the last paragraph of chunk i reappears at the start of chunk i+1
    assert cs[0].text.split("\n")[-1] == cs[1].text.split("\n")[0]
    assert text[cs[1].start:].startswith("Paragraph")


def test_long_paragraph_split_on_sentences():
    para = " ".join(f"Sentence number {i} has some words." for i in range(60))
    cs = chunk_text("d", para, size=50, overlap=0)
    assert len(cs) > 1 and all(c.text.rstrip().endswith(".") for c in cs)


def test_html_table_row_stays_on_one_line(tmp_path):
    p = tmp_path / "f.htm"
    p.write_text("<html><body><p>Results</p><table><tr><td>Revenue</td><td>$4,200</td>"
                 "<td>$3,750</td></tr></table><script>x=1</script></body></html>")
    text = read_document(p)
    assert any("Revenue" in l and "$4,200" in l for l in text.splitlines())
    assert "x=1" not in text


# ------------------------------------------------------------------ retrieval
def test_bm25_exact_token(chunks):
    hits = BM25Retriever(chunks).search("GBX symbol", 1)
    assert "GBX" in hits[0].chunk.text


def test_rrf_rewards_agreement(chunks):
    a, b, c = chunks[0], chunks[1], chunks[2]
    l1 = [Hit(a, 1, "x"), Hit(b, 1, "x")]
    l2 = [Hit(c, 1, "y"), Hit(a, 1, "y")]
    fused = rrf([l1, l2], 3)
    assert fused[0].chunk.chunk_id == a.chunk_id  # in both lists


def test_parse_citations():
    assert parse_citations("x [2] y [1, 3] z [2][9]", 5) == [1, 0, 2]


# ------------------------------------------------------------------ metrics
def test_numbers_and_numeric_match():
    assert numbers("$4.2 billion") == [4.2e9]
    assert numbers("a net loss of $(12) million")[0] == -12e6
    assert numeric_match("Revenue was $4,200 million.", "$4.2 billion")
    assert not numeric_match("Revenue was $4.0 billion.", "$4.2 billion")
    assert numeric_match("It grew 12%.", "12%")
    assert numeric_match("whatever", "GBX") is None


def test_retrieval_and_citation_metrics():
    ctx = ["noise", "Total revenue was $4.2 billion in fiscal 2024, up 12%"]
    r = retrieval_metrics(ctx, ["total revenue was $4.2 billion"])
    assert r == {"hit": 1.0, "evidence_recall": 1.0, "rr": 0.5}
    assert citation_precision([1, 0], ctx, ["total revenue was $4.2 billion"]) == 0.5
    assert retrieval_metrics(ctx, [])["hit"] is None


def test_paired_bootstrap_detects_improvement():
    a = [0] * 50 + [1] * 50
    b = [1] * 80 + [0] * 20
    d, lo, hi, p = paired_bootstrap(a, b)
    assert d > 0 and lo > 0 and p < 0.05
    d, lo, hi, p = paired_bootstrap(a, a)
    assert d == 0 and p == 1.0


def test_kappa():
    assert calibrate.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(calibrate.cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_judge_parse_error_is_not_a_pass():
    class Bad:
        def __call__(self, s, u):
            return LLMResponse("sure, looks right!", 1, 1, 0, 0)
    j = Judge(Bad())
    assert j.correctness("q", "a", "b")["correct"] is None
    assert j.faithfulness(["c"], "b")["faithful"] is None


# ------------------------------------------------------------------ end to end
def test_pipeline_answers_cites_and_abstains(chunks):
    rag = make_rag(chunks)
    a = rag.answer("What was ACME's total revenue in fiscal 2024?")
    assert "$4.2 billion" in a.text and a.citations and not a.abstained
    assert a.trace["total_ms"] >= 0 and "input_tokens" in a.trace
    assert rag.answer("Who is the CEO of Initech?").abstained


def test_ablation_run_and_report(chunks, tmp_path):
    judge = Judge(FakeLLM())
    results = {}
    for name, retr, rr in [("bm25", "bm25", None), ("hybrid+rerank", "hybrid", "overlap")]:
        cfg = Config(name=name, retriever=retr, rerank=rr, k=3, fanout=10, embedder="hash:256")
        results[name] = run_config(cfg, None, GOLDEN, judge, tmp_path / name, chunks=chunks,
                                   llm=FakeLLM(), embedder=Embedder("hash:256"))
    agg = results["hybrid+rerank"][1]
    assert agg["n"] == 4 and agg["abstain_on_unanswerable"] == 1.0
    assert agg["correct"] >= 0.75 and agg["retrieval_hit"] == 1.0
    assert agg["judge_errors"] == 0
    md = markdown(results, "bm25", "fake", "v2")
    assert "hybrid+rerank" in md and "p=" in md
    # calibration round trip: human agrees with judge everywhere -> kappa 1 or trivially trusted
    sheet = tmp_path / "sheet.csv"
    calibrate.make_sheet(tmp_path / "hybrid+rerank", sheet)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    for r in rows:
        r["human_correct"] = r["judge_correct"]
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    res = calibrate.score(tmp_path / "hybrid+rerank", sheet)
    assert res["accuracy"] == 1.0


def test_api(chunks, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import rag.api as api
    monkeypatch.setattr(api, "_rag", make_rag(chunks))
    monkeypatch.setenv("RAG_TRACES", str(tmp_path / "t.jsonl"))
    c = TestClient(api.app)
    r = c.post("/ask", json={"question": "How much did ACME spend on research and development?"}).json()
    assert "$610 million" in r["answer"] and r["citations"] and r["sources"]
    assert (tmp_path / "t.jsonl").read_text().count("\n") == 1
    assert c.post("/ask", json={"question": ""}).status_code == 422
