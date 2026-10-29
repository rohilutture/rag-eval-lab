"""One-off: assemble data/golden/golden.jsonl from verified candidates, multi-hop and unanswerable items.

Verification done by Claude (not a human): every evidence quote must appear verbatim in the indexed corpus,
numeric table rows were checked against their column headers, and unanswerable questions were checked
for absence of the relevant terms across all 8 filings.
"""
import collections
import json
import re

cand = {r["id"]: r for r in json.load(open("data/golden/_candidates.json", encoding="utf-8"))}
ch = [json.loads(l) for l in open("indexes/c200/chunks.jsonl", encoding="utf-8")]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
corpus = norm(" ".join(c["text"] for c in ch))

KEEP = ("q0005 q0007 q0010 q0012 q0015 q0021 q0022 q0023 q0025 q0027 q0028 q0029 q0031 q0033 q0035 q0038 q0039 "
        "q0042 q0043 q0044 q0046 q0050 q0051 q0052 q0053 q0055 q0057 q0059 q0062 q0065 q0066 q0067 q0068 q0073 "
        "q0076 q0078 q0081 q0082 q0083 q0088 q0093 q0095 q0097 q0098 q0100 q0101 q0102 q0103 q0104 q0107 q0109 "
        "q0112 q0113 q0114 q0116 q0118 q0119 q0125 q0128 q0130 q0132 q0136 q0137 q0138 q0139 q0140 q0142 q0144 "
        "q0145 q0148 q0149").split()


def clean(q):
    q = re.sub(r"\b[A-Z]{2,5}_10-K_[\d-]+\b", "its 10-K", q)
    q = re.sub(r",? ?(as reported|as stated|as outlined|as referenced)?,? ?(in|according to|per) (the )?(filing( period)?,? )?(the company's|its) 10-K( filing)?", "", q)
    q = re.sub(r" (in|for|of) the (filing|filing period),? the company's 10-K", "", q)
    q = q.replace("the company's 10-K", "its 10-K")
    q = re.sub(r"\b[A-Z]{2,5} 10-K filing", "10-K filing", q)
    return re.sub(r"\s+", " ", q).strip(" ,")


gold = []
for i in KEEP:
    r = cand[i]
    q, a = clean(r["question"]), r["answer"]
    if i == "q0073":  # first column of the table is 2025 (11.8%), 2024 is 17.6%
        q, a = "What was Meta Platforms' effective tax rate for the year ended December 31, 2025?", "11.8%"
    gold.append({"question": q, "answer": a, "evidence": r["evidence"], "type": r["type"]})

e = lambda i: cand[i]["evidence"][0]
MH = [
    ("By how much did Microsoft's depreciation expense increase from fiscal 2025 to fiscal 2026?",
     "$12.3 billion ($34.3 billion in fiscal 2026 vs $22.0 billion in fiscal 2025)", [e("q0012")]),
    ("By how much did Tesla's inventory write-downs increase from 2024 to 2025?",
     "$120 million ($310 million in 2025 vs $190 million in 2024)", [e("q0112")]),
    ("By how many percentage points did the share of NVIDIA's revenue from customers headquartered outside the United States fall from fiscal year 2025 to fiscal year 2026?",
     "10 percentage points (41% to 31%)", [e("q0093")]),
    ("By how much did Tesla's net investment in sales-type leases decrease between December 31, 2024 and December 31, 2025?",
     "$200 million ($440 million to $240 million)", ["Net investment in sales-type leases | $ | 240 | $ | 440 |"]),
    ("By how much did Tesla's total other long-term liabilities grow from December 31, 2024 to December 31, 2025?",
     "$2,365 million ($10,495 million to $12,860 million)", [e("q0007")]),
    ("By how much did JPMorgan Chase's total pledged assets increase from December 31, 2024 to December 31, 2025?",
     "$159.2 billion ($1,174.4 billion to $1,333.6 billion)", ["Total pledged assets | $ | 1,333.6 | $ | 1,174.4 |"]),
    ("By how much did JPMorgan Chase's pledged loans increase from December 31, 2024 to December 31, 2025?",
     "$22.9 billion ($740.9 billion to $763.8 billion)", ["Loans | 763.8 | 740.9 |"]),
    ("By how much did JPMorgan Chase's total stockholders' equity increase from 2024 to 2025?",
     "$17,680 million ($344,758 million to $362,438 million)", [e("q0095")]),
    ("By how much did Amazon's net property and equipment grow from 2024 to 2025?",
     "$104,360 million ($252,665 million to $357,025 million)", [e("q0015")]),
    ("By how much did Amazon's operating income grow from 2024 to 2025?",
     "$11.4 billion ($68.6 billion to $80.0 billion)", [e("q0068")]),
    ("By how much did the notional value of NVIDIA's foreign currency contracts designated as accounting hedges increase between January 26, 2025 and January 25, 2026?",
     "$341 million ($1,424 million to $1,765 million)", ["Designated as accounting hedges | $ | 1,765 | $ | 1,424 |"]),
    ("How many more reportable segments does Apple have than Meta Platforms?",
     "3 (Apple has five; Meta has two)", [e("q0088"), e("q0042")]),
    ("Do Meta Platforms and Alphabet report the same number of reportable segments?",
     "Yes, both report two (Meta: Family of Apps and Reality Labs; Alphabet: Google Services and Google Cloud)", [e("q0042"), e("q0130")]),
    ("How much larger is the administrative fine the IDPC issued to Meta Platforms Ireland than the fine the European Commission imposed on Apple in the Article 5(4) Investigation?",
     "About EUR 700 million (EUR 1.2 billion vs EUR 500 million)", [e("q0027"), e("q0081")]),
    ("What is the combined amount of the European Commission's fine on Apple in the Article 5(4) Investigation and the European Commission's fine on Meta related to Facebook Marketplace?",
     "About EUR 1,298 million (EUR 500 million plus approximately EUR 798 million)", [e("q0081"), e("q0046")]),
    ("Which was larger: Microsoft's total purchase price for Activision Blizzard or the amount remaining available for Alphabet's Class A and Class C share repurchases, and by how much?",
     "Microsoft's Activision Blizzard purchase price ($75.4 billion) was larger, by $5.9 billion (vs $69.5 billion remaining at Alphabet)", [e("q0028"), e("q0021")]),
    ("Which accounting firm audits the effectiveness of Tesla's internal control over financial reporting, and since what year has Apple's auditor served the company?",
     "PricewaterhouseCoopers LLP audits Tesla's internal control; Apple's auditor (Ernst & Young LLP) has served since 2009", [e("q0140"), e("q0118")]),
    ("Which has the higher PCAOB ID number: the principal accountant for Microsoft or the independent registered public accounting firm for Meta Platforms?",
     "Meta Platforms' (PCAOB ID No. 42) is higher than Microsoft's Deloitte & Touche LLP (PCAOB ID No. 34)", [e("q0083"), e("q0076")]),
    ("Of Apple, Meta Platforms and Alphabet, which companies report exactly two segments?",
     "Meta Platforms and Alphabet (Apple reports five)", [e("q0088"), e("q0042"), e("q0130")]),
    ("How many research and development employees does NVIDIA have, and how many votes per share does each share of Alphabet's Class B common stock carry?",
     "NVIDIA has 31,000 employees engaged in research and development; each Alphabet Class B share carries 10 votes", [e("q0066"), e("q0044")]),
]
bad = [(q[:60], ev[:70]) for q, a, evs in MH for ev in evs if norm(ev) not in corpus]
print("multi-hop evidence not verbatim in corpus:", bad)
for q, a, evs in MH:
    gold.append({"question": q, "answer": a, "evidence": evs, "type": "multi_hop"})
for tk, q in json.load(open("data/golden/_unans.json")):
    gold.append({"question": q, "answer": "", "evidence": [], "type": "unanswerable"})

for n, g in enumerate(gold, 1):
    g["id"] = f"q{n:04d}"
nf = [g["id"] for g in gold if any(norm(x) not in corpus for x in g["evidence"])]
print("golden items with evidence not in corpus:", nf)
print(len(gold), dict(collections.Counter(g["type"] for g in gold)))
with open("data/golden/golden.jsonl", "w", encoding="utf-8") as f:
    for g in gold:
        f.write(json.dumps({k: g[k] for k in ("id", "question", "answer", "evidence", "type")}, ensure_ascii=False) + "\n")
for g in gold[:8]:
    print(g["id"], g["question"])
