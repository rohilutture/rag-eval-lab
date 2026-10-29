"""Claude-assigned correctness labels for the hybrid+rerank run (NOT human labels).

Labeled blind to the judge's verdicts: only question / reference / candidate were shown.
Rubric: correct = the candidate states the reference facts accurately (numbers match, equivalent units,
nothing contradicting the reference). Partial answers to multi-part questions, "I don't know" on an
answerable question, empty answers, and a different fact than the one asked are incorrect.
Extra non-contradictory detail is allowed. Writes data/label_sheet_claude.csv for `evals.calibrate score`.
"""
import csv
from pathlib import Path

ONES = """q0001 q0004 q0005 q0006 q0007 q0008 q0009 q0010 q0011 q0012 q0013 q0014 q0015 q0016 q0017 q0018 q0019 q0020
q0021 q0022 q0024 q0025 q0026 q0028 q0029 q0030 q0031 q0032 q0033 q0035 q0036 q0037 q0039 q0040 q0041 q0042 q0043
q0044 q0045 q0046 q0047 q0048 q0050 q0052 q0053 q0055 q0056 q0057 q0058 q0059 q0060 q0061 q0062 q0064 q0065 q0069
q0070 q0072 q0073 q0074 q0076 q0079 q0080 q0082 q0083 q0085 q0086 q0087 q0088 q0090""".split()
ZEROS = """q0002 q0003 q0023 q0027 q0034 q0038 q0049 q0051 q0054 q0063 q0066 q0067 q0068 q0071 q0075 q0077 q0078 q0081
q0084 q0089 q0091""".split()
labels = {**{i: 1 for i in ONES}, **{i: 0 for i in ZEROS}}

src = Path("data/label_sheet.csv")
rows = list(csv.DictReader(open(src, encoding="utf-8")))
missing = [r["id"] for r in rows if r["id"] not in labels]
extra = [i for i in labels if i not in {r["id"] for r in rows}]
assert not missing and not extra, (missing, extra)
with open("data/label_sheet_claude.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "question", "reference", "candidate", "judge_correct", "human_correct"])  # column name required by evals.calibrate; values are CLAUDE labels
    for r in rows:
        w.writerow([r["id"], r["question"], r["reference"], r["candidate"], r["judge_correct"], labels[r["id"]]])
print(len(rows), "labels written;", sum(labels.values()), "correct,", len(labels) - sum(labels.values()), "incorrect")
