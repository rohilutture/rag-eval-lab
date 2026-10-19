"""Download recent 10-K filings from SEC EDGAR into data/raw/.

EDGAR requires a descriptive User-Agent with contact info and allows <= 10 req/s.

    python scripts/fetch_sec.py --tickers AAPL MSFT NVDA AMZN GOOGL META TSLA JPM \
        --years 2 --user-agent "Your Name your@email.com"
"""
import argparse
import json
import time
from pathlib import Path

import httpx

p = argparse.ArgumentParser()
p.add_argument("--tickers", nargs="+", required=True)
p.add_argument("--years", type=int, default=1, help="most recent N annual reports per company")
p.add_argument("--user-agent", required=True)
p.add_argument("--out", default="data/raw")
a = p.parse_args()

http = httpx.Client(headers={"User-Agent": a.user_agent}, timeout=60, follow_redirects=True)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

tickers = http.get("https://www.sec.gov/files/company_tickers.json").json()
cik_of = {v["ticker"]: int(v["cik_str"]) for v in tickers.values()}

manifest = []
for t in a.tickers:
    cik = cik_of.get(t.upper())
    if cik is None:
        print("unknown ticker", t)
        continue
    sub = http.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    recent = sub["filings"]["recent"]
    n = 0
    for form, acc, doc, date in zip(recent["form"], recent["accessionNumber"],
                                    recent["primaryDocument"], recent["reportDate"]):
        if form != "10-K":
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        name = f"{t.upper()}_10-K_{date}.htm"
        (out / name).write_bytes(http.get(url).content)
        manifest.append({"ticker": t.upper(), "form": form, "period": date, "url": url, "file": name})
        print("saved", name)
        time.sleep(0.2)
        n += 1
        if n >= a.years:
            break
    time.sleep(0.2)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
