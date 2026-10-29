"""Run each ablation config in its own process, retrying crashes, then aggregate with --resume.

Why: a native crash (or the OS killing a process under memory pressure) in one config shouldn't discard
the others. Configs whose results/<name>/per_item.jsonl already exists are skipped.
Python runs with faulthandler, so a crash leaves a traceback in results/_logs/<name>.log.

    python scripts/run_ablation.py --config configs/ablation.yaml --attempts 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/ablation.yaml")
p.add_argument("--attempts", type=int, default=3)
a = p.parse_args()

names = [c["name"] for c in yaml.safe_load(Path(a.config).read_text())["configs"]]
logs = Path("results/_logs")
logs.mkdir(parents=True, exist_ok=True)

failed = []
for name in names:
    if Path("results", name, "per_item.jsonl").exists():
        print(f"== {name}: already done, skipping", flush=True)
        continue
    for attempt in range(1, a.attempts + 1):
        print(f"== {name}: attempt {attempt}", flush=True)
        with open(logs / f"{name}.log", "a", encoding="utf-8") as log:
            log.write(f"\n##### attempt {attempt}\n")
            log.flush()
            rc = subprocess.call([sys.executable, "-X", "faulthandler", "-m", "evals.run",
                                  "--config", a.config, "--only", name], stdout=log, stderr=subprocess.STDOUT)
        if rc == 0 and Path("results", name, "per_item.jsonl").exists():
            print(f"== {name}: done", flush=True)
            break
        print(f"== {name}: exit code {rc} (see {logs / (name + '.log')})", flush=True)
    else:
        failed.append(name)

print("failed configs:", failed or "none", flush=True)
if not failed:
    subprocess.call([sys.executable, "-m", "evals.run", "--config", a.config, "--resume"])
