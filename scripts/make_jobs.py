"""Append extract -> score -> report job chains to the queue.

  python scripts/make_jobs.py --config configs/phase0_replication.yaml --size 1.4b --steps 143000 --dry-run-first
  python scripts/make_jobs.py --config configs/X.yaml --size 1.4b --steps logspaced     # Phase 3 trajectory

With --dry-run-first, each chain starts with a --limit 8 extraction + scoring of the same unit;
the full run only starts if the dry run succeeds. Existing ids in the queue are not duplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topodur.models import FINAL_STEP, log_spaced_steps  # noqa: E402

PY = "uv run --frozen python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--steps", default=str(FINAL_STEP), help="comma list, or 'logspaced'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1, help="2 for 6.9B with --multi-gpu")
    ap.add_argument("--dry-run-first", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--queue", default=str(ROOT / "jobs" / "queue.jsonl"))
    a = ap.parse_args()

    steps = log_spaced_steps() if a.steps == "logspaced" else [int(s) for s in a.steps.split(",")]
    cfg = a.config
    tag = Path(cfg).stem
    multi = " --multi-gpu" if a.gpus > 1 else ""
    qpath = Path(a.queue)
    existing = set()
    if qpath.exists():
        existing = {json.loads(l)["id"] for l in qpath.read_text().splitlines() if l.strip() and not l.startswith("#")}

    new = []
    for step in steps:
        unit = f"{a.size}_s{a.seed}_step{step}"
        base = f"{tag}-{unit}"
        common = f"--config {cfg} --size {a.size} --step {step} --seed {a.seed}{multi}"
        after: list[str] = []
        if a.dry_run_first:
            new.append({"id": f"{base}-dry-extract", "gpus": a.gpus, "retries": 0,
                        "cmd": f"{PY} scripts/extract.py {common} --limit 8 --dry-run"})
            new.append({"id": f"{base}-dry-score", "gpus": 0, "retries": 0, "after": [f"{base}-dry-extract"],
                        "cmd": f"{PY} scripts/score.py --config {cfg} --unit {unit} --limit 2 --dry-run --workers 4 --force"})
            after = [f"{base}-dry-score"]
        new.append({"id": f"{base}-extract", "gpus": a.gpus, "after": after,
                    "cmd": f"{PY} scripts/extract.py {common}"})
        new.append({"id": f"{base}-score", "gpus": 0, "after": [f"{base}-extract"],
                    "cmd": f"{PY} scripts/score.py --config {cfg} --unit {unit} --workers 16"})
        if not a.no_report:
            new.append({"id": f"{base}-report", "gpus": 0, "after": [f"{base}-score"],
                        "cmd": f"{PY} scripts/report.py --config {cfg} --unit {unit}"})
    new = [j for j in new if j["id"] not in existing]
    with open(qpath, "a") as f:
        for j in new:
            f.write(json.dumps(j) + "\n")
    print(f"appended {len(new)} jobs to {qpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
