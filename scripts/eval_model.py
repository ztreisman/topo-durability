"""Held-out perplexity + behavioral offset queries for one (model, step, seed, variant). One GPU.

  uv run python scripts/eval_model.py --config configs/phase3_finetune.yaml --size 1.4b --step 143000 \
      --variant lora:data/finetune/1.4b_code/adapters/5M
  ... --limit 20 --dry-run

Writes eval.json (perplexities) and behavior.parquet (one row per family) into the unit's results dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.behavior import evaluate  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.finetune import perplexity  # noqa: E402
from topodur.models import load_model, unit_id  # noqa: E402

log = logging.getLogger("eval_model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=None, help="perplexity sequences (test pass)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run or a.limit else "")
    uid = unit_id(a.size, a.step, a.seed, a.variant)
    out = runs.run_dir(exp, uid)
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "eval.log")
    target = out / "behavior.parquet"
    if runs.is_done(target):
        log.info("%s already evaluated", uid)
        return 0
    ev = cfg.get("eval", {})
    lm = load_model(a.size, a.step, dtype=a.dtype or cfg.get("dtype", "fp16"), seed=a.seed or None,
                    variant=a.variant)
    n_seqs = a.limit or ev.get("ppl_seqs", 200)
    ppl = {}
    for name, src in (ev.get("ppl_sources") or {"pile_test": "mirror:pile_test_5000.jsonl"}).items():
        ppl[name] = perplexity(lm, src, n_seqs=n_seqs, seq_len=ev.get("ppl_seq_len", 512))
        log.info("perplexity %s: %.3f", name, ppl[name])
    runs.atomic_write_text(out / "eval.json", json.dumps({"unit": uid, "variant": a.variant, "ppl": ppl}, indent=1))
    rows = []
    for f in cfg["families"]:
        fam = load_family(f)
        if fam.behavior:
            r = evaluate(lm, fam, fam.behavior)
            log.info("%s: acc %.2f (wrap %.2f, nowrap %.2f)", f, r["acc"], r["acc_wrap"], r["acc_nowrap"])
            rows.append(r)
    df = pd.DataFrame(rows).assign(unit=uid, variant=a.variant or "base", **{f"ppl_{k}": v for k, v in ppl.items()})
    with runs.atomic_path(target) as tmp:
        df.to_parquet(tmp, index=False)
    if not a.dry_run:
        runs.mark_done(target, {"families": len(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
