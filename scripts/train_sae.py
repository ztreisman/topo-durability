"""Train a TopK SAE on the residual stream after one block (Phase 5). One GPU; resumable.

  uv run python scripts/train_sae.py --size 1.4b --layer 14 --latents 32768 --tokens 200e6
  ... --limit 300 --dry-run        # 300 steps into data/sae_dryrun/ (timing estimate)

Output: data/sae/<size>_L<layer>_<latents>/final/{sae.pt,cfg.json}, train_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topodur import runs  # noqa: E402
from topodur.graph.sae import SAEConfig, train_sae  # noqa: E402
from topodur.models import load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latents", type=int, default=32768)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tokens", type=float, default=200e6)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--source", default="mirror:pile_mine_1000000.jsonl")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="train only this many steps")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = runs.DATA_DIR / ("sae_dryrun" if a.dry_run or a.limit else "sae")
    out = root / f"{a.size}_L{a.layer}_{a.latents}"
    runs.setup_logging(out / "train.log")
    if (out / "final" / "sae.pt").exists() and not (a.dry_run or a.limit):
        print(f"{out} done")
        return 0
    lm = load_model(a.size, a.step, dtype=a.dtype, device=a.device)
    cfg = SAEConfig(d_in=lm.d_model, n_latents=a.latents, k=a.k)
    t0 = time.time()
    train_sae(lm, a.layer, a.source, out, cfg, int(a.tokens), batch_tokens=a.batch_tokens, limit_steps=a.limit)
    el = time.time() - t0
    if a.limit:
        est_h = el / a.limit * (a.tokens / a.batch_tokens) / 3600
        msg = {"steps": a.limit, "seconds": el, "estimated_hours_full_run": est_h}
        (out / "timing.json").write_text(json.dumps(msg))
        print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
