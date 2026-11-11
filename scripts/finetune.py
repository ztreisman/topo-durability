"""LoRA fine-tune Pythia on an unrelated corpus; resumable. One GPU.

  uv run python scripts/finetune.py --name code --corpus mirror:corpora/code_py_30000.jsonl
  uv run python scripts/finetune.py --name wiki_de --corpus mirror:corpora/wiki_de_40000.jsonl
  ... --limit 20 --dry-run      # 20 steps into data/finetune_dryrun/<name>

Adapters land in data/finetune/<size>_<name>/adapters/<N>M; extract them with
  scripts/extract.py ... --variant lora:data/finetune/<size>_<name>/adapters/<N>M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topodur import runs  # noqa: E402
from topodur.finetune import FTConfig, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="train only this many steps")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = runs.DATA_DIR / ("finetune_dryrun" if a.dry_run or a.limit else "finetune")
    out = root / f"{a.size}_{a.name}"
    runs.setup_logging(out / "finetune.log")
    cfg = FTConfig(size=a.size, corpus=a.corpus, total_tokens=int(a.tokens), batch_size=a.batch_size,
                   grad_accum=a.grad_accum, seq_len=a.seq_len, lr=a.lr, r=a.r, seed=a.seed)
    if a.limit:
        cfg.milestones = [0, a.limit * cfg.seq_len * cfg.batch_size * cfg.grad_accum]
        cfg.ckpt_every = max(1, a.limit // 2)
    if (out / ".complete").exists():
        print(f"{out} already complete")
        return 0
    train(cfg, out, limit_steps=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
