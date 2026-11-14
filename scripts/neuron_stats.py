"""Neuron-space statistics for a group of layers (src/topodur/neurons.py). One GPU.

  uv run python scripts/neuron_stats.py --size 1.4b --layers 0 1
  ... --limit 0.2          # test pass: 0.2M tokens, writes data/neurons_dryrun/

Output: data/neurons[_dryrun]/<size>_s0_step<step>/L<ll>.npz (+ .done), calib.json.
Two layers per job fit a 10.5 GB card for 1.4B (per-token sums: vocab x 8192 fp32 = 1.65 GB per layer).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.models import load_model  # noqa: E402
from topodur.neurons import calibration_tokens, neuron_stats  # noqa: E402

log = logging.getLogger("neuron_stats")
CALIB_FAMILIES = ["months", "days", "numbers",  # clock_hours = same tokens as numbers
                  "ordinals", "zodiac", "compass", "hues", "sizes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl")
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="test pass: tokens in millions")
    a = ap.parse_args()
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / f"{a.size}_s0_step{a.step}"
    root.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(root / f"stats_L{'_'.join(map(str, a.layers))}.log")
    todo = [l for l in a.layers if not runs.is_done(root / f"L{l:02d}.npz")]
    if not todo:
        log.info("all layers done")
        return 0
    lm = load_model(a.size, a.step, dtype=a.dtype)
    lm.model.gpt_neox.layers = lm.model.gpt_neox.layers[: max(todo) + 1]   # later blocks never run
    cal = calibration_tokens(lm.tokenizer, CALIB_FAMILIES)
    calib_ids = sorted({t for v in cal.values() for _, t in v})
    runs.atomic_write_text(root / "calib.json", json.dumps({"ids": calib_ids, "families": cal}))
    if not (root / "vocab.json").exists():
        V = lm.model.gpt_neox.embed_in.weight.shape[0]
        runs.atomic_write_text(root / "vocab.json", json.dumps([lm.tokenizer.decode([i]) for i in range(V)]))
    n_tok = int((a.limit or a.tokens / 1e6) * 1e6)
    res = neuron_stats(lm, todo, a.source, n_tok, calib_ids, batch_seqs=a.batch_seqs)
    for l, d in res.items():
        out = root / f"L{l:02d}.npz"
        with runs.atomic_path(out) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, **d)
        runs.mark_done(out, {"tokens": int(d["n_tokens"])})
        log.info("wrote %s (median r2_tok %.3f)", out, float(np.median(d["r2_tok"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
