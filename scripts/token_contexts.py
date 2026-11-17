"""Recover the contexts behind the token footprints (scripts/token_footprint.py) as token ids. CPU.

The footprint pass kept activations but not the positions. Its selection depends only on the token stream (same
source, skip, batch size and quotas), so replaying the stream here finds exactly the same occurrences, in the
same order. scripts/token_durability.py checks this on the base model (activations must match the footprint).

  uv run python scripts/token_contexts.py --size 1.4b
  ... --limit 1            # same settings as the footprint dry run

Output: data/neurons[_dryrun]/<unit>/footprint/contexts.npz with, per panel token T<id>: seq [n, 256] int32
(the full sequence) and pos [n] int16 (where the token sits); reports/.../token_contexts.md shows examples.
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
from topodur.finetune import token_stream  # noqa: E402
from topodur.models import load_tokenizer  # noqa: E402

log = logging.getLogger("token_contexts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl")
    ap.add_argument("--skip-seqs", type=int, default=100_000)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--max-tokens", type=float, default=150e6)
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None)
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    fp = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / unit / "footprint"
    runs.setup_logging(fp / "contexts.log")
    out = fp / "contexts.npz"
    if runs.is_done(out):
        log.info("already done")
        return 0
    if a.limit:                                          # as in token_footprint.py
        a.max_tokens, a.n_ctx, a.skip_seqs = a.limit * 1e6, 16, 0
    meta = json.loads((fp / "footprint.json").read_text())
    ids = {int(t): v for t, v in meta["tokens"].items()}
    tok = load_tokenizer(a.size)
    seqs = {t: [] for t in ids}
    pos = {t: [] for t in ids}
    open_ = set(ids)
    stream = token_stream(tok, a.source, 256, skip_seqs=a.skip_seqs)
    seen, step = 0, 0
    while seen < a.max_tokens and open_:
        batch = np.array([next(stream) for _ in range(a.batch_seqs)], dtype=np.int64)
        hit = np.isin(batch, list(open_))
        hit[:, :4] = False
        for b, p in zip(*np.nonzero(hit)):                 # row-major, as torch.nonzero in the footprint pass
            t = int(batch[b, p])
            if len(pos[t]) < a.n_ctx:
                seqs[t].append(batch[b].astype(np.int32))
                pos[t].append(p)
                if len(pos[t]) == a.n_ctx:
                    open_.discard(t)
        seen += batch.size
        step += 1
        if step % 2000 == 0:
            log.info("%.1fM tokens, %d tokens still open", seen / 1e6, len(open_))
    bad = [t for t in ids if len(pos[t]) != ids[t]["n"]]
    if bad or seen != meta["tokens_streamed"]:
        log.error("replay does not match the footprint: streamed %d vs %d; counts differ for %d tokens",
                  seen, meta["tokens_streamed"], len(bad))
        return 1
    np.savez_compressed(out, **{f"T{t}": np.stack(seqs[t]) for t in ids if seqs[t]},
                        **{f"P{t}": np.array(pos[t], np.int16) for t in ids if pos[t]})
    ex = ["# Example contexts per panel token (first 3; the token is in [brackets])", ""]
    for t, v in ids.items():
        ex.append(f"## {v['word']} ({v['class']})")
        for s, p in list(zip(seqs[t], pos[t]))[:3]:
            left = tok.decode(s[max(0, p - 24):p]).replace("\n", " ")
            right = tok.decode(s[p + 1:p + 12]).replace("\n", " ")
            ex.append(f"- ...{left}[{tok.decode(s[p:p + 1])}]{right}...")
        ex.append("")
    rep = runs.REPO_ROOT / "reports" / ("neurons_dryrun" if a.limit else "neurons") / unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.atomic_write_text(rep / "token_contexts.md", "\n".join(ex))
    runs.mark_done(out, {"tokens": seen})
    log.info("done: %d tokens replayed, matches the footprint", seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
