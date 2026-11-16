"""Token footprints: MLP activations of every layer at many occurrences of each panel token (version B of the
fork/rejoin study, scripts/token_reeb.py --mode ctx). One GPU.

  uv run python scripts/token_footprint.py --size 1.4b
  ... --limit 1            # 1M tokens, 16 contexts per token, writes data/neurons_dryrun/

Streams Pile text (after the neuron_stats tokens, so contexts are fresh), and at each position (>= 4) holding a
panel token whose quota is not yet full, records the post-activation MLP vector of every layer.
Output: data/neurons[_dryrun]/<unit>/footprint/T<token id>.npy [n_ctx, n_layers, N] fp16 + footprint.json.
Size at 1.4B: 256 contexts x 24 layers x 8192 x 2 B = 100 MB per token (~25 GB for the panel; not published).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.finetune import token_stream  # noqa: E402
from topodur.models import load_model  # noqa: E402
from topodur.reeb import panel  # noqa: E402

log = logging.getLogger("token_footprint")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--panel", default=str(runs.REPO_ROOT / "configs" / "token_panel.yaml"))
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl")
    ap.add_argument("--skip-seqs", type=int, default=100_000, help="start after the neuron_stats tokens (20M = 78k seqs)")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--max-tokens", type=float, default=150e6)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="test pass: tokens in millions")
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / unit
    out = root / "footprint"
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "footprint.log")
    if runs.is_done(out / "footprint.json"):
        log.info("already done")
        return 0
    if a.limit:
        a.max_tokens, a.n_ctx, a.skip_seqs = a.limit * 1e6, 16, 0
    lm = load_model(a.size, a.step, dtype=a.dtype)
    tok = lm.tokenizer
    calib = json.loads((root / "calib.json").read_text()) if (root / "calib.json").exists() else None
    pan = panel(a.panel, calib)
    ids = {}
    for cls, words in pan.items():
        for w in words:
            t = tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.setdefault(t[0], (cls, w))
    L = lm.model.config.num_hidden_layers
    N = lm.model.config.intermediate_size
    dev = lm.device
    caps = {}
    hooks = [lm.model.gpt_neox.layers[l].mlp.act.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o))
             for l in range(L)]
    buf = {t: [] for t in ids}
    have = {t: 0 for t in ids}
    want = torch.zeros(lm.model.gpt_neox.embed_in.weight.shape[0], dtype=torch.bool, device=dev)
    want[list(ids)] = True
    stream = token_stream(tok, a.source, 256, skip_seqs=a.skip_seqs)
    seen, step = 0, 0
    try:
        while seen < a.max_tokens and want.any():
            seqs = torch.tensor([next(stream) for _ in range(a.batch_seqs)], device=dev)
            hit = want[seqs]
            hit[:, :4] = False
            if hit.any():
                lm.model.gpt_neox(input_ids=seqs, use_cache=False)
                b, p = torch.nonzero(hit, as_tuple=True)
                tt = seqs[b, p].tolist()
                V = torch.stack([caps[l][b, p] for l in range(L)], 1).half().cpu().numpy()   # [hits, L, N]
                for k, t in enumerate(tt):
                    if have[t] < a.n_ctx:
                        buf[t].append(V[k])
                        have[t] += 1
                        if have[t] == a.n_ctx:
                            want[t] = False
            seen += seqs.numel()
            step += 1
            if step % 500 == 0:
                full = sum(v >= a.n_ctx for v in have.values())
                log.info("%.1fM tokens: %d / %d tokens full, median contexts %d", seen / 1e6, full, len(ids),
                         int(np.median(list(have.values()))))
    finally:
        for h in hooks:
            h.remove()
    for t, rows in buf.items():
        if rows:
            np.save(out / f"T{t}.npy", np.stack(rows))
    meta = {"layers": list(range(L)), "tokens_streamed": seen, "n_ctx": a.n_ctx,
            "tokens": {str(t): {"class": ids[t][0], "word": ids[t][1], "n": have[t]} for t in ids}}
    runs.atomic_write_text(out / "footprint.json", json.dumps(meta, indent=1))
    runs.mark_done(out / "footprint.json", {"tokens": seen})
    log.info("done: %d tokens streamed; contexts per token min %d median %d", seen, min(have.values()),
             int(np.median(list(have.values()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
