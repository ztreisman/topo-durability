"""Per-token durability pass: run one model variant over each panel token's footprint contexts. One GPU.

  uv run python scripts/token_durability.py --size 1.4b                                  # base (also the check)
  uv run python scripts/token_durability.py --size 1.4b --variant lora:data/finetune/1.4b_code/adapters/20M
  ... --variant nf4 | prune:0.25 | noise:0.03          (spec strings: topodur.models.parse_variant)
  ... --limit 0.3 --max-tokens-panel 6                 # dry run on the footprint dry-run data

For each context (scripts/token_contexts.py) records:
  logp   log-probability the model gives the panel token at its position (behavioral use of the concept)
  acts   post-activation MLP vector of every layer at that position (--save-acts; for representational retention)
Base model: activations are compared with the footprint pass; the job fails unless they agree (median per-layer
correlation >= 0.99), which also confirms the context replay.
Output: data/neurons[_dryrun]/<unit>/durab/<variant tag or 'base'>/logp.npz (+ A_T<id>.npy fp16 [n, L, N])
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
from topodur.models import load_model, variant_tag  # noqa: E402

log = logging.getLogger("token_durability")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--n-ctx", type=int, default=224, help="first n contexts per token (the senses analysis uses 224)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-acts", action="store_true")
    ap.add_argument("--limit", type=float, default=None, help="footprint dry-run data")
    ap.add_argument("--max-tokens-panel", type=int, default=None, help="only the first N panel tokens (dry run)")
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / unit
    fp = root / "footprint"
    tag = variant_tag(a.variant) or "base"
    out = root / "durab" / (tag + ("_dry" if a.max_tokens_panel else ""))      # a partial pass never counts as done
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "durab.log")
    if runs.is_done(out / "logp.npz"):
        log.info("already done")
        return 0
    C = np.load(fp / "contexts.npz")
    meta = json.loads((fp / "footprint.json").read_text())
    toks = [int(t) for t in meta["tokens"] if f"T{t}" in C.files]
    if a.max_tokens_panel:
        toks = toks[:a.max_tokens_panel]
    lm = load_model(a.size, a.step, dtype=a.dtype, variant=a.variant)
    net = lm.model.gpt_neox
    L = lm.model.config.num_hidden_layers
    caps = {}
    hooks = [net.layers[l].mlp.act.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o)) for l in range(L)]
    logp, corr = {}, []
    try:
        for k, t in enumerate(toks):
            S, P = C[f"T{t}"][:a.n_ctx], C[f"P{t}"][:a.n_ctx].astype(np.int64)
            lp, acts = [], []
            for s in range(0, len(S), a.batch):
                x = torch.from_numpy(S[s:s + a.batch].astype(np.int64)).to(lm.device)
                p = torch.from_numpy(P[s:s + a.batch]).to(lm.device)
                r = torch.arange(len(x), device=lm.device)
                h = net(input_ids=x, use_cache=False).last_hidden_state[r, p - 1]       # predicts position p
                lz = torch.log_softmax(lm.model.get_output_embeddings()(h).float(), -1)
                lp.append(lz[r, x[r, p]].cpu().numpy())
                if a.save_acts or tag == "base":
                    acts.append(torch.stack([caps[l][r, p] for l in range(L)], 1).half().cpu().numpy())
            logp[f"T{t}"] = np.concatenate(lp)
            if acts:
                A = np.concatenate(acts)
                if a.save_acts:
                    np.save(out / f"A_T{t}.npy", A)
                if tag == "base":                                                    # replay + numerics check
                    F = np.load(fp / f"T{t}.npy", mmap_mode="r")[:len(A)].astype(np.float32)
                    A32 = A.astype(np.float32)
                    for l in range(1, L):
                        u, v = F[:, l].ravel(), A32[:, l].ravel()
                        corr.append(float(np.corrcoef(u, v)[0, 1]))
            if (k + 1) % 20 == 0:
                log.info("%d / %d tokens; median log p so far %.2f", k + 1, len(toks),
                         float(np.median(np.concatenate(list(logp.values())))))
    finally:
        for hk in hooks:
            hk.remove()
    if tag == "base":
        med = float(np.median(corr))
        log.info("footprint agreement: median per-layer correlation %.4f (min %.4f)", med, float(np.min(corr)))
        if not med >= 0.99:
            log.error("activations do not match the footprint: contexts are not the same")
            return 1
    np.savez_compressed(out / "logp.npz", **logp)
    runs.atomic_write_text(out / "durab.json", json.dumps({"variant": a.variant, "tag": tag, "tokens": len(toks),
                                                           "n_ctx": a.n_ctx, "dtype": str(lm.dtype)}, indent=1))
    runs.mark_done(out / "logp.npz", {"tokens": len(toks)})
    log.info("done: %d tokens, median log p %.3f", len(toks), float(np.median(np.concatenate(list(logp.values())))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
