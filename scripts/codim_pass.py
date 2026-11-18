"""First-order effect of each perturbation on each concept (test 1 of the codimension study, 2026-11-18). One GPU.

  uv run python scripts/codim_pass.py --size 1.4b
  ... --limit 3 --max-concepts 6          # dry run on the footprint dry-run data

A rank-r update moves activations along few directions; a concept is hit to the extent its activations overlap the
update. For every perturbation v this computes dW_l = W_v - W_base for each block's MLP output matrix
(dense_4h_to_h; LoRA merged, pruning and noise as in topodur.models), then runs the BASE model over each concept's
contexts and records, at the position q = p - n_tok that predicts the concept's first token and at the concept's
last token p:
  rel[v, l]    ||dW_l a_l|| / ||W_l a_l||    (a_l = MLP post-activation): how much of the layer's MLP output the
               update changes for this context
  dlogp[v]     direct-path first-order change in log p of the concept's first token at q: the summed MLP-output
               changes dx = sum_l dW_l a_l, through the final LayerNorm (frozen scale) and the unembedding,
               dlogp ~ dlogit_t - sum_j p_j dlogit_j
Attention and MLP-input changes are not included; this is the part of the damage the MLP-output update explains
directly. Output: data/neurons[_dryrun]/<unit>/codim/pass.npz (per concept C<i>: rel [n, V, 2, L], dlogp [n, V]),
pass.json (variant order).
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

log = logging.getLogger("codim_pass")
VARIANTS = ["lora:data/finetune/1.4b_code/adapters/20M", "lora:data/finetune/1.4b_wiki_de/adapters/20M",
            "prune:0.25", "prune:0.5"]


@torch.no_grad()
def out_weights(size, step, variant):
    lm = load_model(size, step, dtype="fp32", device="cpu", variant=variant, check=False)
    W = [blk.mlp.dense_4h_to_h.weight.detach().clone() for blk in lm.model.gpt_neox.layers]
    del lm
    return W


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--fp-dir", default="cfoot")
    ap.add_argument("--n-ctx", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--max-concepts", type=int, default=None)
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / unit
    fp = root / a.fp_dir
    out = root / "codim"
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "pass.log")
    if runs.is_done(out / "pass.npz") and not a.max_concepts:
        log.info("already done")
        return 0
    base_W = out_weights(a.size, a.step, None)
    dW = []
    for v in a.variants:
        Wv = out_weights(a.size, a.step, v)
        dW.append([(w1 - w0).half() for w0, w1 in zip(base_W, Wv)])
        log.info("%s: |dW|/|W| per layer (median) %.4f", v,
                 float(np.median([float(d.float().norm() / w.norm()) for d, w in zip(dW[-1], base_W)])))
        del Wv
    lm = load_model(a.size, a.step, dtype="fp16")
    dev = lm.device
    net = lm.model.gpt_neox
    L = len(net.layers)
    V = len(a.variants)
    wdt = next(lm.model.parameters()).dtype
    Wg = [w.to(dev, wdt) for w in base_W]
    dWg = [[d.to(dev, wdt) for d in dv] for dv in dW]
    del base_W, dW
    g = net.final_layer_norm.weight.float()
    U = lm.model.get_output_embeddings().weight                         # [vocab, d]
    caps, pre = {}, {}
    hooks = [net.layers[l].mlp.act.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o)) for l in range(L)]
    hooks.append(net.final_layer_norm.register_forward_hook(lambda m, i, o: pre.__setitem__("x", i[0])))
    C = np.load(fp / "contexts.npz")
    meta = json.loads((fp / "footprint.json").read_text())
    toks = [int(t) for t in meta["tokens"] if f"T{t}" in C.files]
    if a.max_concepts:
        toks = toks[:a.max_concepts]
    res = {}
    try:
        for k, t in enumerate(toks):
            S, P = C[f"T{t}"][:a.n_ctx], C[f"P{t}"][:a.n_ctx].astype(np.int64)
            n_tok = int(meta["tokens"][str(t)].get("n_tok", 1))
            rel_all, dl_all = [], []
            for s in range(0, len(S), a.batch):
                x = torch.from_numpy(S[s:s + a.batch].astype(np.int64)).to(dev)
                p = torch.from_numpy(P[s:s + a.batch]).to(dev)
                r = torch.arange(len(x), device=dev)
                q = p - n_tok
                hs = net(input_ids=x, use_cache=False).last_hidden_state
                tgt = x[r, q + 1]                                           # the concept's first token
                logits = lm.model.get_output_embeddings()(hs[r, q]).float()
                prob = torch.softmax(logits, -1)
                xq = pre["x"][r, q].float()
                sig = xq.var(-1, unbiased=False).add(1e-5).sqrt()          # final LayerNorm scale (frozen)
                rel = torch.zeros(len(x), V, 2, L, device=dev)
                dx = torch.zeros(len(x), V, xq.shape[-1], device=dev)
                for l in range(L):
                    for j, pos in enumerate((q, p)):
                        act = caps[l][r, pos].to(wdt)                       # [b, 8192]
                        base_out = (act @ Wg[l].T).float().norm(dim=-1).clamp_min(1e-6)
                        for v in range(V):
                            d = (act @ dWg[v][l].T).float()
                            rel[:, v, j, l] = d.norm(dim=-1) / base_out
                            if j == 0:
                                dx[:, v] += d
                dxc = dx - dx.mean(-1, keepdim=True)
                dy = (g * dxc) / sig[:, None, None]                         # [b, V, d]
                dlog = (dy.to(U.dtype) @ U.T).float()                          # [b, V, vocab]
                dl = dlog[r, :, tgt] - (prob[:, None, :] * dlog).sum(-1)   # [b, V]
                rel_all.append(rel.cpu().numpy())
                dl_all.append(dl.cpu().numpy())
            res[f"R{t}"] = np.concatenate(rel_all).astype(np.float32)
            res[f"D{t}"] = np.concatenate(dl_all).astype(np.float32)
            if (k + 1) % 50 == 0:
                log.info("%d / %d concepts", k + 1, len(toks))
    finally:
        for h in hooks:
            h.remove()
    np.savez_compressed(out / "pass.npz", **res)
    runs.atomic_write_text(out / "pass.json", json.dumps({"variants": a.variants, "tags": [variant_tag(v) for v in a.variants],
                                                          "concepts": len(toks)}, indent=1))
    if not a.max_concepts:
        runs.mark_done(out / "pass.npz", {"concepts": len(toks)})
    log.info("done: %d concepts", len(toks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
