"""Cross-layer neuron correlations for a block of layer pairs (neurons.neuron_cross_stats). One GPU.
Runs after neuron_stats.py has finished every layer in the block: it replays the same token stream and
uses those means, sds and per-token means to split each cross-layer correlation into
total / tok (between-token, inherited) / ctx (within-token, emergent candidate).

  uv run python scripts/neuron_xstats.py --size 1.4b --pairs 0,1 0,5 3,20
  ... --limit 1            # must match the --limit of the neuron_stats.py dry run

Output per pair: data/neurons[_dryrun]/<unit>/X_L<ll>_L<mm>.npz (+ .done): sparse candidate edges (union
of each neuron's top-k partners in the other layer under any of the three graphs, all three values kept)
and correlation quantiles. 12 pairs per job fit a 10.5 GB card (fp32 accumulators, 256 MB per pair).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.models import load_model  # noqa: E402
from topodur.neurons import cross_split, neuron_cross_stats, sparse_topk  # noqa: E402

log = logging.getLogger("neuron_xstats")
QS = [0.5, 0.9, 0.99, 0.999]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--pairs", nargs="+", required=True, help="l,m with l < m (l,l = self-check against neuron_stats)")
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl")
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--k-store", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="test pass: tokens in millions")
    a = ap.parse_args()
    pairs = sorted({tuple(sorted(map(int, p.split(",")))) for p in a.pairs})
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / f"{a.size}_s0_step{a.step}"
    runs.setup_logging(root / f"xstats_{pairs[0][0]}-{pairs[0][1]}_n{len(pairs)}.log")
    xp = lambda l, m: root / f"X_L{l:02d}_L{m:02d}.npz"
    todo = [p for p in pairs if not runs.is_done(xp(*p))]
    if not todo:
        log.info("all pairs done")
        return 0
    layers = sorted({l for p in todo for l in p})
    st = {}
    for l in layers:
        if not runs.is_done(root / f"L{l:02d}.npz"):
            log.error("layer %d has no neuron_stats output yet", l)
            return 1
        with np.load(root / f"L{l:02d}.npz") as z:
            st[l] = {k: z[k] for k in ("mean", "sd", "sd_tok", "sd_ctx", "n_tokens")}
    n_tok = int((a.limit or a.tokens / 1e6) * 1e6)
    lm = load_model(a.size, a.step, dtype=a.dtype)
    lm.model.gpt_neox.layers = lm.model.gpt_neox.layers[: max(layers) + 1]
    Ezz, zbar, seen = neuron_cross_stats(lm, todo, a.source, n_tok, {l: st[l]["mean"] for l in layers},
                                         {l: st[l]["sd"] for l in layers}, batch_seqs=a.batch_seqs)
    if any(int(st[l]["n_tokens"]) != seen for l in layers):
        log.error("token count %d differs from the within-layer pass %s: streams do not match", seen,
                  {l: int(st[l]['n_tokens']) for l in layers})
        return 1
    del lm
    torch.cuda.empty_cache()
    tc = np.load(root / "tokcounts.npz")["cnt"]
    tm = {l: np.load(root / f"L{l:02d}_tokmeans.npy", mmap_mode="r") for l in layers}
    for (l, m) in todo:
        C = cross_split(Ezz.pop((l, m)), zbar[l], zbar[m], tm[l], tm[m], tc, st[l], st[m], seen)
        sp_ = sparse_topk(C, a.k_store)
        q = {f"q_{g}": torch.quantile(C[g].abs().ravel()[torch.randperm(C[g].numel(), device=C[g].device)[:2_000_000]],
                                     torch.tensor(QS, device=C[g].device)).cpu().numpy() for g in C}
        with runs.atomic_path(xp(l, m)) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, **sp_, **q, qs=np.array(QS), n_tokens=seen)
        if l == m:   # self-check: a layer against itself must reproduce neuron_stats' within-layer correlations
            with np.load(root / f"L{l:02d}.npz") as z:
                ii, jj = sp_["i"].astype(np.int64), sp_["j"].astype(np.int64)
                off = ii != jj
                err = {g: float(np.abs(z[f"corr_{g}"][ii[off], jj[off]].astype(np.float32) - sp_[g][off].astype(np.float32)).max())
                       for g in ("total", "tok", "ctx")}
            log.info("self-check L%d: max |cross - within| %s", l, err)
            if max(err.values()) > 0.02:
                log.error("self-check failed: the cross-layer split does not reproduce the within-layer one")
                return 1
        runs.mark_done(xp(l, m), {"edges": int(len(sp_["i"]))})
        log.info("L%d-L%d: %d candidate edges; |ctx| q99 %.3f, |tok| q99 %.3f", l, m, len(sp_["i"]),
                 q["q_ctx"][2], q["q_tok"][2])
        del C
    return 0


if __name__ == "__main__":
    sys.exit(main())
