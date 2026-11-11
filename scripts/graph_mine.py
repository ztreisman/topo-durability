"""Phase 5 pipeline for one trained SAE: firing rates -> node set -> window co-activation -> NPMI and
decoder-cosine graphs -> H1 cycle mining -> rewiring nulls -> top cycles with top-activating contexts.
One GPU for the streaming stages; resumable stage by stage.

  uv run python scripts/graph_mine.py --sae data/sae/1.4b_L14_32768 --layer 14
  ... --limit 2 --dry-run           # 2M tokens of co-activation, 2 nulls

Outputs in <sae dir>/graph/: rates.npy, coact.npz, graph_{npmi,decoder}.npz, cycles_{graph}.parquet,
nulls_{graph}.npz; reports/cycles/<sae name>/<graph>_cycle<k>.md for the top cycles.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.graph.coact import CoactStats, coactivation, decoder_cosine, firing_rates, knn_graph, npmi_matrix  # noqa: E402
from topodur.graph.cycles import is_chordless, mine_cycles, null_maxima  # noqa: E402
from topodur.graph.sae import ActStream, TopKSAE  # noqa: E402
from topodur.models import load_model  # noqa: E402

log = logging.getLogger("graph_mine")


@torch.no_grad()
def top_contexts(lm, sae, layer, source, latents, n_tokens, k=8, seq_len=128, batch_seqs=32, width=24, skip_seqs=0):
    """Top-k activating token contexts per latent (decoded text up to and including the token)."""
    from topodur.finetune import token_stream

    want = torch.as_tensor(sorted(set(latents)), device=sae.W_dec.device)
    best: dict[int, list] = {int(l): [] for l in want.tolist()}
    cap = {}
    hk = lm.model.gpt_neox.layers[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    toks = token_stream(lm.tokenizer, source, seq_len, skip_seqs=skip_seqs)
    seen = 0
    try:
        while seen < n_tokens:
            seqs = [next(toks) for _ in range(batch_seqs)]
            lm.model.gpt_neox(input_ids=torch.tensor(seqs, device=lm.device), use_cache=False)
            h = cap["h"].float()
            pre = sae.pre(h.reshape(-1, h.shape[-1]))[:, want]          # pre-activations of wanted latents
            vals, pos = pre.topk(min(k, pre.shape[0]), dim=0)
            for j, l in enumerate(want.tolist()):
                for v, p in zip(vals[:, j].tolist(), pos[:, j].tolist()):
                    s, t = divmod(p, seq_len)
                    best[l].append((v, seqs[s][max(0, t - width):t + 1]))
                best[l] = sorted(best[l], key=lambda x: -x[0])[:k]
            seen += batch_seqs * seq_len
    finally:
        hk.remove()
    return {l: [(v, lm.tokenizer.decode(ids)) for v, ids in lst] for l, lst in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True, help="SAE run dir (contains final/)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl",
                    help="different documents from the SAE training data (pile_mine_1000000)")
    ap.add_argument("--rate-tokens", type=float, default=5e6)
    ap.add_argument("--coact-tokens", type=float, default=50e6)
    ap.add_argument("--min-rate", type=float, default=1e-5)
    ap.add_argument("--max-rate", type=float, default=1e-2)
    ap.add_argument("--k-nn", type=int, default=15)
    ap.add_argument("--min-npmi", type=float, default=0.1)
    ap.add_argument("--min-cos", type=float, default=0.2)
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--top-cycles", type=int, default=50)
    ap.add_argument("--limit", type=float, default=None, help="test pass: coact tokens in millions")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sae_dir = Path(a.sae)
    out = sae_dir / ("graph_dryrun" if a.dry_run or a.limit else "graph")
    out.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out / "graph.log")
    if a.limit:
        a.coact_tokens, a.rate_tokens, a.n_null, a.top_cycles = a.limit * 1e6, 1e6, 2, 5

    lm = load_model(a.size, a.step, dtype="fp16")
    lm.model.gpt_neox.layers = lm.model.gpt_neox.layers[: a.layer + 1]
    sae = TopKSAE.load(sae_dir / "final", device=lm.device)

    if not (out / "rates.npy").exists():
        rates = firing_rates(sae, ActStream(lm, a.layer, a.source), int(a.rate_tokens))
        np.save(out / "rates.npy", rates)
    rates = np.load(out / "rates.npy")
    nodes = np.where((rates >= a.min_rate) & (rates <= a.max_rate))[0]
    log.info("%d / %d latents in the firing band [%g, %g]", len(nodes), len(rates), a.min_rate, a.max_rate)

    if not runs.is_done(out / "coact.npz"):
        st = coactivation(sae, lm, a.layer, a.source, int(a.coact_tokens), nodes)
        st.save(out / "coact.npz")
        runs.mark_done(out / "coact.npz", {"tokens": st.n_tokens})
    st = CoactStats.load(out / "coact.npz")

    graphs = {"npmi": (npmi_matrix(st), a.min_npmi), "decoder": (decoder_cosine(sae, st.nodes), a.min_cos)}
    summary = {}
    for g, (S, thr) in graphs.items():
        D = knn_graph(S, a.k_nn, thr)
        sp.save_npz(out / f"graph_{g}.npz", D)
        cyc = mine_cycles(D, top=500)
        nulls = {kind: null_maxima(D, n_null=a.n_null, preserve_triangles=(kind == "triangle"))
                 for kind in ("degree", "triangle")}
        np.savez(out / f"nulls_{g}.npz", **nulls)
        thr_life = max(np.quantile(v[:, 0], 0.95) for v in nulls.values())
        thr_score = max(np.quantile(v[:, 1], 0.95) for v in nulls.values())
        rows = []
        for i, c in enumerate(cyc):
            lat = st.nodes[c.vertices]
            rows.append({"graph": g, "rank": i, "n_vertices": len(c.vertices), "birth": c.birth, "death": c.death,
                         "lifetime": c.lifetime, "score": c.score, "chordless": is_chordless(D, c.vertices, c.birth),
                         "beats_nulls": bool(c.lifetime > thr_life and c.score > thr_score),
                         "latents": lat.tolist(), "mean_rate": float(rates[lat].mean())})
        df = pd.DataFrame(rows, columns=["graph", "rank", "n_vertices", "birth", "death", "lifetime", "score",
                                         "chordless", "beats_nulls", "latents", "mean_rate"])
        df.to_parquet(out / f"cycles_{g}.parquet", index=False)
        summary[g] = {"nodes": int(D.shape[0]), "edges": int(D.nnz // 2), "h1_classes": len(cyc),
                      "beat_nulls": int(df.beats_nulls.sum()), "null_life_95": float(thr_life),
                      "null_score_95": float(thr_score)}
        log.info("%s graph: %s", g, summary[g])

        top = df[df.beats_nulls].head(a.top_cycles) if df.beats_nulls.any() else df.head(min(5, a.top_cycles))
        if top.empty:
            log.info("%s graph: no H1 classes, no cycle reports", g)
            continue
        lat_all = sorted({l for ls in top.latents for l in ls})
        ctx = top_contexts(lm, sae, a.layer, a.source, lat_all, n_tokens=int(2e6 if not a.limit else 2e5),
                           skip_seqs=0 if a.limit else int(a.coact_tokens // 128) + 1000)  # past the coact data
        rep = runs.REPO_ROOT / "reports" / "cycles" / (sae_dir.name + ("_dryrun" if a.limit else "")) / g
        rep.mkdir(parents=True, exist_ok=True)
        for r in top.itertuples():
            lines = [f"# {g} cycle rank {r.rank}: {r.n_vertices} latents, lifetime {r.lifetime:.3f}, "
                     f"score {r.score:.2f}, chordless {r.chordless}, beats nulls {r.beats_nulls}", "",
                     "Latents in cycle order. Label each with a short description; note which edge looks like "
                     "the closing (rarest) edge.", ""]
            for l in r.latents:
                lines.append(f"## latent {l} (rate {rates[l]:.2e})  label: ____")
                lines += [f"- `{t!r}` ({v:.1f})" for v, t in ctx.get(l, [])]
                lines.append("")
            (rep / f"cycle{r.rank:03d}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
