"""Phase 5 durability of matched cycles / induced paths / cliques in the latent graph. One GPU.

  uv run python scripts/struct_durability.py --sae data/sae/1.4b_L14_32768 --layer 14 --graph npmi \
      --variants lora:data/finetune/1.4b_code/adapters/20M lora:data/finetune/1.4b_wiki_de/adapters/20M \
                 prune:0.25 nf4 noise:0.03
  ... --limit 1 --dry-run

For each null-beating chordless cycle: up to --per-cycle induced paths of the same length with matched
firing rate, and cliques from the same thresholded graph. Per structure: activation correlation
(base vs each variant, original SAE), and single-latent ablation effects on the base model.
Output: <sae>/graph/structures.parquet and reports/phase5/<sae name>_<graph>.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.finetune import token_stream  # noqa: E402
from topodur.graph.coact import CoactStats  # noqa: E402
from topodur.graph.sae import TopKSAE  # noqa: E402
from topodur.graph.structures import (ablation_effects, act_corr, cliques, latent_acts, matched_paths,  # noqa: E402
                                      thresholded)
from topodur.models import load_model, variant_tag  # noqa: E402

log = logging.getLogger("struct_durability")


def boot_diff(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--graph", default="npmi")
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--source", default="mirror:pile_sample_200000.jsonl")
    ap.add_argument("--tokens", type=float, default=5e5)
    ap.add_argument("--ablate-tokens", type=float, default=3e4)
    ap.add_argument("--per-cycle", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="test pass: number of cycles")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sae_dir = Path(a.sae)
    gdir = sae_dir / ("graph_dryrun" if a.dry_run or a.limit else "graph")
    runs.setup_logging(gdir / "structures.log")
    if a.limit:
        a.max_cycles, a.tokens, a.ablate_tokens = a.limit, 2e4, 4e3
    rng = np.random.default_rng(0)
    st = CoactStats.load(gdir / "coact.npz")
    rates = np.load(gdir / "rates.npy")
    rn = rates[st.nodes]
    D = sp.load_npz(gdir / f"graph_{a.graph}.npz").tocsr()
    cyc = pd.read_parquet(gdir / f"cycles_{a.graph}.parquet")
    cand = cyc[cyc.chordless & cyc.beats_nulls] if (cyc.chordless & cyc.beats_nulls).any() else cyc[cyc.chordless]
    cand = cand[cand.n_vertices >= 4].head(a.max_cycles)
    if cand.empty:
        log.warning("no chordless cycles with >= 4 vertices in %s; nothing to measure", a.graph)
        return 0
    node_of = {int(l): i for i, l in enumerate(st.nodes)}
    structs = []
    for r in cand.itertuples():
        verts = [node_of[l] for l in r.latents]
        k = len(verts)
        A = thresholded(D, r.birth)
        structs.append({"kind": "cycle", "cycle_rank": r.rank, "nodes": verts})
        target = 10 ** np.log10(rn[verts]).mean()           # geometric-mean firing rate of the cycle
        for p in matched_paths(A, k, target, rn, a.per_cycle, rng):
            structs.append({"kind": "path", "cycle_rank": r.rank, "nodes": p})
        for c in cliques(A, k, a.per_cycle, rng):
            structs.append({"kind": "clique", "cycle_rank": r.rank, "nodes": c})
    for s in structs:
        s["latents"] = [int(st.nodes[i]) for i in s["nodes"]]
        s["size"] = len(s["nodes"])
        s["mean_log_rate"] = float(np.log10(rn[s["nodes"]]).mean())
    log.info("%d structures: %s", len(structs), pd.Series([s["kind"] for s in structs]).value_counts().to_dict())
    lat_all = np.array(sorted({l for s in structs for l in s["latents"]}))
    col = {l: i for i, l in enumerate(lat_all)}

    seq_len, bs = 128, 32
    lm = load_model(a.size, a.step, dtype="fp16")
    sae = TopKSAE.load(sae_dir / "final", device=lm.device)
    stream = token_stream(lm.tokenizer, a.source, seq_len, skip_seqs=600_000)  # past the coact/context data
    n_b = max(1, int(a.tokens // (seq_len * bs)))
    batches = [torch.tensor([next(stream) for _ in range(bs)]) for _ in range(n_b)]

    # ablations on the full base model
    n_ab = max(1, int(a.ablate_tokens // (seq_len * bs)))
    for s in structs:
        effs = [ablation_effects(lm, sae, a.layer, b, s["latents"]) for b in batches[:n_ab]]
        s["abl_dloss"] = float(np.mean([e["dloss_mean"] for e in effs]))
        s["abl_others_rel"] = float(np.nanmean([e["others_rel_change"] for e in effs]))
    base = latent_acts(lm, sae, a.layer, batches, lat_all).astype(np.float16)
    del lm
    torch.cuda.empty_cache()

    for v in a.variants:
        lmv = load_model(a.size, a.step, dtype="fp16", variant=v)
        mod = latent_acts(lmv, sae, a.layer, batches, lat_all).astype(np.float16)
        rho = act_corr(base.astype(np.float32), mod.astype(np.float32))
        tag = variant_tag(v)
        for s in structs:
            s[f"act_corr_{tag}"] = float(np.nanmean(rho[[col[l] for l in s["latents"]]]))
        del lmv
        torch.cuda.empty_cache()

    df = pd.DataFrame(structs)
    df.to_parquet(gdir / "structures.parquet", index=False)
    lines = [f"# Phase 5 structures: {sae_dir.name} / {a.graph}", "",
             f"{len(cand)} chordless cycles; counts: {df.kind.value_counts().to_dict()}; "
             f"sizes by kind: {df.groupby('kind')['size'].median().to_dict()}", "",
             "Higher act_corr = more durable; lower abl_others_rel = more tolerant of a single ablation.", "",
             "| metric | cycle | path | clique | cycle - path [95% CI] | cycle - clique [95% CI] |",
             "|---|---|---|---|---|---|"]
    for m in [c for c in df.columns if c.startswith("act_corr_")] + ["abl_others_rel", "abl_dloss"]:
        g = {k: df[df.kind == k][m] for k in ("cycle", "path", "clique")}
        d1, d2 = boot_diff(g["cycle"], g["path"]), boot_diff(g["cycle"], g["clique"])
        lines.append(f"| {m} | {g['cycle'].mean():.3g} | {g['path'].mean():.3g} | {g['clique'].mean():.3g} | "
                     f"{d1[0]:+.3g} [{d1[1]:+.3g}, {d1[2]:+.3g}] | {d2[0]:+.3g} [{d2[1]:+.3g}, {d2[2]:+.3g}] |")
    rep = runs.REPO_ROOT / "reports" / ("phase5_dryrun" if a.limit else "phase5")
    rep.mkdir(parents=True, exist_ok=True)
    runs.atomic_write_text(rep / f"{sae_dir.name}_{a.graph}.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
