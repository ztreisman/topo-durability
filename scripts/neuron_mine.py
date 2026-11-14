"""Neuron-space loops for one layer (primary Phase 5 analysis; decided with Zack 2026-11-13). CPU.

  uv run python scripts/neuron_mine.py --unit 1.4b_s0_step143000 --layer 14
  ... --limit                # test pass on data/neurons_dryrun: few nulls, top 300 neurons by variance
  uv run python scripts/neuron_mine.py --unit 1.4b_s0_step143000 --summarize   # after all layers

Graphs on MLP neurons (k-NN sparsified, distance = 1 - similarity), mined for H1 cycles (flag complex,
src/topodur/graph/cycles.py) and tested against degree- and triangle-preserving rewiring nulls:
  ctx    within-token correlation: co-variation the network adds beyond the current token (EMERGENT candidates)
  tok    between-token correlation: co-variation fixed by the token's identity (INHERITED from the input)
  total  ordinary activation correlation
  out    cosine of the neurons' output weight vectors (what they write; functional side)
Closing-edge test (primary effect; the closing edge of an H1 class is its birth edge, the last edge whose
addition closes the loop), per null-beating chordless ctx cycle of >= 4 neurons:
  ce_out        output-weight cosine of the two neurons the closing edge joins
  d_path        ce_out - mean output cosine of pairs at the same hop distance (k-1) that are NOT joined
                (what a line would give); > 0 = the loop closes on the functional side too
  d_edge        ce_out - mean output cosine of graph edges with the same input (ctx) correlation
  tok_share     mean between-token / within-token correlation along the cycle's edges (inheritance)
Calibration (token-level families, kept as the known-answer set): neurons tuned to the family's
single-token items, as points with their tuning curves; a ring should appear for months/days and not
for line controls; validated against each neuron's preferred item.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from topodur import nulls, runs, tda  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.geometry import distance_matrix  # noqa: E402
from topodur.graph.coact import knn_graph  # noqa: E402
from topodur.graph.cycles import is_chordless, mine_cycles, rewire  # noqa: E402

log = logging.getLogger("neuron_mine")
GRAPHS = ["ctx", "tok", "total", "out"]
LINE_FAMILIES = {"numbers", "ordinals", "sizes"}


def similarity(z, g: str) -> np.ndarray:
    if g == "out":
        W = z["w_out"].astype(np.float32)
        W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-8)
        return W @ W.T
    return z[f"corr_{g}"].astype(np.float32)


def _null_task(D: sp.csr_matrix, kind: str, seed: int) -> tuple[float, float]:
    N = rewire(D, preserve_triangles=(kind == "triangle"), seed=seed)
    cyc = mine_cycles(N, top=50)
    return max((c.lifetime for c in cyc), default=0.0), max((c.score for c in cyc), default=0.0)


def hop_pairs(A: sp.csr_matrix, hops: int, n: int, rng, exclude_adjacent=True) -> list[tuple[int, int]]:
    """Random node pairs at exactly `hops` shortest-path hops in A (not adjacent)."""
    out, N = [], A.shape[0]
    indptr, ind = A.indptr, A.indices
    for _ in range(n * 20):
        s = int(rng.integers(N))
        dist = {s: 0}
        q = deque([s])
        while q:
            x = q.popleft()
            if dist[x] >= hops:
                continue
            for y in ind[indptr[x]:indptr[x + 1]]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        far = [y for y, d in dist.items() if d == hops]
        if far:
            out.append((s, int(rng.choice(far))))
            if len(out) >= n:
                break
    return out


def closing_edge_rows(D: sp.csr_matrix, cyc_df: pd.DataFrame, Sout: np.ndarray, Sctx: np.ndarray,
                      Stok: np.ndarray, rng, n_ctrl: int = 50) -> pd.DataFrame:
    E = sp.triu(D, k=1).tocoo()
    e_corr = Sctx[E.row, E.col]
    rows = []
    for r in cyc_df.itertuples():
        v = list(r.vertices)
        k = len(v)
        u, w = v[0], v[-1]                     # birth edge = closing edge (representative is u ... w)
        A = D.copy()
        A.data[A.data > r.birth + 1e-12] = 0
        A.eliminate_zeros()
        A = A.tolil()
        A[u, w] = 0
        A[w, u] = 0
        A = A.tocsr()
        A.eliminate_zeros()
        ce = float(Sout[u, w])
        other = [float(Sout[v[i], v[i + 1]]) for i in range(k - 1)]
        pairs = hop_pairs(A, k - 1, n_ctrl, rng)
        path = [float(Sout[a, b]) for a, b in pairs]
        near = np.argsort(np.abs(e_corr - Sctx[u, w]))[:n_ctrl]
        edge = Sout[E.row[near], E.col[near]]
        cyc_edges = [(v[i], v[(i + 1) % k]) for i in range(k)]
        tok_share = float(np.mean([Stok[a, b] for a, b in cyc_edges]) / max(np.mean([Sctx[a, b] for a, b in cyc_edges]), 1e-6))
        rows.append({"rank": r.rank, "k": k, "ce_out": ce, "edges_out": float(np.mean(other)),
                     "path_out": float(np.mean(path)) if path else np.nan, "n_path_pairs": len(pairs),
                     "edge_out_matched": float(edge.mean()),
                     "d_path": ce - (float(np.mean(path)) if path else np.nan), "d_edge": ce - float(edge.mean()),
                     "tok_share": tok_share})
    return pd.DataFrame(rows)


def calibrate(z, calib: dict, n_neurons: int, n_null: int, seed: int, min_count: int = 20) -> pd.DataFrame:
    ids = list(calib["ids"])
    col = {t: j for j, t in enumerate(ids)}
    mu, sd = z["calib_mu"], z["sd"]
    n_tok = z["calib_n"] if "calib_n" in z else np.full(len(ids), np.inf)
    rows = []
    for fam_name, pairs in calib["families"].items():
        pairs = [(i, t) for i, t in pairs if n_tok[col[t]] >= min_count]   # rare tokens: unreliable means
        if len(pairs) < 5:
            continue
        fam = load_family(fam_name)
        items = [i for i, _ in pairs]
        T = mu[:, [col[t] for _, t in pairs]].astype(np.float64)        # [N, I]
        sel = np.argsort(-(T.var(1) / np.maximum(sd.astype(np.float64) ** 2, 1e-12)))[:n_neurons]
        X = T[sel] - T[sel].mean(1, keepdims=True)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        D = distance_matrix(X, "cosine")
        sc = tda.scores(tda.diagrams(D)["dgms"])
        nl = nulls.gaussian_null(X, "cosine", 1, n_null, seed)
        p = nulls.empirical_p(nl["h1_max"], sc["h1_max"])
        val = val_p = np.nan
        cc = tda.circular_coordinates(D)
        if cc is not None:
            pref = np.asarray(items)[T[sel].argmax(1)]
            truth = fam.angles()[pref]
            val, val_p = tda.order_test(cc.theta, truth, n_perm=500, seed=seed)
        rows.append({"family": fam_name, "line_control": fam_name in LINE_FAMILIES or fam.topology == "interval",
                     "n_items": len(pairs), "h1_max": sc["h1_max"], "p": p, "val": val, "val_p": val_p,
                     "ring": bool(p < 0.05 and np.nan_to_num(val_p, nan=1) < 0.05)})
    return pd.DataFrame(rows)


def mine_layer(a) -> int:
    root = runs.DATA_DIR / ("neurons_dryrun" if a.limit else "neurons") / a.unit
    rep = runs.REPO_ROOT / "reports" / ("neurons_dryrun" if a.limit else "neurons") / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(root / f"mine_L{a.layer:02d}.log")
    out = root / f"mine_L{a.layer:02d}.json"
    if runs.is_done(out) and not a.force:
        log.info("layer %d already mined", a.layer)
        return 0
    z = dict(np.load(root / f"L{a.layer:02d}.npz"))
    calib = json.loads((root / "calib.json").read_text())
    rng = np.random.default_rng(a.seed)
    nodes = np.arange(len(z["sd"]))
    nodes = nodes[z["sd"] > 1e-6]
    if a.limit:
        nodes = nodes[np.argsort(-z["sd"][nodes])[:300]]
    sub = lambda S: S[np.ix_(nodes, nodes)]
    S = {g: sub(similarity(z, g)) for g in GRAPHS}
    summary, cyc_tables, ce_tables = {"layer": a.layer, "n_nodes": int(len(nodes))}, [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for g in GRAPHS:
            D = knn_graph(S[g], a.k_nn, a.min_sim)
            cyc = mine_cycles(D, top=500)
            futs = [ex.submit(_null_task, D, kind, a.seed + 1000 * j + s) for j, kind in enumerate(("degree", "triangle"))
                    for s in range(a.n_null)]
            nl = np.array([f.result() for f in futs])
            thr_life, thr_score = np.quantile(nl[:, 0], 0.95), np.quantile(nl[:, 1], 0.95)
            df = pd.DataFrame([{"graph": g, "rank": i, "n_vertices": len(c.vertices), "birth": c.birth,
                                "death": c.death, "lifetime": c.lifetime, "score": c.score,
                                "chordless": is_chordless(D, c.vertices, c.birth),
                                "beats_nulls": bool(c.lifetime > thr_life and c.score > thr_score),
                                "vertices": list(map(int, c.vertices)), "neurons": nodes[c.vertices].tolist()}
                               for i, c in enumerate(cyc)])
            if df.empty:
                df = pd.DataFrame(columns=["graph", "rank", "n_vertices", "birth", "death", "lifetime", "score",
                                           "chordless", "beats_nulls", "vertices", "neurons"])
            good = df[df.beats_nulls & df.chordless & (df.n_vertices >= 4)] if len(df) else df
            summary[g] = {"edges": int(D.nnz // 2), "h1": int(len(df)), "beat_nulls": int(df.beats_nulls.sum()) if len(df) else 0,
                          "good_cycles": int(len(good)), "null_life_95": float(thr_life), "null_score_95": float(thr_score)}
            log.info("layer %d %s: %s", a.layer, g, summary[g])
            cyc_tables.append(df)
            if len(good) and g in ("ctx", "tok", "total"):
                ce = closing_edge_rows(D, good.head(a.max_cycles), S["out"], S["ctx"], S["tok"], rng)
                ce["graph"] = g
                ce_tables.append(ce)
    cyc_all = pd.concat(cyc_tables, ignore_index=True)
    cyc_all["layer"] = a.layer
    cyc_all.to_parquet(root / f"cycles_L{a.layer:02d}.parquet", index=False)
    ce_all = pd.concat(ce_tables, ignore_index=True) if ce_tables else pd.DataFrame()
    if len(ce_all):
        ce_all["layer"] = a.layer
    ce_all.to_parquet(root / f"closing_L{a.layer:02d}.parquet", index=False)
    cal = calibrate(z, calib, a.calib_neurons, a.n_null_calib, a.seed)
    cal["layer"] = a.layer
    cal.to_parquet(root / f"calib_L{a.layer:02d}.parquet", index=False)
    summary["closing"] = {g: {"n": int((ce_all.graph == g).sum()), "d_path_mean": float(ce_all[ce_all.graph == g].d_path.mean()),
                              "d_edge_mean": float(ce_all[ce_all.graph == g].d_edge.mean())}
                          for g in ce_all.graph.unique()} if len(ce_all) else {}
    summary["calib_rings"] = cal[cal.ring].family.tolist() if len(cal) else []
    summary["median_r2_tok"] = float(np.median(z["r2_tok"]))
    runs.atomic_write_text(out, json.dumps(summary, indent=1))
    runs.mark_done(out, {})
    # human-readable top cycles: top tokens of each neuron
    top = cyc_all[(cyc_all.graph == "ctx") & cyc_all.beats_nulls & cyc_all.chordless].head(10)
    lines = [f"# Layer {a.layer}: top emergent-candidate (ctx) cycles", "",
             "Neurons in cycle order; the closing edge joins the first and last. Top tokens = largest mean activation by token.", ""]
    tok_names = json.loads((root / "vocab.json").read_text()) if (root / "vocab.json").exists() else None
    for r in top.itertuples():
        lines.append(f"## rank {r.rank}: {r.n_vertices} neurons, lifetime {r.lifetime:.3f}")
        for n in r.neurons:
            ids = z["top_tok_ids"][n][:8]
            toks = [tok_names[int(t)] if tok_names else str(int(t)) for t in ids]
            lines.append(f"- n{n} (r2_tok {z['r2_tok'][n]:.2f}): " + ", ".join(repr(t) for t in toks))
        lines.append("")
    runs.atomic_write_text(rep / f"L{a.layer:02d}_cycles.md", "\n".join(lines))
    return 0


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    S = [json.loads(p.read_text()) for p in sorted(root.glob("mine_L*.json"))]
    if not S:
        log.error("no mined layers in %s", root)
        return 1
    ce = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("closing_L*.parquet"))], ignore_index=True)
    cal = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("calib_L*.parquet"))], ignore_index=True)
    lt = pd.DataFrame([{"layer": s["layer"], "median_r2_tok": s["median_r2_tok"],
                        **{f"{g}_good": s[g]["good_cycles"] for g in GRAPHS if g in s}} for s in S]).sort_values("layer")
    lines = [f"# Neuron-space loops: {a.unit}", "",
             "Nodes: MLP neurons. Good cycle = chordless, >= 4 neurons, lifetime and score above the 95th percentile of",
             "degree- and triangle-preserving rewiring nulls. ctx = within-token correlation (emergent candidates),",
             "tok = between-token correlation (inherited from token identity), out = output-weight cosine.", "",
             "## Cycles per layer", "", lt.to_string(index=False), "",
             "## Closing-edge test (pooled over layers; mean [95% bootstrap CI over cycles])", "",
             "d_path > 0: the two neurons the closing edge joins write more similar outputs than pairs at the same",
             "hop distance that are not joined (the loop closes functionally). d_edge ~ 0: the closing edge acts like",
             "an ordinary edge of the same input strength.", ""]
    for g in ["ctx", "tok", "total"]:
        x = ce[ce.graph == g] if len(ce) else ce
        if len(x):
            dp, de = boot_ci(x.d_path), boot_ci(x.d_edge)
            lines.append(f"- {g}: n={len(x)} cycles; d_path {dp[0]:+.4f} [{dp[1]:+.4f}, {dp[2]:+.4f}]; "
                         f"d_edge {de[0]:+.4f} [{de[1]:+.4f}, {de[2]:+.4f}]; median tok_share {x.tok_share.median():.2f}")
    lines += ["", "## Calibration (token-level families as neuron tuning rings; known answers)", "",
              cal.pivot_table(index="family", columns="layer", values="ring", aggfunc="first").astype(float)
              .fillna(-1).astype(int).replace(-1, ".").to_string() if len(cal) else "(none)", "",
              "1 = ring (Gaussian-null p < 0.05 and order matches the neurons' preferred items).", "",
              "![layers](neuron_layers.png)", ""]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for g, c in zip(["ctx", "tok", "total", "out"], ["#2a6f97", "#c9772f", "#6c757d", "#7b2cbf"]):
        if f"{g}_good" in lt:
            ax[0].plot(lt.layer, lt[f"{g}_good"], marker="o", ms=3, lw=2, color=c, label=g)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("null-beating chordless cycles"); ax[0].legend(frameon=False)
    if len(ce):
        for g, c in zip(["ctx", "tok"], ["#2a6f97", "#c9772f"]):
            x = ce[ce.graph == g].groupby("layer").d_path.mean()
            ax[1].plot(x.index, x.values, marker="o", ms=3, lw=2, color=c, label=g)
        ax[1].axhline(0, color="#999", lw=1)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("closing edge d_path (mean)"); ax[1].legend(frameon=False)
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rep / "neuron_layers.png", dpi=130)
    runs.atomic_write_text(rep / "summary.md", "\n".join(lines))
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--k-nn", type=int, default=15)
    ap.add_argument("--min-sim", type=float, default=0.1)
    ap.add_argument("--n-null", type=int, default=10, help="per null kind")
    ap.add_argument("--max-cycles", type=int, default=100)
    ap.add_argument("--calib-neurons", type=int, default=128)
    ap.add_argument("--n-null-calib", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.limit:
        a.n_null, a.n_null_calib, a.max_cycles = 2, 20, 10
    return summarize(a) if a.summarize else mine_layer(a)


if __name__ == "__main__":
    sys.exit(main())
