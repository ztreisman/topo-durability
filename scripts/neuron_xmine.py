"""Cross-layer neuron loops: fork and rejoin across any number of layers (Phase 5, neuron space). CPU.

  uv run python scripts/neuron_xmine.py --unit 1.4b_s0_step143000 --graph ctx
  ... --limit                  # test pass on data/neurons_dryrun: top 300 neurons per layer, 2 nulls
  uv run python scripts/neuron_xmine.py --unit 1.4b_s0_step143000 --summarize   # after all graphs

One graph over the MLP neurons of all layers (default: every layer with neuron_stats and neuron_xstats
output). Edges, from the same correlation type in and across layers (ctx = emergent candidate, tok =
inherited from token identity, total):
  within a layer: each neuron's top --k-within partners (as in neuron_mine.py)
  across layers:  each neuron's top --k-cross partners in ANY other layer (from X_L<l>_L<m>.npz)
The flag complex's H1 cycles are mined and every cycle is typed by its layer profile:
  flat     all neurons in one layer
  diamond  one lowest and one highest stretch: a fork and rejoin. pure = a single root neuron r (lowest) and a
           single target t (highest), i.e. r's influence splits into two unrelated arms (chordless: no edge
           between the arms) that rejoin at t. span = layer(t) - layer(r), any number of layers
  zigzag   several local tops and bottoms
Null: layered rewiring (each neuron keeps its degree into every layer, each layer pair keeps its edge
count and correlation values), degree-only and triangle-preserving. Two tests:
  1. a cycle beats the nulls (lifetime and score above the 95th percentile of null maxima)
  2. typology: counts of diamonds by span among the top cycles, real vs null (are multi-layer fork/rejoin
     loops more common than the layered wiring alone makes them?)
For good pure diamonds, the residual-stream check: direct_pct = percentile of r's direct virtual weight
onto t (r's output weights, mean-centred as LayerNorm does, dotted with t's LayerNorm-scaled input weights)
among r's weights onto every neuron of t's layer. Low direct_pct + high arm_pct = r reaches t only through
the arms (the chord the residual stream could supply is weak).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.graph.coact import knn_graph  # noqa: E402
from topodur.graph.cycles import is_chordless, mine_cycles, rewire  # noqa: E402

log = logging.getLogger("neuron_xmine")
VALS = ["ctx", "tok", "total"]
SPAN_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 99)]


# ------------------------------------------------------------------ typology

def typology(lay: np.ndarray) -> dict:
    L = np.asarray(lay)
    k = len(L)
    span = int(L.max() - L.min())
    base = {"type": "flat", "span": span, "n_peaks": 0, "pure": False, "monotone": False, "root": -1, "target": -1,
            "arm1": 0, "arm2": 0}
    if span == 0:
        return base
    start = next(i for i in range(k) if L[i] != L[i - 1])
    idx = np.roll(np.arange(k), -start)
    plats = []                                        # [layer, [positions]]
    for p in idx:
        if plats and plats[-1][0] == L[p]:
            plats[-1][1].append(int(p))
        else:
            plats.append([L[p], [int(p)]])
    vals = [p[0] for p in plats]
    n = len(vals)
    peaks = [i for i in range(n) if vals[i] > vals[i - 1] and vals[i] > vals[(i + 1) % n]]
    valleys = [i for i in range(n) if vals[i] < vals[i - 1] and vals[i] < vals[(i + 1) % n]]
    out = dict(base, type="diamond" if len(peaks) == 1 else "zigzag", n_peaks=len(peaks))
    if len(peaks) == 1:
        pk, vl = plats[peaks[0]][1], plats[valleys[0]][1]
        if len(pk) == 1 and len(vl) == 1:
            r, t = vl[0], pk[0]
            a1 = (t - r) % k                          # edges along one arm, the other arm has k - a1
            out.update(pure=True, root=r, target=t, arm1=min(a1, k - a1), arm2=max(a1, k - a1),
                       monotone=all(len(p[1]) == 1 for p in plats))
    return out


def span_bin(s: int) -> str:
    for a, b in SPAN_BINS:
        if a <= s <= b:
            return f"{a}" if a == b else f"{a}-{b}" if b < 99 else f">={a}"
    return "?"


def type_counts(cyc, layer_of) -> dict:
    """Counts of cycle types among mined cycles: diamond / pure diamond by span bin, zigzag, flat."""
    c = {}
    for cy in cyc:
        t = typology(layer_of[cy.vertices])
        key = "flat" if t["type"] == "flat" else "zigzag" if t["type"] == "zigzag" else \
            f"{'pure' if t['pure'] else 'diamond'} span {span_bin(t['span'])}"
        c[key] = c.get(key, 0) + 1
    return c


def _null_task(D, layer_of, kind, seed, top):
    N = rewire(D, preserve_triangles=(kind == "triangle"), seed=seed, node_block=layer_of)
    cyc = mine_cycles(N, top=top)
    return (max((c.lifetime for c in cyc), default=0.0), max((c.score for c in cyc), default=0.0),
            type_counts(cyc, layer_of))


# ------------------------------------------------------------------ graph

def available_layers(root: Path) -> list[int]:
    return sorted(int(m.group(1)) for p in root.glob("L*.npz") if (m := re.fullmatch(r"L(\d+)\.npz", p.name))
                  and runs.is_done(p))


def build_graph(root: Path, layers: list[int], g: str, a):
    """Global node ids, distance matrix D (1 - similarity of graph g), value matrices for ctx/tok/total on the
    same edges, node layer and neuron index, per-layer weights for the virtual-weight check."""
    nodes, info = {}, {}
    for l in layers:
        z = np.load(root / f"L{l:02d}.npz")
        sd = z["sd"]
        nd = np.flatnonzero(sd > 1e-6)
        if a.limit:
            nd = nd[np.argsort(-sd[nd])[: a.limit_neurons]]
        nodes[l] = np.sort(nd)
        info[l] = {"r2_tok": z["r2_tok"], "top_tok_ids": z["top_tok_ids"], "w_in": z["w_in"], "w_out": z["w_out"],
                   "ln_w": z["ln_w"] if "ln_w" in z else np.ones(z["w_in"].shape[1], np.float32)}
    off, tot = {}, 0
    for l in layers:
        off[l] = tot
        tot += len(nodes[l])
    layer_of = np.concatenate([np.full(len(nodes[l]), l) for l in layers])
    neuron_of = np.concatenate([nodes[l] for l in layers])
    R, C, V = [], [], {v: [] for v in VALS}
    # within-layer edges
    for l in layers:
        z = np.load(root / f"L{l:02d}.npz")
        ix = np.ix_(nodes[l], nodes[l])
        M = {v: z[f"corr_{v}"][ix].astype(np.float32) for v in VALS}
        Dl = sp.triu(knn_graph(M[g], a.k_within, a.min_sim), 1).tocoo()
        R.append(Dl.row + off[l]); C.append(Dl.col + off[l])
        for v in VALS:
            V[v].append(M[v][Dl.row, Dl.col])
        del M
    # cross-layer candidate edges
    loc = {}
    for l in layers:
        m_ = np.full(int(nodes[l].max()) + 1 if len(nodes[l]) else 1, -1)
        m_[nodes[l]] = np.arange(len(nodes[l]))
        loc[l] = m_
    cr, cc, cv = [], [], {v: [] for v in VALS}
    missing = []
    for i, l in enumerate(layers):
        for m in layers[i + 1:]:
            p = root / f"X_L{l:02d}_L{m:02d}.npz"
            if not runs.is_done(p):
                missing.append((l, m))
                continue
            x = np.load(p)
            ii, jj = x["i"].astype(np.int64), x["j"].astype(np.int64)
            ok = (ii < len(loc[l])) & (jj < len(loc[m]))
            ii, jj = ii[ok], jj[ok]
            gi, gj = loc[l][ii], loc[m][jj]
            keep = (gi >= 0) & (gj >= 0) & (x[g][ok].astype(np.float32) > a.min_sim)
            cr.append(gi[keep] + off[l]); cc.append(gj[keep] + off[m])
            for v in VALS:
                cv[v].append(x[v][ok][keep].astype(np.float32))
    if missing:
        log.warning("%d layer pairs have no cross stats (first: %s); their cross edges are absent", len(missing), missing[:3])
    if cr:
        cr, cc = np.concatenate(cr), np.concatenate(cc)
        cv = {v: np.concatenate(cv[v]) for v in VALS}
        # each node's top k_cross partners over all other layers (union over both endpoints, like knn_graph)
        u = np.concatenate([cr, cc]); s = np.concatenate([cv[g], cv[g]])
        e = np.concatenate([np.arange(len(cr))] * 2)
        o = np.lexsort((-s, u))
        u, e = u[o], e[o]
        first = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
        rank = np.arange(len(u)) - np.repeat(first, np.diff(np.r_[first, len(u)]))
        sel = np.unique(e[rank < a.k_cross])
        R.append(cr[sel]); C.append(cc[sel])
        for v in VALS:
            V[v].append(cv[v][sel])
    R, C = np.concatenate(R), np.concatenate(C)
    V = {v: np.concatenate(V[v]) for v in VALS}
    sym = lambda x: (lambda M: M.maximum(M.T).tocsr())(sp.coo_matrix((x, (R, C)), shape=(tot, tot)).tocsr())
    D = sym(np.maximum(1 - V[g], 1e-6))
    Vm = {v: sp.coo_matrix((V[v], (R, C)), shape=(tot, tot)).tocsr() for v in VALS}
    Vm = {v: (M + M.T).tocsr() for v, M in Vm.items()}             # no duplicates: R, C pairs are unique
    return D, Vm, layer_of, neuron_of, info


def virtual_pct(info, la, na, lb, nb) -> float:
    """Percentile of |direct residual-path weight| from neuron na (layer la) onto nb (layer lb > la) among na's
    weights onto every neuron of layer lb."""
    wo = info[la]["w_out"][na].astype(np.float32)
    wo = wo - wo.mean()                                            # LayerNorm removes the mean over d
    Win = info[lb]["w_in"].astype(np.float32) * info[lb]["ln_w"][None, :]
    row = np.abs(Win @ wo)
    return float((row < row[nb]).mean())


# ------------------------------------------------------------------ mining

def mine(a) -> int:
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(root / f"xmine_{a.graph}.log")
    out = root / f"xmine_{a.graph}.json"
    if runs.is_done(out) and not a.force:
        log.info("already mined")
        return 0
    layers = a.layers or available_layers(root)
    D, Vm, layer_of, neuron_of, info = build_graph(root, layers, a.graph, a)
    n_cross = int(((layer_of[sp.triu(D, 1).tocoo().row] != layer_of[sp.triu(D, 1).tocoo().col])).sum())
    log.info("graph %s: %d neurons in %d layers, %d edges (%d across layers)", a.graph, D.shape[0], len(layers),
             D.nnz // 2, n_cross)
    cyc = mine_cycles(D, top=a.top)
    log.info("mined %d H1 cycles", len(cyc))
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_null_task, D, layer_of, kind, a.seed + 1000 * j + s, a.top)
                for j, kind in enumerate(("degree", "triangle")) for s in range(a.n_null)]
        nl = [f.result() for f in futs]
    life = np.array([x[0] for x in nl]); score = np.array([x[1] for x in nl])
    thr_life, thr_score = np.quantile(life, 0.95), np.quantile(score, 0.95)
    rows = []
    for i, c in enumerate(cyc):
        v = np.asarray(c.vertices)
        t = typology(layer_of[v])
        e = [(v[j], v[(j + 1) % len(v)]) for j in range(len(v))]
        ctx = np.mean([Vm["ctx"][x, y] for x, y in e]); tok = np.mean([Vm["tok"][x, y] for x, y in e])
        row = {"graph": a.graph, "rank": i, "n_vertices": len(v), "birth": c.birth, "lifetime": c.lifetime,
               "score": c.score, "chordless": is_chordless(D, c.vertices, c.birth),
               "beats_nulls": bool(c.lifetime > thr_life and c.score > thr_score), **t,
               "ce_cross": bool(layer_of[v[0]] != layer_of[v[-1]]), "ce_span": int(abs(layer_of[v[0]] - layer_of[v[-1]])),
               "tok_share": float(tok / max(ctx, 1e-6)),
               "layers": layer_of[v].tolist(), "neurons": neuron_of[v].tolist()}
        if t["pure"]:
            r, tg = int(v[t["root"]]), int(v[t["target"]])
            lr, lt = int(layer_of[r]), int(layer_of[tg])
            row["direct_pct"] = virtual_pct(info, lr, neuron_of[r], lt, neuron_of[tg])
            arm = [virtual_pct(info, int(layer_of[x]), neuron_of[x], int(layer_of[y]), neuron_of[y])
                   if layer_of[x] < layer_of[y] else
                   virtual_pct(info, int(layer_of[y]), neuron_of[y], int(layer_of[x]), neuron_of[x])
                   for x, y in e if layer_of[x] != layer_of[y] and {x, y} != {r, tg}]
            row["arm_pct"] = float(np.mean(arm)) if arm else np.nan
            row["root_r2_tok"] = float(info[lr]["r2_tok"][neuron_of[r]])
            row["target_r2_tok"] = float(info[lt]["r2_tok"][neuron_of[tg]])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(root / f"xcycles_{a.graph}.parquet", index=False)
    real = type_counts(cyc, layer_of)
    keys = sorted(set(real) | {k for x in nl for k in x[2]})
    nt = pd.DataFrame([{**{k: x[2].get(k, 0) for k in keys}, "kind": kind}
                       for x, kind in zip(nl, ["degree"] * a.n_null + ["triangle"] * a.n_null)])
    nt.to_parquet(root / f"xnull_{a.graph}.parquet", index=False)
    typ = []
    for k in keys:
        for kind in ("degree", "triangle"):
            x = nt[nt.kind == kind][k].to_numpy()
            typ.append({"type": k, "null": kind, "real": real.get(k, 0), "null_mean": float(x.mean()),
                        "null_95": float(np.quantile(x, 0.95)), "p": float((1 + (x >= real.get(k, 0)).sum()) / (1 + len(x)))})
    good = df[df.beats_nulls & df.chordless & (df.n_vertices >= 4)] if len(df) else df
    summary = {"graph": a.graph, "layers": layers, "n_nodes": int(D.shape[0]), "edges": int(D.nnz // 2),
               "cross_edges": n_cross, "h1": len(cyc), "null_life_95": float(thr_life), "null_score_95": float(thr_score),
               "good": int(len(good)), "good_pure": int(good.pure.sum()) if len(good) else 0,
               "good_pure_span_ge2": int((good.pure & (good.span >= 2)).sum()) if len(good) else 0,
               "typology": typ}
    runs.atomic_write_text(out, json.dumps(summary, indent=1))
    runs.mark_done(out, {})
    log.info("%s", {k: v for k, v in summary.items() if k != "typology"})
    write_examples(root, rep, a.graph, good, info)
    return 0


def write_examples(root, rep, g, good, info, n=12):
    vocab = json.loads((root / "vocab.json").read_text()) if (root / "vocab.json").exists() else None
    pure = good[good.pure].sort_values(["span", "score"], ascending=False).head(n) if len(good) else good
    lines = [f"# Cross-layer fork/rejoin loops ({g} graph): longest-span good pure diamonds", "",
             "Root = lowest neuron, target = highest; the two arms are listed in cycle order. "
             "direct_pct: percentile of the root's direct residual weight onto the target (low = only via the arms).", ""]
    for r in pure.itertuples():
        lines.append(f"## span {r.span} (L{min(r.layers)} -> L{max(r.layers)}), {r.n_vertices} neurons, "
                     f"arms {r.arm1}+{r.arm2} edges, lifetime {r.lifetime:.3f}, direct_pct {r.direct_pct:.2f}, "
                     f"arm_pct {r.arm_pct:.2f}, tok_share {r.tok_share:.2f}")
        k = len(r.neurons)
        order = [(r.root + j) % k for j in range(k)]
        for j in order:
            l, nn = r.layers[j], r.neurons[j]
            ids = info[l]["top_tok_ids"][nn][:8]
            toks = [vocab[int(t)] if vocab else str(int(t)) for t in ids]
            mark = " (root)" if j == r.root else " (target)" if j == r.target else ""
            lines.append(f"- L{l} n{nn}{mark} r2_tok {info[l]['r2_tok'][nn]:.2f}: " + ", ".join(repr(t) for t in toks))
        lines.append("")
    runs.atomic_write_text(rep / f"crosslayer_{g}_examples.md", "\n".join(lines))


def summarize(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    S = {g: json.loads((root / f"xmine_{g}.json").read_text()) for g in VALS if (root / f"xmine_{g}.json").exists()}
    if not S:
        log.error("nothing mined in %s", root)
        return 1
    lines = [f"# Cross-layer neuron loops: {a.unit}", "",
             "One graph over the MLP neurons of all layers; edges within and across layers from one correlation type.",
             "Good = chordless, >= 4 neurons, beats layered rewiring nulls (degree- and triangle-preserving).",
             "Pure diamond = fork at one root neuron, two unrelated arms, rejoin at one target; span = layers from root to target.", "",
             "| graph | neurons | edges (cross) | H1 | good | good pure diamonds | ... span >= 2 |", "|---|---|---|---|---|---|---|"]
    for g, s in S.items():
        lines.append(f"| {g} | {s['n_nodes']} | {s['edges']} ({s['cross_edges']}) | {s['h1']} | {s['good']} | "
                     f"{s['good_pure']} | {s['good_pure_span_ge2']} |")
    lines += ["", "## Typology of the top cycles: real vs layered null (triangle-preserving; p = share of nulls >= real)", ""]
    for g, s in S.items():
        t = pd.DataFrame(s["typology"])
        t = t[t.null == "triangle"]
        lines += [f"### {g}", "", "| type | real | null mean | null 95% | p |", "|---|---|---|---|---|"]
        for r in t.itertuples():
            lines.append(f"| {r.type} | {r.real} | {r.null_mean:.1f} | {r.null_95:.1f} | {r.p:.3f} |")
        lines.append("")
    pure = []
    for g in S:
        p = root / f"xcycles_{g}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d = d[d.beats_nulls & d.chordless & (d.n_vertices >= 4) & d.pure]
            if len(d):
                pure.append(d)
    if pure:
        P = pd.concat(pure)
        lines += ["## Good pure diamonds: residual-path check (medians)", "",
                  "| graph | n | median span | direct_pct | arm_pct | root r2_tok | target r2_tok | tok_share |",
                  "|---|---|---|---|---|---|---|---|"]
        for g, d in P.groupby("graph"):
            lines.append(f"| {g} | {len(d)} | {d.span.median():.0f} | {d.direct_pct.median():.2f} | {d.arm_pct.median():.2f} | "
                         f"{d.root_r2_tok.median():.2f} | {d.target_r2_tok.median():.2f} | {d.tok_share.median():.2f} |")
        lines.append("")
    lines += ["Examples: crosslayer_<graph>_examples.md. ![spans](crosslayer.png)", ""]
    fig, axes = plt.subplots(1, len(S), figsize=(4.2 * len(S), 3.4), squeeze=False)
    for ax, (g, s) in zip(axes[0], S.items()):
        t = pd.DataFrame(s["typology"])
        t = t[(t.null == "triangle") & t.type.str.startswith("pure")]
        t = t.assign(o=t.type.str.extract(r"(\d+)")[0].astype(int)).sort_values("o")
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.real, 0.4, color="#2a6f97", label="real")
        ax.bar(x + 0.2, t.null_mean, 0.4, color="#c9c9c9", label="layered null")
        ax.errorbar(x + 0.2, t.null_mean, yerr=[np.zeros(len(t)), t.null_95 - t.null_mean], fmt="none", ecolor="#666", lw=1)
        ax.set_xticks(x, [k.replace("pure span ", "") for k in t.type])
        ax.set_xlabel("span (layers, root to target)"); ax.set_title(f"{g}: pure fork/rejoin loops", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("count among top cycles"); axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(rep / "crosslayer.png", dpi=130)
    runs.atomic_write_text(rep / "crosslayer.md", "\n".join(lines))
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--graph", choices=VALS, default="ctx")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--k-within", type=int, default=15)
    ap.add_argument("--k-cross", type=int, default=15)
    ap.add_argument("--min-sim", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=1000, help="cycles kept (real and null) for the typology")
    ap.add_argument("--n-null", type=int, default=10, help="per null kind")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", action="store_true")
    ap.add_argument("--limit-neurons", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.limit:
        a.n_null = min(a.n_null, 2)
    return summarize(a) if a.summarize else mine(a)


if __name__ == "__main__":
    sys.exit(main())
