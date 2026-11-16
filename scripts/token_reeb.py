"""Fork/rejoin (Reeb graph) study of single-token footprints across layers (src/topodur/reeb.py). CPU.

  uv run python scripts/token_reeb.py --unit 1.4b_s0_step143000 --mode mean   # version A: from saved means
  uv run python scripts/token_reeb.py --unit 1.4b_s0_step143000 --mode ctx    # version B: token_footprint.py data
  ... --limit                                                                   # data/neurons_dryrun

Per token and layer: active neurons (z-score of the token's response > --z-min, at most --max-active), grouped
into clusters (correlation > tau), clusters at adjacent layers linked when coupled.
  mean: correlation = global within-token (ctx) correlation from neuron_stats; link = share of cross-layer ctx
        edges (X_L<l>_L<l+1>, value > tau) between the two clusters' neurons, normalised by sqrt(|a| |b|)
  ctx:  correlation across THIS token's own contexts (so different senses can separate); link = correlation
        across contexts of the two clusters' mean activity
Reported by token class (polysemous / monosemous / function / ring_items / line_items): clusters per layer,
forks, rejoins, independent loops (b1 = cycle rank of the Reeb graph), loop spans, branch entropy.
Known answer: polysemous tokens should split into more clusters than monosemous ones (senses).
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

from topodur import runs  # noqa: E402
from topodur.reeb import active_neurons, clusters, panel, reeb_stats  # noqa: E402

log = logging.getLogger("token_reeb")
CLASSES = ["polysemous", "monosemous", "function", "ring_items", "line_items"]


def token_ids(root: Path, pan: dict) -> list[tuple[str, str, int]]:
    vocab = json.loads((root / "vocab.json").read_text())
    rev = {}
    for i, s in enumerate(vocab):
        rev.setdefault(s, i)
    out = []
    for cls, words in pan.items():
        for w in words:
            t = rev.get(" " + w)
            if t is not None:
                out.append((cls, w, t))
    return out


def run_mean(root, toks, layers, a):
    tc = np.load(root / "tokcounts.npz")
    row = {int(t): k for k, t in enumerate(tc["ids"])}
    toks = [x for x in toks if x[2] in row and tc["cnt"][row[x[2]]] >= a.min_count]
    act, cl, mass = {}, {}, {}
    for l in layers:
        z = np.load(root / f"L{l:02d}.npz")
        mu, sd = z["mean"], np.maximum(z["sd"], 1e-6)
        C = z["corr_ctx"]
        tm = np.load(root / f"L{l:02d}_tokmeans.npy", mmap_mode="r")
        for cls, w, t in toks:
            zz = (tm[row[t]].astype(np.float32) - mu) / sd
            A = active_neurons(zz, a.z_min, a.max_active)
            act[(w, l)] = A
            for tau in a.tau:
                c = clusters(C[np.ix_(A, A)].astype(np.float32), tau, a.min_size) if len(A) > 1 else []
                cl[(w, l, tau)] = [A[k] for k in c]
                mass[(w, l, tau)] = [float(zz[A[k]].sum()) for k in c]
        log.info("layer %d: median active %d", l, int(np.median([len(act[(w, l)]) for _, w, _ in toks])))
    links = {}
    for l in layers[:-1]:
        p = root / f"X_L{l:02d}_L{l + 1:02d}.npz"
        if not runs.is_done(p):
            log.warning("no cross stats for L%d-L%d: no links", l, l + 1)
            continue
        x = np.load(p)
        ii, jj, v = x["i"].astype(np.int64), x["j"].astype(np.int64), x["ctx"].astype(np.float32)
        for tau in a.tau:
            keep = v > tau
            E = set(zip(ii[keep].tolist(), jj[keep].tolist()))
            for _, w, _ in toks:
                for i, ca in enumerate(cl[(w, l, tau)]):
                    for j, cb in enumerate(cl[(w, l + 1, tau)]):
                        n = sum((u, q) in E for u in ca.tolist() for q in cb.tolist())
                        if n:
                            links[(w, tau, l, i, j)] = n / np.sqrt(len(ca) * len(cb))
    return toks, cl, mass, links


n_ctx: dict[str, int] = {}


def run_ctx(root, toks, layers, a):
    fp = root / "footprint"
    st = {l: np.load(root / f"L{l:02d}.npz") for l in layers}
    mu = {l: st[l]["mean"] for l in layers}
    sd = {l: np.maximum(st[l]["sd"], 1e-6) for l in layers}
    keep, cl, mass, links = [], {}, {}, {}
    for cls, w, t in toks:
        f = fp / f"T{t}.npy"
        if not f.exists():
            continue
        X = np.load(f, mmap_mode="r")                                       # [n_ctx, n_layers_saved, N]
        if len(X) < max(a.min_ctx, a.n_ctx_eq):
            continue
        if a.n_ctx_eq:
            X = X[:a.n_ctx_eq]                          # same number of contexts for every token
        keep.append((cls, w, t))
        n_ctx[w] = len(X)
        lay_index = json.loads((fp / "footprint.json").read_text())["layers"]
        acts = {}
        for l in layers:
            Z = (X[:, lay_index.index(l)].astype(np.float32) - mu[l]) / sd[l]   # [n_ctx, N]
            A = active_neurons(Z.mean(0), a.z_min, a.max_active)
            acts[l] = (A, Z[:, A])
            for tau in a.tau:
                if len(A) > 1:
                    C = np.nan_to_num(np.corrcoef(Z[:, A].T))
                    c = clusters(C, tau, a.min_size)
                else:
                    c = []
                cl[(w, l, tau)] = [A[k] for k in c]
                mass[(w, l, tau)] = [float(Z[:, A[k]].mean(0).sum()) for k in c]
        for l in layers[:-1]:
            for tau in a.tau:
                (A0, Z0), (A1, Z1) = acts[l], acts[l + 1]
                pos0 = {n: k for k, n in enumerate(A0)}
                pos1 = {n: k for k, n in enumerate(A1)}
                for i, ca in enumerate(cl[(w, l, tau)]):
                    ma = Z0[:, [pos0[n] for n in ca]].mean(1)
                    for j, cb in enumerate(cl[(w, l + 1, tau)]):
                        mb = Z1[:, [pos1[n] for n in cb]].mean(1)
                        r = np.corrcoef(ma, mb)[0, 1]
                        if np.isfinite(r) and r > 0:
                            links[(w, tau, l, i, j)] = float(r)
    return keep, cl, mass, links


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--mode", choices=["mean", "ctx"], default="mean")
    ap.add_argument("--panel", default=str(runs.REPO_ROOT / "configs" / "token_panel.yaml"))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--max-active", type=int, default=128)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--w-min", type=float, default=None, help="link threshold (mean: 0.1, ctx: 0.2)")
    ap.add_argument("--min-count", type=int, default=50, help="mean mode: token occurrences needed")
    ap.add_argument("--min-ctx", type=int, default=64, help="ctx mode: contexts needed")
    ap.add_argument("--n-ctx-eq", type=int, default=0,
                    help="ctx mode: use exactly this many contexts per token, dropping tokens with fewer "
                         "(controls for frequency); outputs get the suffix _eq<N>")
    ap.add_argument("--limit", action="store_true")
    a = ap.parse_args()
    a.tag = a.mode + (f"_eq{a.n_ctx_eq}" if a.n_ctx_eq else "")
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(root / f"reeb_{a.tag}.log")
    if a.limit:
        a.min_count, a.min_ctx = 5, 4
    w_min = a.w_min if a.w_min is not None else (0.1 if a.mode == "mean" else 0.2)
    layers = a.layers or sorted(int(p.name[1:3]) for p in root.glob("L??.npz") if runs.is_done(p))
    calib = json.loads((root / "calib.json").read_text())
    toks = token_ids(root, panel(a.panel, calib))
    log.info("%d panel tokens, layers %s", len(toks), layers)
    toks, cl, mass, links = (run_mean if a.mode == "mean" else run_ctx)(root, toks, layers, a)
    rows = []
    for cls, w, t in toks:
        for tau in a.tau:
            L = {(l, i, j): v for (ww, tt, l, i, j), v in links.items() if ww == w and tt == tau}
            s = reeb_stats(layers, {l: cl[(w, l, tau)] for l in layers}, L, w_min,
                           {l: mass[(w, l, tau)] for l in layers})
            rows.append({"class": cls, "token": w, "token_id": t, "tau": tau, "n_ctx": n_ctx.get(w, -1), "forks": s["forks"],
                         "rejoins": s["rejoins"], "b1": s["b1"], "b0": s["b0"],
                         "max_span": max(s["loop_spans"], default=0),
                         **{f"c{l:02d}": s["clusters"][l] for l in layers},
                         **{f"h{l:02d}": s["entropy"][l] for l in layers}})
    df = pd.DataFrame(rows)
    df.to_parquet(root / f"reeb_{a.tag}.parquet", index=False)
    report(df, layers, a, rep, w_min)
    return 0


def report(df, layers, a, rep, w_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    lines = [f"# Token fork/rejoin across layers ({a.tag}): {a.unit}", "",
             f"Active neurons: z > {a.z_min} (at most {a.max_active}); cluster = correlation > tau, size >= {a.min_size};"
             f" link threshold {w_min}. Mean [95% bootstrap CI over tokens].", ""]
    for tau in a.tau:
        d = df[df.tau == tau]
        mid = [f"c{l:02d}" for l in layers if len(layers) // 3 <= layers.index(l) < 2 * len(layers) // 3]
        lines += [f"## tau = {tau}", "", "| class | tokens | clusters (middle layers) | forks | rejoins | loops (b1) | max loop span |",
                  "|---|---|---|---|---|---|---|"]
        for c in CLASSES:
            x = d[d["class"] == c]
            if not len(x):
                continue
            cm = boot(x[mid].mean(1)); f = boot(x.forks); r = boot(x.rejoins); b = boot(x.b1); sp_ = boot(x.max_span)
            fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
            lines.append(f"| {c} | {len(x)} | {fmt(cm)} | {fmt(f)} | {fmt(r)} | {fmt(b)} | {fmt(sp_)} |")
        p_, m_ = d[d["class"] == "polysemous"], d[d["class"] == "monosemous"]
        if len(p_) > 2 and len(m_) > 2:
            pv = {k: mannwhitneyu(p_[k] if k != "mid" else p_[mid].mean(1), m_[k] if k != "mid" else m_[mid].mean(1),
                                  alternative="greater").pvalue for k in ("mid", "forks", "b1")}
            lines += ["", f"Known answer, polysemous > monosemous (Mann-Whitney, one-sided): middle-layer clusters p = "
                          f"{pv['mid']:.3g}, forks p = {pv['forks']:.3g}, loops p = {pv['b1']:.3g}"]
        if (d.n_ctx > 0).any() and d.n_ctx.nunique() > 1:
            from scipy.stats import spearmanr
            r = spearmanr(d.n_ctx, d[mid].mean(1))
            lines += ["", f"Frequency check: Spearman(contexts per token, middle-layer clusters) = {r.statistic:.2f} "
                          f"(p = {r.pvalue:.2g}, {len(d)} tokens)"]
        lines.append("")
    tau = a.tau[len(a.tau) // 2]
    d = df[df.tau == tau]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = {"polysemous": "#c9772f", "monosemous": "#2a6f97", "function": "#6c757d", "ring_items": "#7b2cbf", "line_items": "#2b9348"}
    for c in CLASSES:
        x = d[d["class"] == c]
        if len(x):
            ax[0].plot(layers, [x[f"c{l:02d}"].mean() for l in layers], lw=2, color=cols[c], label=c)
            ax[1].plot(layers, [x[f"h{l:02d}"].mean() for l in layers], lw=2, color=cols[c], label=c)
    ax[0].set_ylabel("clusters per layer (mean over tokens)"); ax[1].set_ylabel("branch entropy (nats)")
    for s in ax:
        s.set_xlabel("layer"); s.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"token footprints, {a.tag}, tau = {tau}", fontsize=10)
    fig.tight_layout()
    fig.savefig(rep / f"reeb_{a.tag}.png", dpi=130)
    lines += [f"![layers](reeb_{a.tag}.png)", ""]
    runs.atomic_write_text(rep / f"reeb_{a.tag}.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
