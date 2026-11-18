"""Senses of a token: clusters of its CONTEXTS at each layer, tracked across layers (fork/rejoin study, part C).
CPU, reads the token_footprint.py data.

  uv run python scripts/token_senses.py --unit 1.4b_s0_step143000 --n-ctx-eq 224
  ... --limit                                                     # data/neurons_dryrun

Per token and layer: the contexts' MLP activations (standardised by the layer's global mean / sd) are projected
on their top --dims principal components and split by k-means, with k chosen by the gap statistic against one
Gaussian blob of the same spread (src/topodur/reeb.py gap_k; k = 1 unless the contexts really separate).
Context clusters at adjacent layers are linked by overlap (|A & B| / min(|A|, |B|) >= --w-min); the result is the
token's Reeb graph over its contexts: a fork = one group of contexts separating into senses, a rejoin = senses
merging again, b1 = independent fork-then-rejoin loops (layer H0 becoming concept H1).
Unlike token_reeb.py (clusters of neurons), every context sits in exactly one cluster per layer, so the
partitions are directly comparable across layers (adjusted Rand index of adjacent layers is reported too).
Known answer: polysemous tokens should have more senses than monosemous ones.
Outputs: data/neurons/<unit>/senses_<tag>.parquet, senses_<tag>_labels.npz; reports senses_<tag>.md / .png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.reeb import context_links, gap_k, panel, pcs, reeb_stats  # noqa: E402

log = logging.getLogger("token_senses")
CLASSES = ["polysemous", "monosemous", "function", "ring_items", "line_items"]
_G = {}


def _init(mu, sd):
    _G["mu"], _G["sd"] = mu, sd


def one_token(path: str, layers: list[int], lay_index: list[int], a_dict: dict):
    from sklearn.metrics import adjusted_rand_score

    X = np.load(path, mmap_mode="r")
    if a_dict["n_ctx_eq"]:
        X = X[:a_dict["n_ctx_eq"]]
    n = len(X)
    groups, labels = {}, {}
    for l in layers:
        Z = (X[:, lay_index.index(l)].astype(np.float32) - _G["mu"][l]) / _G["sd"][l]
        k, g = gap_k(pcs(Z, a_dict["dims"]), kmax=a_dict["kmax"], n_ref=a_dict["n_ref"], min_size=a_dict["min_size"],
                     ref_inits=a_dict["n_ref_inits"])
        groups[l] = g
        lab = np.full(n, -1, np.int8)
        for c, idx in enumerate(g):
            lab[idx] = c
        labels[l] = lab
    links = {}
    for l in layers[:-1]:
        for (i, j), w in context_links(groups[l], groups[l + 1]).items():
            links[(l, i, j)] = w
    s = reeb_stats(layers, groups, links, a_dict["w_min"], {l: [len(x) for x in groups[l]] for l in layers})
    ari = [adjusted_rand_score(labels[l], labels[l + 1]) for l in layers[:-1]
           if len(groups[l]) > 1 or len(groups[l + 1]) > 1]
    return {"n_ctx": n, "forks": s["forks"], "rejoins": s["rejoins"], "b1": s["b1"], "b0": s["b0"],
            "max_span": max(s["loop_spans"], default=0), "ari_adj": float(np.mean(ari)) if ari else np.nan,
            **{f"k{l:02d}": s["clusters"][l] for l in layers}}, np.stack([labels[l] for l in layers])


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
    ap.add_argument("--panel", default=str(runs.REPO_ROOT / "configs" / "token_panel.yaml"))
    ap.add_argument("--dims", type=int, default=10, help="principal components per layer")
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--n-ref", type=int, default=10, help="Gaussian reference draws for the gap statistic")
    ap.add_argument("--min-size", type=int, default=8, help="smallest context cluster kept")
    ap.add_argument("--w-min", type=float, default=0.3, help="overlap needed to link clusters of adjacent layers")
    ap.add_argument("--n-ctx-eq", type=int, default=0, help="use exactly this many contexts per token (drop tokens with fewer)")
    ap.add_argument("--fp-dir", default="footprint", help="footprint (token panel) or cfoot (concept panel)")
    ap.add_argument("--n-ref-inits", type=int, default=3, help="k-means restarts on the reference draws (3 = as in the exploratory run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", action="store_true")
    a = ap.parse_args()
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    a.tag = ("" if a.fp_dir == "footprint" else f"{a.fp_dir}_") + ("eq%d" % a.n_ctx_eq if a.n_ctx_eq else "all")
    runs.setup_logging(root / f"senses_{a.tag}.log")
    if a.limit:
        a.min_size = 2
    fp = root / a.fp_dir
    meta = json.loads((fp / "footprint.json").read_text())
    lay_index = meta["layers"]
    layers = [l for l in lay_index if (root / f"L{l:02d}.npz").exists()]
    mu, sd = {}, {}
    for l in layers:
        z = np.load(root / f"L{l:02d}.npz")
        s = z["sd"].astype(np.float32)
        mu[l], sd[l] = z["mean"].astype(np.float32), np.maximum(s, np.quantile(s, 0.1))   # no near-dead blow-up
    toks = [(v["class"], v["word"], int(t)) for t, v in meta["tokens"].items()
            if (fp / f"T{t}.npy").exists() and v["n"] >= max(a.n_ctx_eq, 2 * a.min_size)]
    log.info("%d tokens, layers %s", len(toks), layers)
    ad = {k: getattr(a, k) for k in ("n_ctx_eq", "dims", "kmax", "n_ref", "min_size", "w_min", "n_ref_inits")}
    rows, labs = [], {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(mu, sd)) as ex:
        futs = {ex.submit(one_token, str(fp / f"T{t}.npy"), layers, lay_index, ad): (c, w, t) for c, w, t in toks}
        for i, (f, (c, w, t)) in enumerate(futs.items()):
            r, lab = f.result()
            rows.append({"class": c, "token": w, "token_id": t, **r})
            labs[f"T{t}"] = lab
            if (i + 1) % 20 == 0:
                log.info("%d / %d tokens", i + 1, len(toks))
    df = pd.DataFrame(rows)
    df.to_parquet(root / f"senses_{a.tag}.parquet", index=False)
    np.savez_compressed(root / f"senses_{a.tag}_labels.npz", layers=np.array(layers), **labs)
    report(df, layers, a, rep)
    return 0


def report(df, layers, a, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, spearmanr

    mid = [f"k{l:02d}" for l in layers if len(layers) // 3 <= layers.index(l) < 2 * len(layers) // 3]
    df = df.assign(senses_mid=df[mid].mean(axis=1), split_share=(df[[f"k{l:02d}" for l in layers]] > 1).mean(axis=1))
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    lines = [f"# Token senses: context clusters across layers ({a.tag}): {a.unit}", "",
             f"Per layer: k-means on the top {a.dims} PCs of the token's contexts, k by gap statistic vs one Gaussian blob "
             f"(k <= {a.kmax}, clusters >= {a.min_size} contexts); clusters at adjacent layers linked when overlap >= {a.w_min}.",
             "Mean [95% bootstrap CI over tokens]. Middle layers = the middle third.", "",
             "| class | tokens | senses (middle layers) | layers split | forks | rejoins | loops (b1) | max loop span | ARI adjacent |",
             "|---|---|---|---|---|---|---|---|---|"]
    present = [c for c in CLASSES if c in set(df["class"])] + sorted(set(df["class"]) - set(CLASSES))
    for c in present:
        x = df[df["class"] == c]
        if len(x):
            lines.append(f"| {c} | {len(x)} | {fmt(boot(x.senses_mid))} | {fmt(boot(x.split_share))} | {fmt(boot(x.forks))} | "
                         f"{fmt(boot(x.rejoins))} | {fmt(boot(x.b1))} | {fmt(boot(x.max_span))} | {fmt(boot(x.ari_adj))} |")
    p_, m_ = df[df["class"] == "polysemous"], df[df["class"] == "monosemous"]
    if len(p_) > 2 and len(m_) > 2:
        pv = {k: mannwhitneyu(p_[k], m_[k], alternative="greater").pvalue for k in ("senses_mid", "split_share", "forks", "b1")}
        lines += ["", "Known answer, polysemous > monosemous (Mann-Whitney, one-sided): " +
                  ", ".join(f"{k} p = {v:.3g}" for k, v in pv.items())]
    if df.n_ctx.nunique() > 1:
        r = spearmanr(df.n_ctx, df.senses_mid)
        lines += ["", f"Frequency check: Spearman(contexts per token, middle-layer senses) = {r.statistic:.2f} (p = {r.pvalue:.2g})"]
    top = df.sort_values(["senses_mid", "b1"], ascending=False).head(15)
    lines += ["", "Most split tokens:", "", "| token | class | senses (middle) | forks | loops |", "|---|---|---|---|---|"]
    lines += [f"| {r.token} | {getattr(r, "_1")} | {r.senses_mid:.2f} | {r.forks} | {r.b1} |" for r in top.itertuples()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = {"polysemous": "#c9772f", "monosemous": "#2a6f97", "function": "#6c757d", "ring_items": "#7b2cbf", "line_items": "#2b9348"}
    for c in present:
        x = df[df["class"] == c]
        if len(x):
            ax[0].plot(layers, [x[f"k{l:02d}"].mean() for l in layers], lw=2, color=cols.get(c), label=c)
            ax[1].plot(layers, [(x[f"k{l:02d}"] > 1).mean() for l in layers], lw=2, color=cols.get(c), label=c)
    ax[0].set_ylabel("senses per layer (mean over tokens)"); ax[1].set_ylabel("share of tokens split")
    for s in ax:
        s.set_xlabel("layer"); s.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"token senses (context clusters), {a.tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(rep / f"senses_{a.tag}.png", dpi=130)
    lines += ["", f"![layers](senses_{a.tag}.png)", ""]
    runs.atomic_write_text(rep / f"senses_{a.tag}.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
