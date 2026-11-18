"""Does a token's sense topology predict its durability? CPU.

  uv run python scripts/token_durability_analysis.py --unit 1.4b_s0_step143000 --senses eq224
  ... --limit --senses all                     # dry-run data

Topology per token (from scripts/token_senses.py labels, base model):
  senses      mean number of context clusters over the middle third of the layers
  bars, mean_len, max_len, total_pers   barcode of sense clusters across layers (topodur.reeb.sense_barcode:
              a cluster persists while consecutive layers share >= half their union; the whole-token cluster is
              not a bar)
  b1          rank of H1 of the token's Reeb graph with strict links (a cluster at the next layer at least
              70% contained in one here): independent fork-then-rejoin loops
Each is turned into an EXCESS: the residual after regressing on log frequency and token class, i.e. how much more
the token splits / persists than comparable tokens (the 2026-11-17 hypothesis: s >> k or long persistence =>
more robustly embedded).
Durability per token: mean change in log p of the token in its own contexts, variant minus base (closer to 0 =
more durable). Primary = mean over the two 20M-token LoRA fine-tunes (code, German Wikipedia); every variant is
also reported. Secondary, representational: does the base sense partition still separate the contexts in the
variant's activations (eta^2 ratio, middle layers), and linear CKA base vs variant.
Test: durability ~ excess topology + log frequency + base log p + class (OLS; 95% bootstrap CIs over tokens).
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
from topodur.reeb import context_links, reeb_stats, sense_barcode  # noqa: E402

log = logging.getLogger("token_durability_analysis")
TOPO = ["senses", "total_pers", "mean_len", "max_len", "b1"]
PRIMARY = ("lora-code-20M", "lora-wiki_de-20M")


def topology(lab: np.ndarray, layers: list[int], mid: list[int], w_strict: float) -> dict:
    groups = {l: [np.flatnonzero(lab[i] == c) for c in np.unique(lab[i]) if c >= 0] for i, l in enumerate(layers)}
    links = {}
    for l in layers[:-1]:
        for (i, j), w in context_links(groups[l], groups[l + 1]).items():
            links[(l, i, j)] = w
    s = reeb_stats(layers, groups, links, w_strict, {l: [len(g) for g in groups[l]] for l in layers})
    bars = sense_barcode(lab)
    ln = np.array([b[1] for b in bars]) if bars else np.zeros(0)
    return {"senses": float(np.mean([len(groups[l]) for l in mid])), "bars": len(bars),
            "mean_len": float(ln.mean()) if len(ln) else 0.0, "max_len": int(ln.max()) if len(ln) else 0,
            "total_pers": int(ln.sum()), "b1": s["b1"], "bar_lengths": ln.tolist()}


def eta2(X: np.ndarray, lab: np.ndarray) -> float:
    """Share of the contexts' variance explained by the sense partition."""
    keep = lab >= 0
    X, lab = X[keep] - X[keep].mean(0), lab[keep]
    tot = float((X ** 2).sum())
    if tot <= 0 or len(np.unique(lab)) < 2:
        return np.nan
    between = sum(float(((X[lab == c].mean(0)) ** 2).sum()) * (lab == c).sum() for c in np.unique(lab))
    return between / tot


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    X, Y = X - X.mean(0), Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


def residual(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(y)), Z])
    b, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ b


def ols_boot(y, X, names, n_boot=2000, seed=0):
    """Standardised OLS coefficients with percentile bootstrap CIs (resampling tokens)."""
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    ys = (y - y.mean()) / y.std()
    X1 = np.column_stack([np.ones(len(y)), Xs])
    fit = lambda idx: np.linalg.lstsq(X1[idx], ys[idx], rcond=None)[0][1:]
    b = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    B = np.array([fit(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
    p = np.minimum(1, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)))
    return pd.DataFrame({"term": names, "beta": b, "lo": lo, "hi": hi, "p_boot": p})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--senses", default="eq224", help="token_senses.py output tag")
    ap.add_argument("--w-strict", type=float, default=0.7)
    ap.add_argument("--fp-dir", default="footprint", help="footprint (token panel) or cfoot (concept panel)")
    ap.add_argument("--limit", action="store_true")
    a = ap.parse_args()
    a.sfx = "" if a.fp_dir == "footprint" else f"_{a.fp_dir}"
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    runs.setup_logging(root / f"durab_analysis{a.sfx}.log")
    S = pd.read_parquet(root / f"senses_{a.senses}.parquet")
    LB = np.load(root / f"senses_{a.senses}_labels.npz")
    layers = LB["layers"].tolist()
    mid = [l for l in layers if len(layers) // 3 <= layers.index(l) < 2 * len(layers) // 3]
    fpdir = root / a.fp_dir
    fmeta = json.loads((fpdir / "footprint.json").read_text())
    if a.fp_dir == "footprint":                       # single tokens: occurrences in the neuron_stats stream
        tc = np.load(root / "tokcounts.npz")
        cnt = dict(zip(tc["ids"].tolist(), tc["cnt"].tolist()))
    else:                                             # concepts: counts from the panel builder
        cnt = {int(t): v["count"] for t, v in fmeta["tokens"].items()}
    ddir = root / ("durab" if a.fp_dir == "footprint" else f"durab_{a.fp_dir}")
    vdirs = sorted(p for p in ddir.iterdir() if runs.is_done(p / "logp.npz") and not p.name.endswith("_dry"))
    lp = {p.name: np.load(p / "logp.npz") for p in vdirs}
    if "base" not in lp:
        log.error("no base durability pass")
        return 1
    variants = [v for v in lp if v != "base"]
    primary = [v for v in variants if v in PRIMARY] or variants
    log.info("variants %s; primary = mean of %s", variants, primary)
    lay_index = fmeta["layers"]
    rows = []
    for r in S.itertuples():
        key = f"T{r.token_id}"
        if key not in LB.files or key not in lp["base"].files:
            continue
        lab = LB[key]
        n = lab.shape[1]
        base = lp["base"][key][:n]
        row = {"class": r._1, "token": r.token, "token_id": r.token_id, "log_freq": np.log1p(cnt.get(r.token_id, 0)),
               "base_logp": float(base.mean()), **topology(lab, layers, mid, a.w_strict)}
        tm = fmeta["tokens"].get(str(r.token_id), {})
        if "synsets" in tm:
            row["log_synsets"] = float(np.log(tm["synsets"]))
            row["n_tok"] = tm["n_tok"]
        for v in variants:
            row[f"d_{v}"] = float((lp[v][key][:n] - base).mean())
        row["durability"] = float(np.mean([row[f"d_{v}"] for v in primary]))
        # representational retention, where the variant's activations were saved
        F = None
        for v in variants:
            af = ddir / v / f"A_{key}.npy"
            if not af.exists():
                continue
            if F is None:
                F = np.load(fpdir / f"{key}.npy", mmap_mode="r")[:n].astype(np.float32)
            V = np.load(af, mmap_mode="r")[:n].astype(np.float32)
            e, c = [], []
            for l in mid:
                i, j = layers.index(l), lay_index.index(l)
                eb = eta2(F[:, j], lab[i])
                if np.isfinite(eb) and eb > 0:
                    e.append(eta2(V[:, j], lab[i]) / eb)
                c.append(cka(F[:, j], V[:, j]))
            row[f"sep_{v}"] = float(np.nanmean(e)) if e else np.nan
            row[f"cka_{v}"] = float(np.mean(c))
        rows.append(row)
    df = pd.DataFrame(rows)
    classes = sorted(df["class"].unique())
    D = np.column_stack([(df["class"] == c).astype(float) for c in classes[1:]] + [df.log_freq])
    for t in TOPO:
        df[f"x_{t}"] = residual(df[t].to_numpy(float), D)
    df.drop(columns="bar_lengths").to_parquet(root / f"durab_tokens{a.sfx}.parquet", index=False)
    report(df, variants, primary, classes, rep, a)
    return 0


def report(df, variants, primary, classes, rep, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    cnames = ["log_freq", "base_logp"] + (["log_synsets"] if "log_synsets" in df else []) + [f"class {c}" for c in classes[1:]]
    cov = np.column_stack([df.log_freq, df.base_logp] + ([df.log_synsets] if "log_synsets" in df else [])
                          + [(df["class"] == c).astype(float) for c in classes[1:]])
    lines = [f"# Sense topology vs durability: {a.unit} (senses {a.senses})", "",
             f"{len(df)} tokens. Durability = mean change in log p of the token in its own contexts, variant minus base "
             f"(primary: mean of {', '.join(primary)}). Excess topology = residual after log frequency + class.", "",
             "## Durability by variant (mean [5%, 95%] over tokens)", "", "| variant | mean d log p | 5% | 95% |", "|---|---|---|---|"]
    for v in variants:
        x = df[f"d_{v}"]
        lines.append(f"| {v} | {x.mean():.3f} | {x.quantile(.05):.3f} | {x.quantile(.95):.3f} |")
    lines += ["", "## Topology by class (means)", "",
              "| class | tokens | senses (middle) | bars | mean bar length | max bar | total persistence | b1 (strict) |",
              "|---|---|---|---|---|---|---|---|"]
    for c in classes:
        x = df[df["class"] == c]
        lines.append(f"| {c} | {len(x)} | {x.senses.mean():.2f} | {x.bars.mean():.1f} | {x.mean_len.mean():.2f} | "
                     f"{x.max_len.mean():.1f} | {x.total_pers.mean():.1f} | {x.b1.mean():.2f} |")
    lines += ["", "## Each excess measure alone (partial Spearman with durability, controlling log frequency, base log p, "
              "class; and standardised OLS beta with 95% bootstrap CI)", "",
              "| measure | target | Spearman | p | beta [95% CI] |", "|---|---|---|---|---|"]
    targets = ["durability"] + [f"d_{v}" for v in variants if f"d_{v}" != "durability"]
    yres = {t: residual(df[t].to_numpy(float), cov) for t in targets}
    for t in TOPO:
        for tg in targets[:1] + [x for x in targets[1:] if x.replace("d_", "") not in primary or len(primary) == 1]:
            r = spearmanr(df[f"x_{t}"], yres[tg])
            ob = ols_boot(df[tg].to_numpy(float), np.column_stack([df[f"x_{t}"], cov]), [t] + ["cov"] * cov.shape[1]).iloc[0]
            lines.append(f"| {t} | {tg} | {r.statistic:+.3f} | {r.pvalue:.3g} | {ob.beta:+.3f} [{ob.lo:+.3f}, {ob.hi:+.3f}] |")
    joint = ["senses", "total_pers", "b1"]
    ob = ols_boot(df.durability.to_numpy(float), np.column_stack([df[f"x_{t}"] for t in joint] + [cov]),
                  [f"excess {t}" for t in joint] + cnames)
    lines += ["", "## Joint model: durability ~ excess senses + excess persistence + excess b1 + covariates", "",
              "| term | beta | 95% CI | p (bootstrap) |", "|---|---|---|---|"]
    lines += [f"| {r.term} | {r.beta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_boot:.3g} |" for r in ob.itertuples()]
    if a.fp_dir != "footprint":
        b = ob[ob.term == "excess b1"].iloc[0]
        ok = b.lo > 0
        lines[2:2] = ["## Confirmatory test (specified 2026-11-17, before these data; exploratory panel: beta +0.19 [0.06, 0.33])", "",
                      "Primary: standardised coefficient of excess b1 (strict links 0.7) in the joint model; durability = "
                      "mean d log p over the two 20M LoRA fine-tunes. Confirmed if the 95% bootstrap CI lies above 0.", "",
                      f"**Result: beta = {b.beta:+.3f} [{b.lo:+.3f}, {b.hi:+.3f}], p = {b.p_boot:.3g}: "
                      f"{'CONFIRMED' if ok else 'NOT confirmed'}**", ""]
    sv = [v for v in variants if f"sep_{v}" in df]
    if sv:
        lines += ["", "## Representational retention (middle layers; mean over tokens)", "",
                  "| variant | sense separation kept (eta^2 ratio) | CKA base vs variant | Spearman(excess persistence, separation kept) |",
                  "|---|---|---|---|"]
        for v in sv:
            ok = df[f"sep_{v}"].notna()
            rs = "n/a (< 3 tokens with a split)"
            if ok.sum() >= 3:
                r = spearmanr(df.x_total_pers[ok], df[f"sep_{v}"][ok])
                rs = f"{r.statistic:+.3f} (p {r.pvalue:.2g}, {ok.sum()} tokens)"
            lines.append(f"| {v} | {df[f'sep_{v}'].mean():.3f} | {df[f'cka_{v}'].mean():.3f} | {rs} |")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = {"polysemous": "#c9772f", "monosemous": "#2a6f97", "function": "#6c757d", "ring_items": "#7b2cbf", "line_items": "#2b9348"}
    yr = yres["durability"]
    for k, t in enumerate(["senses", "total_pers", "b1"]):
        for c in classes:
            m = (df["class"] == c).to_numpy()
            ax[k].scatter(df[f"x_{t}"][m], yr[m], s=12, color=cols.get(c, "#999"), label=c, alpha=.8)
        ax[k].set_xlabel(f"excess {t}"); ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("durability (residual d log p)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle("sense topology vs durability (each point a token; covariates removed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(rep / f"durab_topology{a.sfx}.png", dpi=130)
    lines += ["", f"![scatter](durab_topology{a.sfx}.png)", ""]
    runs.atomic_write_text(rep / f"durab_topology{a.sfx}.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
