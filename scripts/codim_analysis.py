"""Codimension study (2026-11-18): is concept damage explained by the update's direction (test 1), and are concepts
spread over many directions (low codimension) damaged consistently across perturbations (test 2)? CPU.

  uv run python scripts/codim_analysis.py --unit 1.4b_s0_step143000 --workers 32
  ... --limit                                  # dry-run data

Inputs: measured damage (durab_cfoot/<variant>/logp.npz vs base), the first-order pass (codim/pass.npz), the base
footprints (cfoot/T<i>.npy, activations at the concept's last token).
Test 1, per perturbation: Spearman across concepts of measured d log p with (a) the predicted direct-path d log p and
(b) the overlap ||dW a|| / ||W a|| at the predicting position q (mean over layers); median within-concept Spearman
across contexts; and whether the predicted damages reproduce the low agreement between perturbations.
Test 2: effective dimension of each concept (participation ratio of the within-concept covariance, middle layers;
high = spread over many directions = low codimension) and the spread of its mean-shift signature over neurons.
Predictions: higher effective dimension -> (a) its overlap with different updates varies less (dispersion of the
z-scored overlap across perturbations), (b) its measured damage varies less across perturbations, (c) within the top
tercile of effective dimension, damage agrees more across perturbations (disjoint context halves) than in the bottom.
Outputs: data/neurons/<unit>/codim_concepts.parquet; reports .../codim.md / .png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402

log = logging.getLogger("codim_analysis")
_G = {}


def spear(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 3 else np.nan


def _init(mu, sd):
    _G["mu"], _G["sd"] = mu, sd


def eff_dim(path: str, mid_idx: list[tuple[int, int]], n_ctx: int) -> dict:
    """Participation ratio of the within-concept covariance and of the mean-shift signature, middle layers."""
    X = np.load(path, mmap_mode="r")[:n_ctx]
    pr, sig = [], []
    for l, j in mid_idx:
        Z = (X[:, j].astype(np.float32) - _G["mu"][l]) / _G["sd"][l]
        m = Z.mean(0)
        Zc = Z - m
        ev = np.clip(np.linalg.eigvalsh((Zc @ Zc.T).astype(np.float64)), 0, None)
        pr.append(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
        s2 = m.astype(np.float64) ** 2
        sig.append(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-12))
    return {"eff_dim": float(np.mean(pr)), "sig_spread": float(np.mean(sig))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--fp-dir", default="cfoot")
    ap.add_argument("--n-ctx", type=int, default=224)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", action="store_true")
    a = ap.parse_args()
    tag = "neurons_dryrun" if a.limit else "neurons"
    root = runs.DATA_DIR / tag / a.unit
    rep = runs.REPO_ROOT / "reports" / tag / a.unit
    rep.mkdir(parents=True, exist_ok=True)
    cd = root / "codim"
    runs.setup_logging(cd / "analysis.log")
    fp = root / a.fp_dir
    meta = json.loads((fp / "footprint.json").read_text())
    pj = json.loads((cd / "pass.json").read_text())
    tags = pj["tags"]
    PZ = np.load(cd / "pass.npz")
    ddir = root / f"durab_{a.fp_dir}"
    LP = {t: np.load(ddir / t / "logp.npz") for t in ["base"] + tags if (ddir / t / "logp.npz").exists()}
    tags = [t for t in tags if t in LP]
    vi = {t: pj["tags"].index(t) for t in tags}
    n = a.n_ctx
    perm = np.random.default_rng(a.seed).permutation(n)
    HA, HB = np.sort(perm[:n // 2]), np.sort(perm[n // 2:])
    rows, within = [], {t: [] for t in tags}
    for t, v in meta["tokens"].items():
        if f"R{t}" not in PZ.files or v["n"] < n or f"T{t}" not in LP["base"].files:
            continue
        R, D = PZ[f"R{t}"][:n], PZ[f"D{t}"][:n]                  # [n, V, 2, L], [n, V]
        base = LP["base"][f"T{t}"][:n]
        row = {"token_id": int(t), "word": v["word"], "class": v["class"], "n_tok": v.get("n_tok", 1),
               "log_freq": float(np.log(v.get("count", 1)))}
        for tg in tags:
            d = LP[tg][f"T{t}"][:n] - base
            k = vi[tg]
            row[f"meas_{tg}"], row[f"meas_{tg}_A"], row[f"meas_{tg}_B"] = float(d.mean()), float(d[HA].mean()), float(d[HB].mean())
            row[f"pred_{tg}"] = float(D[:, k].mean())
            row[f"relq_{tg}"] = float(R[:, k, 0].mean())
            row[f"relp_{tg}"] = float(R[:, k, 1].mean())
            within[tg].append(spear(d, D[:, k]))
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info("%d concepts, perturbations %s", len(df), tags)
    # effective dimension from the base footprints
    layers = [l for l in meta["layers"] if (root / f"L{l:02d}.npz").exists()]
    mid = [l for l in layers if len(layers) // 3 <= layers.index(l) < 2 * len(layers) // 3]
    mu, sd = {}, {}
    for l in mid:
        z = np.load(root / f"L{l:02d}.npz")
        s = z["sd"].astype(np.float32)
        mu[l], sd[l] = z["mean"].astype(np.float32), np.maximum(s, np.quantile(s, 0.1))
    mid_idx = [(l, meta["layers"].index(l)) for l in mid]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(mu, sd)) as ex:
        futs = [ex.submit(eff_dim, str(fp / f"T{t}.npy"), mid_idx, n) for t in df.token_id]
        E = pd.DataFrame([f.result() for f in futs])
    df = pd.concat([df.reset_index(drop=True), E], axis=1)
    df.to_parquet(root / "codim_concepts.parquet", index=False)
    report(df, tags, within, rep, a)
    return 0


def zcol(s):
    return (s - s.mean()) / s.std()


def residual(y, Z):
    Z1 = np.column_stack([np.ones(len(y)), Z])
    return y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]


def report(df, tags, within, rep, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = [f"# Codimension study: {a.unit}", "",
         f"{len(df)} concepts x {a.n_ctx} contexts. Damage = change in log p of the concept in its own contexts. "
         "Predicted = first-order direct-path change through the MLP-output update alone (scripts/codim_pass.py).", "",
         "## Test 1: is damage explained by the update's direction?", "",
         "| perturbation | measured vs predicted d log p | measured vs overlap at q | measured vs overlap at p | within-concept (median over concepts) |",
         "|---|---|---|---|---|"]
    for t in tags:
        L.append(f"| {t} | {spear(df[f'meas_{t}'], df[f'pred_{t}']):+.3f} | {spear(df[f'meas_{t}'], -df[f'relq_{t}']):+.3f} | "
                 f"{spear(df[f'meas_{t}'], -df[f'relp_{t}']):+.3f} | {np.nanmedian(within[t]):+.3f} |")
    L += ["", "Overlap is entered with a minus sign (more overlap -> more damage expected -> positive correlation).", "",
          "Agreement between perturbations, measured (disjoint context halves) vs predicted:", "",
          "| pair | measured | predicted d log p | overlap at q |", "|---|---|---|---|"]
    for t1, t2 in combinations(tags, 2):
        m = 0.5 * (spear(df[f"meas_{t1}_A"], df[f"meas_{t2}_B"]) + spear(df[f"meas_{t1}_B"], df[f"meas_{t2}_A"]))
        L.append(f"| {t1} / {t2} | {m:+.3f} | {spear(df[f'pred_{t1}'], df[f'pred_{t2}']):+.3f} | "
                 f"{spear(df[f'relq_{t1}'], df[f'relq_{t2}']):+.3f} |")
    # test 2
    cov = np.column_stack([df.log_freq] + [(df["class"] == c).astype(float) for c in sorted(df["class"].unique())[1:]])
    df["eff_dim_x"] = residual(df.eff_dim.to_numpy(float), cov)
    df["overlap_disp"] = np.std(np.column_stack([zcol(df[f"relq_{t}"]) for t in tags]), 1)
    df["damage_disp"] = np.std(np.column_stack([zcol(df[f"meas_{t}"]) for t in tags]), 1)
    df["damage_mean"] = np.mean(np.column_stack([zcol(df[f"meas_{t}"]) for t in tags]), 1)
    L += ["", "## Test 2: are low-codimension (high effective dimension) concepts damaged consistently?", "",
          f"Effective dimension (middle layers): median {df.eff_dim.median():.1f} [5%: {df.eff_dim.quantile(.05):.1f}, "
          f"95%: {df.eff_dim.quantile(.95):.1f}] of {a.n_ctx - 1} possible; mean-shift signature spread median "
          f"{df.sig_spread.median():.0f} neurons.", "",
          "| prediction | Spearman with effective dimension | ... with excess effective dimension (after log freq + n tokens) | with signature spread |",
          "|---|---|---|---|"]
    for name, col in (("(a) overlap dispersion across perturbations (expect < 0)", "overlap_disp"),
                      ("(b) damage dispersion across perturbations (expect < 0)", "damage_disp"),
                      ("mean damage (z, higher = less damaged)", "damage_mean")):
        L.append(f"| {name} | {spear(df.eff_dim, df[col]):+.3f} | {spear(df.eff_dim_x, df[col]):+.3f} | {spear(df.sig_spread, df[col]):+.3f} |")
    L += ["", "(c) Agreement between perturbations (disjoint halves, mean over pairs) by tercile of excess effective dimension "
              "(expect: rises with dimension):", "", "| tercile | concepts | mean agreement | " +
          " | ".join(f"{t1} / {t2}" for t1, t2 in combinations(tags, 2)) + " |",
          "|---|---|---|" + "---|" * len(list(combinations(tags, 2)))]
    df["terc"] = pd.qcut(df.eff_dim_x, 3, labels=["low dim (high codim)", "middle", "high dim (low codim)"])
    agree = {}
    for tc, d in df.groupby("terc", observed=True):
        rs = [0.5 * (spear(d[f"meas_{t1}_A"], d[f"meas_{t2}_B"]) + spear(d[f"meas_{t1}_B"], d[f"meas_{t2}_A"]))
              for t1, t2 in combinations(tags, 2)]
        agree[tc] = np.mean(rs)
        L.append(f"| {tc} | {len(d)} | {np.mean(rs):+.3f} | " + " | ".join(f"{r:+.3f}" for r in rs) + " |")
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(500):
        b = df.iloc[rng.integers(0, len(df), len(df))]
        g = {tc: np.mean([0.5 * (spear(d[f"meas_{t1}_A"], d[f"meas_{t2}_B"]) + spear(d[f"meas_{t1}_B"], d[f"meas_{t2}_A"]))
                          for t1, t2 in combinations(tags, 2)]) for tc, d in b.groupby("terc", observed=True)}
        diffs.append(g.get("high dim (low codim)", np.nan) - g.get("low dim (high codim)", np.nan))
    lo, hi = np.nanquantile(diffs, [0.025, 0.975])
    L += ["", f"High minus low tercile: {agree.get('high dim (low codim)', np.nan) - agree.get('low dim (high codim)', np.nan):+.3f} "
              f"[95% bootstrap {lo:+.3f}, {hi:+.3f}]", ""]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    t0 = tags[0]
    ax[0].scatter(df[f"pred_{t0}"], df[f"meas_{t0}"], s=6, alpha=.5, color="#2a6f97")
    ax[0].set_xlabel(f"predicted d log p ({t0})"); ax[0].set_ylabel("measured d log p")
    ax[1].scatter(df.eff_dim, df.damage_disp, s=6, alpha=.5, color="#c9772f")
    ax[1].set_xlabel("effective dimension (middle layers)"); ax[1].set_ylabel("damage dispersion across perturbations")
    ks = list(agree)
    ax[2].bar(range(len(ks)), [agree[k] for k in ks], color=["#c9772f", "#999", "#2b9348"][:len(ks)])
    ax[2].set_xticks(range(len(ks)), [str(k).replace(" (", "\n(") for k in ks], fontsize=8)
    ax[2].set_ylabel("agreement between perturbations")
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rep / "codim.png", dpi=130)
    L += ["![codim](codim.png)", ""]
    runs.atomic_write_text(rep / "codim.md", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
