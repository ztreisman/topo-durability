"""Persona-trait durability (test 3 of the codimension study, 2026-11-18): measures, their reliability, and
agreement across perturbations. CPU.

  uv run python scripts/trait_analysis.py --size 1.4b --workers 48
  ... --size 70m --dry                                         # on the trait_pass.py dry-run data

Per trait, depth (5 layers) and read-out (mean over the statement / last token), with statements split into the
fixed halves A and B of index.npz. Every measure on half H uses a persona vector fit on the OTHER half, so no
measure is in-sample. Persona vector = mean(matching) - mean(not matching), base model.
  retention  log(d'_variant / d'_base) along the base persona vector: how much of the trait's separation the
             variant keeps where the base model put it (primary; < 0 = damaged)
  own        the same with the variant's own vector: is the information still there, possibly rotated
  cos        cosine between the variant's and the base vector (same half)
  drift      mean shift of the variant's activations along the base vector, in within-class SD units (signed)
  dbeh       change in behavioral separation: mean log-odds(Yes - No) for matching minus not matching statements
Base descriptors: d' (strength), halfspace fit (held-out AUC of the persona vector / AUC of a shrinkage LDA:
1 = the trait is a clean half-space, a codimension-1 object), effective dimension of the within-class cloud.
Report: data/neurons/<unit>/trait_measures.parquet, reports/neurons/<unit>/traits.md/png.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker (rel-topo ran at load 310 on 64 cores)

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from topodur import runs  # noqa: E402

log = logging.getLogger("trait_analysis")
POS = ("mean", "last")
_G = {}


def dprime(p, y):
    a, b = p[y == 1], p[y == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)


def eff_dim(X, y):
    Z = np.concatenate([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    ev = np.clip(np.linalg.eigvalsh(Z @ Z.T), 0, None)
    return ev.sum() ** 2 / (ev ** 2).sum()


def _init(root, variants):
    _G["root"], _G["variants"] = root, variants
    _G["idx"] = np.load(root / "index.npz")


def one_trait(t):
    root, variants, I = _G["root"], _G["variants"], _G["idx"]
    rows = np.flatnonzero(I["trait"] == t)
    lo, hi = rows[0], rows[-1] + 1
    y, half = I["y"][lo:hi].astype(int), I["half"][lo:hi].astype(int)
    beh = {v: np.load(root / v / "beh.npy")[lo:hi] for v in ["base"] + variants}
    out = []
    for pos in POS:
        X = {v: np.load(root / v / f"{pos}.npy", mmap_mode="r")[lo:hi].astype(np.float32) for v in ["base"] + variants}
        nL = X["base"].shape[1]
        for li, H in itertools.product(range(nL), (0, 1)):
            h, o = half == H, half != H
            Xb = X["base"][:, li]
            dO = Xb[o & (y == 1)].mean(0) - Xb[o & (y == 0)].mean(0)
            uO = dO / np.linalg.norm(dO)
            dH = Xb[h & (y == 1)].mean(0) - Xb[h & (y == 0)].mean(0)
            pb = Xb[h] @ uO
            sep_b = dprime(pb, y[h])
            sd_b = np.sqrt((pb[y[h] == 1].var() + pb[y[h] == 0].var()) / 2)
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xb[o], y[o])
            auc_dm = roc_auc_score(y[h], pb)
            auc_lda = roc_auc_score(y[h], lda.decision_function(Xb[h]))
            bb = beh["base"][h]
            base = dict(trait=t, pos=pos, layer=li, half=H, d_base=sep_b, auc_base=auc_dm, auc_lda=auc_lda,
                        halfspace=(auc_dm - 0.5) / max(auc_lda - 0.5, 1e-3), eff_dim=eff_dim(Xb[h], y[h]),
                        beh_base=bb[y[h] == 1].mean() - bb[y[h] == 0].mean())
            for v in variants:
                Xv = X[v][:, li]
                dvO = Xv[o & (y == 1)].mean(0) - Xv[o & (y == 0)].mean(0)
                dvH = Xv[h & (y == 1)].mean(0) - Xv[h & (y == 0)].mean(0)
                own_b = dprime(Xb[h] @ dO, y[h])
                bv = beh[v][h]
                out.append(dict(base, variant=v,
                                retention=np.log(max(dprime(Xv[h] @ uO, y[h]), 1e-3) / max(sep_b, 1e-3)),
                                own=np.log(max(dprime(Xv[h] @ dvO, y[h]), 1e-3) / max(own_b, 1e-3)),
                                cos=float(dvH @ dH / (np.linalg.norm(dvH) * np.linalg.norm(dH))),
                                drift=float(((Xv[h] - Xb[h]) @ uO).mean() / sd_b),
                                dbeh=(bv[y[h] == 1].mean() - bv[y[h] == 0].mean()) - base["beh_base"]))
    return out


def sb(r):
    """Spearman-Brown; a negative split-half r is reported as is (no reliability to correct)."""
    return 2 * r / (1 + r) if r > 0 else r


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok])[0] if ok.sum() >= 3 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None, help="variant tags (default: every finished pass)")
    ap.add_argument("--layer", type=int, default=2, help="primary depth index (2 = middle, L/2)")
    ap.add_argument("--pos", default="mean", choices=POS, help="primary read-out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    unit = f"{a.size}_s0_step{a.step}"
    droot = runs.DATA_DIR / ("neurons_dryrun" if a.dry else "neurons") / unit
    root = droot / "traits"
    rep = Path(__file__).resolve().parents[1] / "reports" / ("neurons_dryrun" if a.dry else "neurons") / unit
    rep.mkdir(parents=True, exist_ok=True)
    runs.setup_logging()
    ok = (lambda p: (p / "beh.npy").exists()) if a.dry else (lambda p: runs.is_done(p / "beh.npy"))
    variants = a.variants or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "base" and ok(p))
    I = np.load(root / "index.npz")
    traits = list(I["traits"])
    LY = json.loads((root / "base" / "pass.json").read_text())["layers"]
    log.info("%d traits, variants %s", len(traits), variants)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(root, variants)) as ex:
        res = [r for rs in ex.map(one_trait, sorted(set(I["trait"].tolist()))) for r in rs]
    df = pd.DataFrame(res)
    df["trait_name"] = [traits[t] for t in df.trait]
    df.to_parquet(droot / "trait_measures.parquet")
    report(df, variants, LY, a, unit, rep)
    return 0


def report(df, variants, LY, a, unit, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = df[(df.layer == a.layer) & (df.pos == a.pos)]
    meas = ["retention", "own", "cos", "drift", "dbeh"]

    def halves(v, m):
        w = P[P.variant == v].pivot(index="trait", columns="half", values=m)
        return w[0].to_numpy(), w[1].to_numpy()

    B = P[P.variant == variants[0]].groupby("trait")[["d_base", "auc_base", "auc_lda", "halfspace", "eff_dim",
                                                         "beh_base"]].mean()
    L = [f"# Persona-trait durability: {unit}", "",
         f"{df.trait.nunique()} persona traits (Anthropic model-written evals), statements split into fixed halves A/B; "
         f"persona vector = mean(matching) - mean(not matching), fit on the other half. Primary: depth "
         f"{a.layer} (layer {LY[a.layer]}), read-out '{a.pos}'. Measures defined in scripts/trait_analysis.py.", "",
         "## Base model: is the trait there?", "",
         "| depth (layer) | read-out | median d' | median AUC persona vector | median AUC shrinkage LDA | median half-space fit | median eff. dim |",
         "|---|---|---|---|---|---|---|"]
    for (li, pos), g in df[df.variant == variants[0]].groupby(["layer", "pos"]):
        g = g.groupby("trait").mean(numeric_only=True)
        L.append(f"| {li} ({LY[li]}) | {pos} | {g.d_base.median():.2f} | {g.auc_base.median():.3f} | "
                 f"{g.auc_lda.median():.3f} | {g.halfspace.median():.2f} | {g.eff_dim.median():.1f} |")
    bb = B.beh_base
    L += ["", f"Behavior (base): matching minus not-matching log-odds(Yes - No), median {bb.median():+.3f}; "
          f"traits with separation > 0: {(bb > 0).mean():.0%}; > 0.5: {(bb > 0.5).mean():.0%}. "
          f"Split-half reliability of base behavior: {sb(rho(*halves(variants[0], 'beh_base'))):+.3f}.", ""]

    L += ["## Reliability (split-half across traits, Spearman-Brown)", "",
          "| variant | median retention | median drift | median dbeh | " + " | ".join(meas) + " |",
          "|---|---|---|---|" + "---|" * len(meas)]
    rel = {}
    for v in variants:
        g = P[P.variant == v]
        cells = []
        for m in meas:
            rel[v, m] = sb(rho(*halves(v, m)))
            cells.append(f"{rel[v, m]:+.2f}")
        L.append(f"| {v} | {g.retention.median():+.3f} | {g.drift.median():+.2f} | {g.dbeh.median():+.3f} | "
                 + " | ".join(cells) + " |")

    def agree(v1, v2, m):
        a1, b1 = halves(v1, m)
        a2, b2 = halves(v2, m)
        r = (rho(a1, b2) + rho(b1, a2)) / 2
        if min(rel[v1, m], rel[v2, m]) < 0.2:      # disattenuating by an unreliable measure is meaningless
            return r, np.nan
        return r, r / np.sqrt(rel[v1, m] * rel[v2, m])

    pairs = list(itertools.combinations(variants, 2))
    L += ["", "## Is durability one trait? Agreement between perturbations (disjoint halves; disattenuated)", "",
          "Concepts (reliability study): ~0.15-0.2 across perturbation types, seed replicates 0.82-0.89, noise seeds 0.07.", "",
          "| variant 1 | variant 2 | retention r | retention disatt. | dbeh r | dbeh disatt. |", "|---|---|---|---|---|---|"]
    M = np.full((len(variants), len(variants)), np.nan)
    for v1, v2 in pairs:
        r, d = agree(v1, v2, "retention")
        rb, db = agree(v1, v2, "dbeh")
        M[variants.index(v1), variants.index(v2)] = M[variants.index(v2), variants.index(v1)] = d
        L.append(f"| {v1} | {v2} | {r:+.3f} | {d:+.3f} | {rb:+.3f} | {db:+.3f} |")

    W = P.groupby(["trait", "variant"]).mean(numeric_only=True).reset_index()
    R = W.pivot(index="trait", columns="variant", values="retention")
    Z = (R - R.mean()) / R.std()
    B["disp"] = Z.std(axis=1)
    B["mean_ret"] = Z.mean(axis=1)
    L += ["", "## Geometry and damage (exploratory)", "",
          "Per variant, Spearman across traits of retention with: |drift| (does the update move the trait along its own "
          "axis?), half-space fit, effective dimension, base d'; and of retention with the behavioral change.", "",
          "| variant | vs abs(drift) | vs half-space fit | vs eff. dim | vs base d' | vs dbeh |", "|---|---|---|---|---|---|"]
    for v in variants:
        g = W[W.variant == v].set_index("trait")
        L.append(f"| {v} | {rho(g.retention, g.drift.abs()):+.3f} | {rho(g.retention, B.halfspace):+.3f} | "
                 f"{rho(g.retention, B.eff_dim):+.3f} | {rho(g.retention, B.d_base):+.3f} | {rho(g.retention, g.dbeh):+.3f} |")
    L += ["", f"Consistency: dispersion of z-retention across variants vs half-space fit {rho(B.disp, B.halfspace):+.3f}, "
          f"vs effective dimension {rho(B.disp, B.eff_dim):+.3f}; mean z-retention vs half-space fit "
          f"{rho(B.mean_ret, B.halfspace):+.3f}, vs effective dimension {rho(B.mean_ret, B.eff_dim):+.3f}.", "",
          "Most and least durable traits (mean z-retention over variants):", ""]
    nm = df.drop_duplicates("trait").set_index("trait").trait_name
    s = B.mean_ret.sort_values()
    L += ["- least: " + ", ".join(f"{nm[t]} ({s[t]:+.2f})" for t in s.index[:5]),
          "- most: " + ", ".join(f"{nm[t]} ({s[t]:+.2f})" for t in s.index[::-1][:5]), "", "![traits](traits.png)"]
    runs.atomic_write_text(rep / "traits.md", "\n".join(L) + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    prof = df[df.pos == a.pos].groupby(["variant", "layer"]).retention.median().unstack()
    for v in variants:
        ax[0].plot([LY[i] for i in prof.columns], prof.loc[v], marker="o", ms=3, lw=1.5, label=v)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("median retention, log(d' variant / d' base)")
    ax[0].legend(fontsize=6, ncol=2)
    im = ax[1].imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1].set_xticks(range(len(variants)), variants, rotation=90, fontsize=6)
    ax[1].set_yticks(range(len(variants)), variants, fontsize=6)
    ax[1].set_title("retention agreement between perturbations (disattenuated)", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    rv = [rel[v, "retention"] for v in variants]
    ax[2].barh(range(len(variants)), rv, color="#2a6f97")
    ax[2].set_yticks(range(len(variants)), variants, fontsize=6)
    ax[2].axvline(0, color="0.6", lw=0.8)
    ax[2].set_xlim(-0.2, 1)
    ax[2].set_xlabel("split-half reliability of retention (SB)")
    for x in ax[[0, 2]]:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rep / "traits.png", dpi=130)
    log.info("report: %s", rep / "traits.md")


if __name__ == "__main__":
    sys.exit(main())
