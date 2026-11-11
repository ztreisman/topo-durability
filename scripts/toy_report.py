"""Phase 4 report: aggregate the toy-model sweep into reports/phase4_toy/summary.md + figures.

  uv run python scripts/toy_report.py --config configs/phase4_toy.yaml

Primary comparisons use the MIXED models (ring, path and cliques trained together in one model, so
capacity pressure is shared): paired differences per seed, ring - path and ring - clique, with 95%
bootstrap CIs over (cell, seed) pairs. Single-structure models give the "how often does geometry form"
and emergence-time results.

Durability metrics (lower = more durable): relative loss increase under weight noise (0.1, 0.3 x rms),
pruning one hidden dimension, and retraining on independent features; Procrustes distance of the
structure's feature directions after retraining. Cost: interference and dims_used (sum of Elhage
dimensionality). "Per unit cost": loss increase divided by dims_used share.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import report as rep  # noqa: E402
from topodur import runs  # noqa: E402

DUR = ["dloss_noise0.1", "dloss_noise0.3", "dloss_prune1", "dloss_retrain", "procrustes_retrain"]
COST = ["interference", "dims_used"]


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run else "")
    cells = runs.run_dir(exp, "cells")
    df = pd.concat([pd.read_parquet(p) for p in sorted(cells.glob("*.parquet"))], ignore_index=True)
    out = runs.REPO_ROOT / "reports" / exp
    out.mkdir(parents=True, exist_ok=True)
    df["closed_ring"] = (df.order_score >= 0.8) & (df.closure_ratio < 2)
    df["dloss_per_dim"] = df["dloss_noise0.1"] / df.dims_used.clip(lower=1e-6)
    lines = [f"# {exp}", "", f"{df.cell.nunique()} cells, {len(df)} rows (seed x substructure).", ""]

    # ---- mixed models: paired within-model differences
    mx = df[df.structure == "mixed"]
    piv = mx.pivot_table(index=["cell", "seed", "m", "n", "q"], columns="sub", dropna=False,
                         values=DUR + COST + ["llc_refined", "dloss_per_dim", "order_score", "closure_ratio"])
    lines += ["## Mixed models: ring vs path vs clique in the same model", "",
              "Mean paired difference over (cell, seed) with 95% bootstrap CI. For durability metrics, "
              "negative ring - X means the ring is MORE durable.", "",
              "| metric | ring | path | clique | ring - path | ring - clique |", "|---|---|---|---|---|---|"]
    res = {}
    for met in DUR + COST + ["dloss_per_dim", "llc_refined"]:
        if met not in piv.columns.get_level_values(0):
            continue
        r_, p_, c_ = (piv[(met, s)].to_numpy() for s in ("ring", "path", "clique"))
        d1, d2 = boot_ci(r_ - p_), boot_ci(r_ - c_)
        res[met] = (d1, d2)
        lines.append(f"| {met} | {np.nanmean(r_):.3g} | {np.nanmean(p_):.3g} | {np.nanmean(c_):.3g} | "
                     f"{d1[0]:+.3g} [{d1[1]:+.3g}, {d1[2]:+.3g}] | {d2[0]:+.3g} [{d2[1]:+.3g}, {d2[2]:+.3g}] |")

    # by m (capacity pressure)
    lines += ["", "### By hidden size m (mixed models): ring - path and ring - clique, dloss_noise0.1 and llc_refined", "",
              "| m | ring-path dloss_noise0.1 | ring-clique dloss_noise0.1 | ring-path llc_refined | ring-clique llc_refined |",
              "|---|---|---|---|---|"]
    for m, g in piv.groupby(level="m"):
        f = lambda met, o: boot_ci((g[(met, "ring")] - g[(met, o)]).to_numpy())  # noqa: E731
        cells_ = [f("dloss_noise0.1", "path"), f("dloss_noise0.1", "clique"), f("llc_refined", "path"),
                  f("llc_refined", "clique")]
        lines.append(f"| {m} | " + " | ".join(f"{c[0]:+.3g} [{c[1]:+.3g}, {c[2]:+.3g}]" for c in cells_) + " |")

    # ---- single-structure models: geometry formation, emergence, durability
    single = df[df.structure != "mixed"]
    lines += ["", "## Single-structure models", "",
              "closed ring = order score >= 0.8 and closure < 2 for the structure's feature directions.", "",
              "| structure | share closed ring | median order score | median closure | median emergence step | "
              "dloss_noise0.1 | dloss_prune1 | dloss_retrain | dims_used | llc_whole |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, g in single.groupby("structure"):
        lines.append(f"| {s} | {g.closed_ring.mean():.2f} | {g.order_score.median():.2f} | {g.closure_ratio.median():.2f} | "
                     f"{g.get('order_emerge_step', pd.Series(dtype=float)).median():.0f} | "
                     f"{g['dloss_noise0.1'].median():.3g} | {g.dloss_prune1.median():.3g} | {g.dloss_retrain.median():.3g} | "
                     f"{g.dims_used.median():.3g} | {g.llc_whole.median():.3g} |")
    lines += ["", "### Share of closed rings by m (single-structure models)", "",
              "| structure | " + " | ".join(f"m={m}" for m in sorted(single.m.unique())) + " |",
              "|---|" + "---|" * single.m.nunique()]
    for s, g in single.groupby("structure"):
        sh = g.groupby("m").closed_ring.mean()
        lines.append(f"| {s} | " + " | ".join(f"{sh.get(m, np.nan):.2f}" for m in sorted(single.m.unique())) + " |")

    # ---- ring with a rare closing edge (H-nov)
    rr = single[single.structure.isin(["ring", "ring_rare", "path"])]
    lines += ["", "### Rare closing edge (H-nov toy version)", "",
              "| structure | median closure | share closure < 2 | median emergence step | median order score after retraining |",
              "|---|---|---|---|---|"]
    for s, g in rr.groupby("structure"):
        lines.append(f"| {s} | {g.closure_ratio.median():.2f} | {(g.closure_ratio < 2).mean():.2f} | "
                     f"{g.get('order_emerge_step', pd.Series(dtype=float)).median():.0f} | "
                     f"{g.get('order_score_retrain', pd.Series(dtype=float)).median():.2f} |")

    # ---- figure: mixed-model durability and LLC by m
    mets = ["dloss_noise0.1", "dloss_prune1", "dloss_retrain", "llc_refined", "dims_used"]
    fig, axes = plt.subplots(1, len(mets), figsize=(3.0 * len(mets), 2.8))
    agg = mx.groupby(["sub", "m"])[mets].median().reset_index()
    for ax, met in zip(axes, mets):
        for c, s in zip(rep.SERIES, ["ring", "path", "clique"]):
            d = agg[agg["sub"] == s]
            ax.plot(d.m, d[met], color=c, marker="o", ms=4, label=s)
        ax.set_xscale("log", base=2)
        ax.set_title(met, loc="left", fontsize=9, color=rep.INK)
        ax.set_xlabel("hidden size m")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, -0.03, "Mixed models (ring, path, cliques in one model); medians over seeds, n and q. "
             "Lower dloss = more durable.", color=rep.MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mixed_by_m.png", bbox_inches="tight")
    plt.close(fig)
    lines += ["", "![mixed](mixed_by_m.png)"]
    runs.atomic_write_text(out / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
