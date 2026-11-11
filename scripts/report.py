"""Gate report for one scored unit: markdown summary + figures in reports/<experiment>/<unit>/.

  uv run python scripts/report.py --config configs/phase0_replication.yaml --unit 1.4b_s0_step143000

Contents:
  - per family: best layer under the primary preprocessing, n_sig_h1, order score/p, gate pass,
    and the fraction of preprocessing settings that pass at that layer (fragility flag < 0.5)
  - layer profile: h1_ratio, order_score, and pass fraction vs layer (small multiples, one y-scale each)
  - at the best layer: persistence diagram and the 2D PCA of the preprocessed centroids, items
    joined in ground-truth order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs, tda  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.extract import Background, FamilyActs  # noqa: E402
from topodur.geometry import Prep, distance_matrix, preprocess  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "figure.dpi": 130,
})


def primary_mask(df: pd.DataFrame, p: dict) -> pd.Series:
    return ((df.k_remove == p["k_remove"]) & (df.d_proj == p["d_proj"]) &
            (df.metric == p["metric"]) & (df.variant == p["variant"]))


def layer_profile(df, fams, prim, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
    panels = [("h1_ratio", "H1 death/birth (primary prep)"), ("order_score", "order score (primary prep)"),
              ("pass_frac", "fraction of preps passing gate")]
    pf = df.groupby(["family", "layer"]).gate_pass.mean().rename("pass_frac").reset_index()
    for ax, (col, title) in zip(axes, panels):
        for c, f in zip(SERIES, fams):
            d = (pf[pf.family == f] if col == "pass_frac" else df[prim & (df.family == f)]).sort_values("layer")
            ax.plot(d.layer, d[col], color=c, label=f)
            if col != "pass_frac":
                passed = d[d.gate_pass]
                ax.scatter(passed.layer, passed[col], s=28, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("layer")
    axes[0].legend(frameon=False)
    fig.text(0.01, -0.02, "Dots: layers where the primary setting passes the gate (significant H1 + exact cyclic order, p<0.05).",
             color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def best_layer_figure(unit_dir, fam, layer, prep: Prep, row, path):
    acts = FamilyActs.load(unit_dir / f"acts_{fam.name}.npz")
    bg = Background.load(unit_dir / "background.npz")
    li, bli = acts.layers.index(layer), bg.layers.index(layer)
    Z = preprocess(acts.centroids[prep.variant][li], bg.mean[bli], bg.pcs[bli], prep)
    D = distance_matrix(Z, prep.metric)
    dg = tda.diagrams(D, maxdim=2 if fam.topology == "torus" else 1)["dgms"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    hi = max(np.nanmax(tda.finite(g)) if len(tda.finite(g)) else 0 for g in dg) * 1.05 or 1
    a1.plot([0, hi], [0, hi], color=MUTED, lw=1)
    c = row.get("c_h1")
    if c is not None and np.isfinite(c):
        a1.fill_between([0, hi], [0, hi], [2 * c, hi + 2 * c], color=GRID, alpha=0.8, lw=0,
                        label="H1 band (lifetime < 2c)")
    for dim, col in zip(range(len(dg)), [MUTED, SERIES[0], SERIES[1]]):
        f = tda.finite(dg[dim])
        if len(f):
            a1.scatter(f[:, 0], f[:, 1], s=22, color=col, edgecolor="white", linewidth=1, label=f"H{dim}", zorder=3)
    a1.set_xlim(0, hi)
    a1.set_ylim(0, hi)
    a1.set_xlabel("birth")
    a1.set_ylabel("death")
    a1.set_title(f"{fam.name}: persistence, layer {layer}", loc="left")
    a1.legend(frameon=False, fontsize=8)

    order = np.argsort(fam.angles())
    xy = Z[:, :2]
    loop = np.r_[order, order[:1]] if fam.is_cyclic else order
    a2.plot(xy[loop, 0], xy[loop, 1], color=GRID, lw=1.5, zorder=1)
    cmap = plt.get_cmap("twilight" if fam.is_cyclic else "viridis")
    a2.scatter(xy[:, 0], xy[:, 1], c=cmap(np.linspace(0, 1, fam.n_items, endpoint=fam.topology == "interval")),
               s=60, edgecolor="white", linewidth=1.5, zorder=3)
    for i, name in enumerate(fam.items):
        a2.annotate(name[:4] if fam.n_items > 12 else name, xy[i], xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    a2.set_title(f"PCA of preprocessed centroids ({prep.tag()})", loc="left")
    a2.set_aspect("equal", "datalim")
    a2.set_xlabel("PC1")
    a2.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run else "")
    unit_dir = runs.run_dir(exp, a.unit)
    df = pd.read_parquet(unit_dir / "tda_scores.parquet")
    out = runs.REPO_ROOT / "reports" / exp / a.unit
    out.mkdir(parents=True, exist_ok=True)
    p = cfg["scoring"]["primary"]
    prim = primary_mask(df, p)
    fams = list(dict.fromkeys(df.family))

    layer_profile(df, fams, prim, out / "layer_profile.png")
    lines = [f"# {exp} / {a.unit}", "",
             f"Primary preprocessing: {p}. Gate: significant H1 (lifetime > 2c, bootstrap) and circular "
             "coordinate in exact ground-truth cyclic order with permutation p < 0.05.", "",
             "closure = distance(last item, first item) / median distance between neighbours in the true order: "
             "about 1 for a closed loop, large for a line.", "",
             "| family | topology | best layer | h1_ratio | n_sig_h1 | order_score | order_p | closure | gate (primary) | "
             "preps passing at best layer | layers passing (primary) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        fam = load_family(f)
        d = df[prim & (df.family == f)]
        cand = d[d.gate_pass] if d.gate_pass.any() else d
        best = cand.sort_values(["h1_ratio"], ascending=False).iloc[0]
        frac = df[(df.family == f) & (df.layer == best.layer)].gate_pass.mean()
        passing = ", ".join(map(str, sorted(d[d.gate_pass].layer))) or "none"
        flag = " (fragile)" if 0 < frac < 0.5 else ""
        lines.append(f"| {f} | {fam.topology} | {best.layer} | {best.h1_ratio:.2f} | {best.get('n_sig_h1', 'na')} | "
                     f"{best.order_score:.2f} | {best.order_p:.3g} | {best.closure_ratio:.2f} | "
                     f"{'PASS' if best.gate_pass else 'fail'} | "
                     f"{frac:.0%}{flag} | {passing} |")
        best_layer_figure(unit_dir, fam, int(best.layer), Prep(**p), best.to_dict(), out / f"best_{f}.png")
    lines += ["", "![layer profile](layer_profile.png)", ""] + [f"![{f}](best_{f}.png)" for f in fams]
    runs.atomic_write_text(out / "gate.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
