"""Phase 1/2 summary (M2) for one scored unit: all families, both context sources.

  uv run python scripts/report_phase1.py --config configs/phase1_families.yaml --unit 1.4b_s0_step143000

Writes reports/<experiment>/<unit>/summary.md with:
  - per family: best layer (primary prep, template contexts), the primary topology score there
    (order x closed for <= 30 items, h1_ratio above), closure, circle_pass layers, robustness across
    preps, Gaussian-null p, and the same score from mined Pile contexts
  - circle families vs line controls at the default layers and at each family's best layer
  - figures: topology score vs layer for every family (small multiples, template vs pile), and the
    persistence diagram + PCA at each family's best layer
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

import report as rep  # noqa: E402  (scripts/report.py: shared style + best-layer figure)
from topodur import runs  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.geometry import Prep  # noqa: E402

DEFAULT_LAYERS = list(range(12, 17))


def passcol(df: pd.DataFrame) -> str:
    return "shape_pass" if "shape_pass" in df else "circle_pass"


def best_row(d: pd.DataFrame) -> pd.Series:
    return d.sort_values(["topology_score", "h1_ratio"], ascending=False).iloc[0]


def profiles(df, prim, fams, path, n_layers):
    n = len(fams)
    cols = 6
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 1.9 * rows), sharex=True)
    for ax, f in zip(axes.flat, fams):
        d = df[prim & (df.family == f)]
        ax.axvspan(DEFAULT_LAYERS[0], DEFAULT_LAYERS[-1], color=rep.GRID, alpha=0.6, lw=0)
        for src, c, ls in (("template", rep.SERIES[0], "-"), ("pile", rep.SERIES[1], "--")):
            s = d[d.source == src].sort_values("layer")
            if len(s):
                ax.plot(s.layer, s.topology_score, color=c, ls=ls, lw=1.6, label=src)
                cp = s[s[passcol(s)]]
                ax.scatter(cp.layer, cp.topology_score, s=14, color=c, edgecolor="white", lw=0.8, zorder=3)
        fam = load_family(f)
        meas = "h1_ratio" if fam.n_items > 30 else ("line order x open" if fam.topology == "interval"
                                                      else "order x closed")
        ax.set_title(f"{f} ({fam.topology[:5]}, {meas})", loc="left", fontsize=7.5, color=rep.INK)
        ax.tick_params(labelsize=7)
        if meas != "h1_ratio":
            ax.set_ylim(-0.05, 1.05)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.text(0.01, -0.01, "Primary preprocessing. Shaded: default layers 12-16. Dots: shape_pass (circles: significant H1, "
             "exact cyclic order p<0.05, closure < 2; line controls: linear order p<0.05, closure >= 2). x: layer.", color=rep.MUTED, fontsize=8)
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
    prim = rep.primary_mask(df, p)
    fams = [f for f in cfg["families"] if f in set(df.family)]
    n_layers = int(df.layer.max()) + 1
    profiles(df, prim, fams, out / "profiles.png", n_layers)

    lines = [f"# {exp} / {a.unit}", "",
             f"Primary preprocessing {p}; template contexts unless stated. Topology score = order_score x "
             "1[closure < 2] for circle families of 30 items or fewer; line_order_score x 1[closure >= 2] for "
             "line controls (an ordered arc that stays open); h1_ratio above 30 items (cities). shape_pass = "
             "significant H1 + exact cyclic order (p < 0.05) + closure < 2 for circles; linear order p < 0.05 + "
             "closure >= 2 for line controls. Robustness = share of the 36 preprocessing settings with "
             "shape_pass at the best layer (< 50% = fragile).", "",
             "| family | topology | origin | items | best layer | score | closure | n_sig_h1 | gauss p | "
             "shape_pass layers | robustness | score @12-16 | pile items | pile best layer | pile score |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for f in fams:
        fam = load_family(f)
        d = df[prim & (df.family == f) & (df.source == "template")]
        if d.empty:
            lines.append(f"| {f} | (no primary-preprocessing template rows) |")
            continue
        b = best_row(d)
        pc = passcol(df)
        rob = df[(df.family == f) & (df.source == "template") & (df.layer == b.layer)][pc].mean()
        layers = sorted(d[d[pc]].layer)
        lay = _ranges(layers) or "none"
        dflt = d[d.layer.isin(DEFAULT_LAYERS)].topology_score.mean()
        dp = df[prim & (df.family == f) & (df.source == "pile")]
        if len(dp):
            bp = best_row(dp)
            pile = (int(bp.n_items), int(bp.layer), f"{bp.topology_score:.2f}")
        else:
            pile = ("-", "-", "-")
        flag = " (fragile)" if rob < 0.5 else ""
        lines.append(f"| {f} | {fam.topology} | {fam.origin} | {fam.n_items} | {b.layer} | {b.topology_score:.2f} | "
                     f"{b.closure_ratio:.2f} | {b.get('n_sig_h1', np.nan):.0f} | {b.get('gauss_p_h1_ratio', np.nan):.2f} | "
                     f"{lay} | {rob:.0%}{flag} | {dflt:.2f} | {pile[0]} | {pile[1]} | {pile[2]} |")
        summary.append(dict(family=f, topology=fam.topology, origin=fam.origin, best=b.topology_score,
                            default=dflt, measure=b.topology_measure))
        try:
            rep.best_layer_figure(unit_dir, fam, int(b.layer), Prep(**p), b.to_dict(), out / f"best_{f}.png")
        except Exception as e:  # a figure must never block the summary
            lines.append(f"| ({f}: figure failed: {e}) |")

    s = pd.DataFrame(summary)
    small = s[s.measure != "h1_ratio"]
    lines += ["", "## Circles vs line controls (families of 30 items or fewer, template contexts)", "",
              "| group | n | mean score at best layer | mean score at layers 12-16 |", "|---|---|---|---|"]
    for name, g in (("circle/torus", small[small.topology.isin(["circle", "torus"])]),
                    ("line controls", small[small.topology == "interval"]),
                    ("unknown (political)", small[small.topology == "unknown"])):
        lines.append(f"| {name} | {len(g)} | {g.best.mean():.2f} | {g.default.mean():.2f} |")
    lines += ["", "Cities (h1_ratio): " + ", ".join(f"{r.family} {r.best:.2f} (best) / {r.default:.2f} (12-16)"
                                                    for r in s[s.measure == 'h1_ratio'].itertuples()), "",
              "![profiles](profiles.png)", ""] + [f"![{f}](best_{f}.png)" for f in fams]
    runs.atomic_write_text(out / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:len(fams) + 20]))
    print(f"\nreport written to {out}")
    return 0


def _ranges(xs: list[int]) -> str:
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


if __name__ == "__main__":
    sys.exit(main())
