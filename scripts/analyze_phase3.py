"""Phase 3 analysis (M4): the preregistered primary metric, contrasts, and model, plus secondary measures.

  uv run python scripts/analyze_phase3.py --phase1 data/results/phase1_families/1.4b_s0_step143000 \
      --exp data/results/phase3_durability --seeds data/results/phase3_seeds

Steps (PREREGISTRATION.md):
  1. L* per family from Phase 1 (template, primary prep, seed-0 contexts).
  2. Primary: retention at L* after 20M tokens, mean over the two corpora; families with base score 0
     are excluded and counted.
  3. Contrasts: circle vs line controls (spec controls only, no torus/political/numbers), implicit vs
     explicit; difference in mean retention with 10,000-resample bootstrap CI over families.
  4. Mixed model: retention ~ z(topology_score) + z(log token_freq) + z(n_items) +
     z(participation_ratio) + z(act_norm) + (1|layer) + (1|prep), rows = family x layer x prep at 20M,
     with a family-clustered bootstrap CI for the topology coefficient.
  5. Secondary: retention vs perplexity across saves, compression variants, trajectory emergence /
     stability / closing-edge timing, seeds, behavior (wraparound split).
Writes reports/phase3/summary.md, CSVs, and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import report as rep  # noqa: E402
from topodur import durability as du  # noqa: E402
from topodur import runs  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.scoring import rescore_topology  # noqa: E402

PRIMARY = {"k_remove": 1, "d_proj": 5, "metric": "cosine", "variant": "last"}
SPEC_LINES = ["ordinals", "sizes", "life_stages", "centuries", "temperature", "city_latitude"]
IMPLICIT = ["implicit", "implicit_long"]


def boot_diff(a, b, n=10_000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def token_freq(families, ctx_dir: Path) -> dict[str, float]:
    out = {}
    for f in families:
        p = ctx_dir / f"{f}.summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            out[f] = float(np.mean(np.log10(np.asarray(s["total_count"], float) + 1)))
    return out


def mixed_model(rows: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    d = rows.dropna(subset=["retention", "topology_score", "log_freq", "n_items", "participation_ratio",
                            "act_norm"]).copy()
    d = d[np.isfinite(d.retention)]
    for c in ["topology_score", "log_freq", "n_items", "participation_ratio", "act_norm"]:
        d[c] = (d[c] - d[c].mean()) / (d[c].std() or 1)
    d["prep"] = d.k_remove.astype(str) + "_" + d.d_proj.astype(str) + "_" + d.metric
    d["one"] = 1
    formula = "retention ~ topology_score + log_freq + n_items + participation_ratio + act_norm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, d, groups="one", re_formula="0",
                        vc_formula={"layer": "0 + C(layer)", "prep": "0 + C(prep)"}).fit()
    ci = m.conf_int().loc["topology_score"].tolist()
    # family-clustered bootstrap of the OLS coefficient (fixed effects only), as the robustness CI
    rng = np.random.default_rng(0)
    fams = d.family.unique()
    coefs = []
    for _ in range(500):
        pick = rng.choice(fams, len(fams))
        b = pd.concat([d[d.family == f] for f in pick])
        try:
            coefs.append(smf.ols(formula, b).fit().params["topology_score"])
        except Exception:
            pass
    return {"coef": float(m.params["topology_score"]), "ci": ci, "n_rows": len(d), "n_families": len(fams),
            "cluster_boot_ci": [float(np.quantile(coefs, .025)), float(np.quantile(coefs, .975))] if coefs else None,
            "summary": str(m.summary())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True, help="Phase 1 unit dir (1.4B final, seed-0 contexts)")
    ap.add_argument("--exp", required=True, help="phase3_durability results dir")
    ap.add_argument("--seeds", default=None, help="phase3_seeds results dir")
    ap.add_argument("--contexts", default="data/contexts/pile_v1")
    ap.add_argument("--base-unit", default="1.4b_s0_step143000")
    ap.add_argument("--final-tokens", default="20M")
    ap.add_argument("--out", default="reports/phase3")
    a = ap.parse_args()
    out = runs.REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    exp = Path(a.exp)
    p1 = rescore_topology(pd.read_parquet(Path(a.phase1) / "tda_scores.parquet"))
    lstar = du.select_layers(p1, PRIMARY)
    df = du.load_units(exp)
    fams = sorted(df.family.unique())
    meta = {f: load_family(f) for f in fams}
    lines = ["# Phase 3 durability (M4)", "", "Preregistered analysis (PREREGISTRATION.md). L* per family: " +
             ", ".join(f"{f} {l}" for f, l in sorted(lstar.items())), ""]

    # perplexity per unit
    ppl = {}
    for p in exp.glob("*/eval.json"):
        ppl[p.parent.name] = json.loads(p.read_text())["ppl"].get("pile_test", np.nan)

    var = du.variant_retention(df[df.step == 143000], lstar, PRIMARY, a.base_unit, exp, ppl)
    var.to_csv(out / "variant_retention.csv", index=False)

    # ---- primary metric
    ft = var[var.variant.str.startswith("lora-") & var.variant.str.endswith(a.final_tokens)]
    prim = ft.groupby("family").agg(retention=("retention", "mean"), score_base=("score_base", "first"),
                                    topology=("topology", "first"), origin=("origin", "first"),
                                    ppl_ratio=("ppl_ratio", "mean")).reset_index()
    excl = prim[~np.isfinite(prim.retention)].family.tolist()
    lines += ["## Primary metric: retention after 20M tokens of LoRA fine-tuning (mean of code, German Wikipedia)", "",
              f"Excluded (base score 0 at L*, retention undefined): {', '.join(excl) or 'none'}.", "",
              "| family | topology | origin | base score | retention | ppl ratio |", "|---|---|---|---|---|---|"]
    for r in prim.sort_values(["topology", "family"]).itertuples():
        lines.append(f"| {r.family} | {r.topology} | {r.origin} | {r.score_base:.2f} | {r.retention:.2f} | {r.ppl_ratio:.3f} |")
    circ = prim[(prim.topology == "circle")]
    line = prim[prim.family.isin(SPEC_LINES)]
    c1 = boot_diff(circ.retention, line.retention)
    impl = circ[circ.origin.isin(IMPLICIT)]
    expl = circ[circ.origin == "explicit"]
    c2 = boot_diff(impl.retention, expl.retention)
    lines += ["", "## Primary contrasts (bootstrap over families, 10,000 resamples)", "",
              "| contrast | n | difference in mean retention [95% CI] |", "|---|---|---|",
              f"| circles - line controls | {len(circ)} vs {len(line)} | {c1[0]:+.3f} [{c1[1]:+.3f}, {c1[2]:+.3f}] |",
              f"| implicit - explicit circles | {len(impl)} vs {len(expl)} | {c2[0]:+.3f} [{c2[1]:+.3f}, {c2[2]:+.3f}] |"]

    # ---- mixed model on family x layer x prep rows at 20M
    freq = token_freq(fams, runs.REPO_ROOT / a.contexts)
    base_rows = df[(df.unit == a.base_unit) & (df.source == "template")]
    ft_units = [u for u in df.unit.unique() if "+lora-" in u and u.endswith(a.final_tokens)]
    rows = []
    keys = ["family", "layer", "k_remove", "d_proj", "metric", "variant"]
    b = base_rows.set_index(keys)
    for u in ft_units:
        v = df[df.unit == u].set_index(keys)
        j = b[["topology_score", "n_items", "participation_ratio", "act_norm"]].join(
            v[["topology_score"]].rename(columns={"topology_score": "score_ft"}), how="inner").reset_index()
        j["retention"] = np.where(j.topology_score > 0, j.score_ft / j.topology_score, np.nan)
        j["unit"] = u
        rows.append(j)
    if rows:
        R = pd.concat(rows)
        R["log_freq"] = R.family.map(freq)
        R.to_csv(out / "model_rows.csv", index=False)
        try:
            mm = mixed_model(R)
            lines += ["", "## Mixed-effects model (20M saves, both corpora)", "",
                      f"topology_score coefficient (z-scored predictors): **{mm['coef']:+.3f}**, 95% CI "
                      f"[{mm['ci'][0]:+.3f}, {mm['ci'][1]:+.3f}]; family-clustered bootstrap CI "
                      f"{mm['cluster_boot_ci']}; {mm['n_rows']} rows, {mm['n_families']} families.", "",
                      "```", mm["summary"], "```"]
        except Exception as e:
            lines += ["", f"Mixed model failed: {e}"]

    # ---- decision rule readout
    lines += ["", "## Decision rules (PREREGISTRATION.md section 5)", ""]
    if np.isfinite(c1[1]):
        if c1[1] > 0:
            lines.append("- Circles retain more than line controls (CI excludes 0).")
        elif c1[2] < 0:
            lines.append("- Circles retain LESS than line controls (CI excludes 0).")
        else:
            lines.append("- Circle minus line retention CI includes 0.")

    # ---- secondary: all variants, retention vs perplexity
    lines += ["", "## Secondary: retention by variant (mean over families with a base loop)", "",
              "| variant | circles | line controls | ppl ratio |", "|---|---|---|---|"]
    for v, g in var.groupby("variant"):
        lines.append(f"| {v} | {g[g.topology == 'circle'].retention.mean():.2f} | "
                     f"{g[g.family.isin(SPEC_LINES)].retention.mean():.2f} | {g.ppl_ratio.mean():.3f} |")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    agg = var.groupby(["variant", "topology"]).agg(r=("retention", "mean"), p=("ppl_ratio", "mean")).reset_index()
    for c, t in zip(rep.SERIES, ["circle", "interval", "torus"]):
        g = agg[agg.topology == t]
        ax.scatter(g.p, g.r, color=c, s=28, label=t, edgecolor="white", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("held-out Pile perplexity ratio (variant / base)")
    ax.set_ylabel("mean retention at L*")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "retention_vs_ppl.png")
    plt.close(fig)
    lines += ["", "![retention vs perplexity](retention_vs_ppl.png)"]

    # ---- trajectory
    traj = df[(df.model_variant == "base") & (df.model == "1.4b")]
    if traj.step.nunique() > 3:
        T = du.trajectory(traj, lstar, PRIMARY, exp)
        T.to_csv(out / "trajectory.csv", index=False)
        lines += ["", "## Trajectory (1.4B, log-spaced checkpoints, at L*)", "",
                  "| family | topology | emergence step | first ordered step | first closed step | score var after |",
                  "|---|---|---|---|---|---|"]
        for f, g in T.groupby("family"):
            ordered = g[(g.order_score >= 0.8) & (g.order_p < 0.05)].step
            closed = g[g.closure_ratio < 2].step
            lines.append(f"| {f} | {g.topology.iloc[0]} | {g.emergence_step.iloc[0]:.0f} | "
                         f"{ordered.min() if len(ordered) else np.nan:.0f} | {closed.min() if len(closed) else np.nan:.0f} | "
                         f"{g.score_var_after.iloc[0]:.3g} |")
        fs = [f for f in fams if meta[f].topology in ("circle", "torus")]
        cols = 5
        fig, axes = plt.subplots(-(-len(fs) // cols), cols, figsize=(2.6 * cols, 2.0 * -(-len(fs) // cols)), sharex=True)
        for ax, f in zip(axes.flat, fs):
            g = T[T.family == f]
            ax.plot(g.step.clip(lower=0.5), g.topology_score, color=rep.SERIES[0], lw=1.6)
            ax2 = ax.twinx()
            ax2.plot(g.step.clip(lower=0.5), g.closure_ratio, color=rep.SERIES[1], lw=1.2, ls="--")
            ax2.axhline(2, color=rep.GRID, lw=1)
            ax.set_xscale("log")
            ax.set_title(f, loc="left", fontsize=8)
        for ax in list(axes.flat)[len(fs):]:
            ax.axis("off")
        fig.text(0.01, -0.01, "Solid: topology score at L* (left axis). Dashed: closure ratio (right axis; < 2 = closed). "
                 "x: training step (log).", color=rep.MUTED, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "trajectory.png", bbox_inches="tight")
        plt.close(fig)
        lines += ["", "![trajectory](trajectory.png)"]

    # ---- behavior
    beh = [pd.read_parquet(p) for p in exp.glob("*/behavior.parquet")]
    if beh:
        B = pd.concat(beh)
        B.to_csv(out / "behavior.csv", index=False)
        lines += ["", "## Behavior (offset queries), base and variants", "",
                  "| unit | family | acc | acc wrap | acc no-wrap | margin wrap | margin no-wrap |", "|---|---|---|---|---|---|---|"]
        for r in B.sort_values(["family", "unit"]).itertuples():
            lines.append(f"| {r.unit} | {r.family} | {r.acc:.2f} | {r.acc_wrap:.2f} | {r.acc_nowrap:.2f} | "
                         f"{r.margin_wrap:.2f} | {r.margin_nowrap:.2f} |")

    # ---- seeds
    if a.seeds and Path(a.seeds).exists():
        sd = du.load_units(Path(a.seeds))
        if len(sd):
            l410 = du.select_layers(sd[sd.seed == 0], PRIMARY)
            units = sorted(sd.unit.unique())
            S = du.seed_agreement(Path(a.seeds), units, fams, l410, PRIMARY, sd)
            S.to_csv(out / "seeds.csv", index=False)
            lines += ["", "## Seeds (PolyPythia 410M)", "", "```", S.round(3).to_string(index=False), "```"]
    runs.atomic_write_text(out / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
