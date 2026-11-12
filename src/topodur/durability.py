"""Durability measures (Phase 3), computed from scored units.

Layer selection (preregistered): each family's layer L* is the layer with the highest primary
topology score in Phase 1 (template contexts, primary preprocessing, context seed 0; ties broken by
h1_ratio, then the lower layer). Phase 3 units are measured with context seed 1, so retention is not
biased by selecting on the same noise.

  trajectory   per family at L*: topology score per checkpoint, emergence step (first checkpoint that
               passes and keeps passing for the next 3), stability after emergence (variance of the
               score, and Procrustes distance between consecutive checkpoints), closure ratio over time
  variants     retention = score(variant) / score(base) at L* (and the mean over layers 12-16),
               Procrustes distance from the base configuration, held-out perplexity ratio
  seeds        cross-seed Procrustes distance and linear CKA of the family configuration, and the share
               of seeds whose configuration passes at the family's layer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .extract import Background, FamilyActs
from .geometry import Prep, linear_cka, preprocess, procrustes_distance
from .scoring import rescore_topology

DEFAULT_BAND = list(range(12, 17))


def primary_rows(df: pd.DataFrame, primary: dict) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for k, v in primary.items():
        m &= (df[k] == v).to_numpy()
    if "source" in df:
        m &= (df["source"] == "template").to_numpy()
    return df[m]


def passes(df: pd.DataFrame) -> pd.Series:
    """Expected shape present: shape_pass for families of <= 30 items (closed ring for circles, open
    ordered arc for line controls); for larger families (cities, where the order test is not the
    measure) a significant H1 class with closure < 2. Line controls of any size use shape_pass."""
    small = df["topology_measure"] != "h1_ratio"
    big_pass = (df.get("n_sig_h1", 0) >= 1) & (df["closure_ratio"] < 2)
    sp = df["shape_pass"] if "shape_pass" in df else df["circle_pass"]
    use_shape = small | (df["topology"] == "interval")  # line controls of any size: the open-arc rule
    return np.where(use_shape, sp, big_pass).astype(bool)


def select_layers(phase1: pd.DataFrame, primary: dict) -> dict[str, int]:
    d = primary_rows(phase1, primary).sort_values(["family", "topology_score", "h1_ratio", "layer"],
                                                  ascending=[True, False, False, True])
    return {f: int(g.iloc[0].layer) for f, g in d.groupby("family")}


def load_units(exp_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(exp_dir.glob("*/tda_scores.parquet")):
        df = rescore_topology(pd.read_parquet(p))
        df["unit"] = p.parent.name
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def at_layers(df: pd.DataFrame, lstar: dict[str, int]) -> pd.DataFrame:
    return df[df.layer == df.family.map(lstar)]


def emergence(steps: np.ndarray, ok: np.ndarray, hold: int = 3) -> float:
    """First step that passes and keeps passing for the next `hold` checkpoints (nan if never)."""
    for i in range(len(steps)):
        if ok[i:i + hold + 1].all() and i + hold < len(steps):
            return float(steps[i])
    return float("nan")


def _Z(unit_dir: Path, family: str, layer: int, prep: Prep, bg_cache: dict) -> np.ndarray | None:
    p = unit_dir / f"acts_{family}.npz"
    if not p.exists():
        return None
    acts = FamilyActs.load(p)
    if unit_dir not in bg_cache:
        bg_cache[unit_dir] = Background.load(unit_dir / "background.npz")
    bg = bg_cache[unit_dir]
    li, bli = acts.layers.index(layer), bg.layers.index(layer)
    return preprocess(acts.centroids[prep.variant][li], bg.mean[bli], bg.pcs[bli], prep)


def trajectory(df: pd.DataFrame, lstar: dict[str, int], primary: dict, exp_dir: Path | None = None,
               size: str = "1.4b") -> pd.DataFrame:
    """One row per family x checkpoint (at L*, primary prep), with emergence and stability columns."""
    d = primary_rows(df[(df.model_variant == "base") & (df.model == size)], primary)
    d = at_layers(d, lstar).sort_values(["family", "step"]).copy()
    d["passes"] = passes(d)
    prep = Prep(**primary)
    out = []
    bg_cache: dict = {}
    for f, g in d.groupby("family"):
        g = g.copy()
        e = emergence(g.step.to_numpy(), g.passes.to_numpy())
        g["emergence_step"] = e
        after = g[g.step >= e] if np.isfinite(e) else g.iloc[0:0]
        g["score_var_after"] = float(after.topology_score.var()) if len(after) > 1 else np.nan
        if exp_dir is not None:
            Zs = [_Z(exp_dir / u, f, lstar[f], prep, bg_cache) for u in g.unit]
            pro = [np.nan] + [procrustes_distance(a, b) if a is not None and b is not None else np.nan
                              for a, b in zip(Zs[:-1], Zs[1:])]
            g["procrustes_prev"] = pro
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def variant_retention(df: pd.DataFrame, lstar: dict[str, int], primary: dict, base_unit: str,
                      exp_dir: Path | None = None, ppl: dict | None = None) -> pd.DataFrame:
    """One row per family x variant unit."""
    p = primary_rows(df, primary)
    base = p[p.unit == base_unit]
    rows = []
    prep = Prep(**primary)
    bg_cache: dict = {}
    for unit, g in p[p.unit != base_unit].groupby("unit"):
        for f, gf in g.groupby("family"):
            bf = base[base.family == f]
            L = lstar.get(f)
            if L is None or bf.empty:
                continue
            s_b = bf[bf.layer == L].topology_score.mean()
            s_v = gf[gf.layer == L].topology_score.mean()
            band_b = bf[bf.layer.isin(DEFAULT_BAND)].topology_score.mean()
            band_v = gf[gf.layer.isin(DEFAULT_BAND)].topology_score.mean()
            r = {"unit": unit, "variant": gf.model_variant.iloc[0], "family": f, "layer": L,
                 "topology": gf.topology.iloc[0], "origin": gf.origin.iloc[0],
                 "score_base": s_b, "score_variant": s_v,
                 "retention": s_v / s_b if s_b > 0 else np.nan,
                 "retention_band": band_v / band_b if band_b > 0 else np.nan,
                 "passes_base": bool(passes(bf[bf.layer == L]).any()),
                 "passes_variant": bool(passes(gf[gf.layer == L]).any())}
            if exp_dir is not None:
                Zb = _Z(exp_dir / base_unit, f, L, prep, bg_cache)
                Zv = _Z(exp_dir / unit, f, L, prep, bg_cache)
                r["procrustes_from_base"] = procrustes_distance(Zb, Zv) if Zb is not None and Zv is not None else np.nan
            if ppl:
                r["ppl_ratio"] = ppl.get(unit, np.nan) / ppl.get(base_unit, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def seed_agreement(exp_dir: Path, units: list[str], families: list[str], lstar: dict[str, int],
                   primary: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per family: mean pairwise Procrustes distance and CKA across seeds at L*, share of seeds passing."""
    prep = Prep(**primary)
    bg_cache: dict = {}
    rows = []
    for f in families:
        L = lstar.get(f)
        if L is None:
            continue
        Zs = [z for z in (_Z(exp_dir / u, f, L, prep, bg_cache) for u in units) if z is not None]
        pairs = [(i, j) for i in range(len(Zs)) for j in range(i + 1, len(Zs))]
        r = {"family": f, "layer": L, "n_seeds": len(Zs),
             "procrustes_mean": float(np.mean([procrustes_distance(Zs[i], Zs[j]) for i, j in pairs])) if pairs else np.nan,
             "cka_mean": float(np.mean([linear_cka(Zs[i], Zs[j]) for i, j in pairs])) if pairs else np.nan}
        if df is not None:
            d = at_layers(primary_rows(df[df.family == f], primary), lstar)
            r["share_passing"] = float(passes(d).mean()) if len(d) else np.nan
            r["score_mean"] = float(d.topology_score.mean()) if len(d) else np.nan
            r["topology"] = d.topology.iloc[0] if len(d) else ""
        rows.append(r)
    return pd.DataFrame(rows)
