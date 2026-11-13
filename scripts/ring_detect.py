"""Label-free ring detection with calibration (src/topodur/ring.py). CPU only.

  uv run python scripts/ring_detect.py --experiment phase1_families --unit 1.4b_s0_step143000
  ... --limit 3            # test pass: first 3 layers, small nulls, writes ring_detect_dryrun

Dials (group size of context mini-centroids x projection dimension) are chosen on calibration cases
only, never on the target families:
  sensitivity  synthetic rings planted in real data (numbers, ordinals): the top-2 centroid coordinates
               are replaced by a ring at 0.5x / 1x their spread; other dims and context scatter stay real
  false pos.   line controls, and synthetic half-circle arcs planted the same way
  reference    days, months, cities_longitude (passed the old labeled rule) are reported, not tuned on
Objective: planted-ring detection rate - max(line-control rate, planted-arc rate).
Output: data/results/ring_detect[_dryrun]/<unit>/ring_scores.parquet, reports/ring_detect[_dryrun]/<unit>/
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.extract import Background, FamilyActs  # noqa: E402
from topodur.ring import ring_task  # noqa: E402

log = logging.getLogger("ring_detect")

REFERENCE = ["days", "months", "cities_longitude"]
LINES = ["ordinals", "sizes", "life_stages", "centuries", "temperature", "city_latitude", "numbers"]
PLANT_BASES = ["numbers", "ordinals"]
PLANT_AMPS = [0.5, 1.0]   # ring radius relative to the real centroids' spread in the top-2 PCs
_CACHE: dict = {}


def _task(unit_dir: str, fam: str, li: int, group: int, d: int, planted: tuple | None, kw: dict) -> dict:
    if fam not in _CACHE:
        _CACHE.clear()
        _CACHE[fam] = FamilyActs.load(Path(unit_dir) / f"acts_{fam}.npz")
    acts = _CACHE[fam]
    bg = Background.load(Path(unit_dir) / "background.npz")
    bli = bg.layers.index(acts.layers[li])
    truth = load_family(fam).angles()
    if load_family(fam).topology == "interval":
        truth = 2 * np.pi * np.arange(len(truth)) / len(truth)
    r = ring_task(acts.instances["last"][li], acts.n_inst, bg.mean[bli], bg.pcs[bli], truth, group=group, d=d,
                  planted=planted[0] if planted else None, amp=planted[1] if planted else 1.0, **kw)
    name = f"planted_{planted[0]}{planted[1]:g}@{fam}" if planted else fam
    role = (f"planted_{planted[0]}" if planted else
            "reference" if fam in REFERENCE else "line" if fam in LINES else "target")
    return {"family": name, "role": role, "layer": acts.layers[li], "group": group, "d_proj": d,
            "n_items": acts.instances["last"].shape[1], **r}


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    fmt = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
    rows = [" | ".join(map(str, d.columns)), " | ".join("---" for _ in d.columns)]
    rows += [" | ".join(fmt(v) for v in r) for r in d.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def calibrate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["group", "d_proj", "role"]).detected.mean().unstack("role")
    g["objective"] = g.get("planted_ring", 0) - np.maximum(g.get("line", 0), g.get("planted_arc", 0))
    return g.sort_values("objective", ascending=False)


def heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (df[["family", "role"]].drop_duplicates()
             .assign(r=lambda x: x.role.map({"reference": 0, "target": 1, "line": 2, "planted_ring": 3, "planted_arc": 4}))
             .sort_values(["r", "family"]).family.tolist())
    p = df.assign(s=-np.log10(df[["pA", "pB", "rep_p"]].max(axis=1).clip(lower=1e-3)))
    M = p.pivot_table(index="family", columns="layer", values="s").reindex(order)
    Det = p.pivot_table(index="family", columns="layer", values="detected").reindex(order)
    fig, ax = plt.subplots(figsize=(0.32 * M.shape[1] + 3, 0.28 * len(order) + 1.5))
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    yy, xx = np.nonzero(Det.fillna(0).values > 0)
    ax.scatter(xx, yy, s=10, c="#d1495b", marker="o", label="detected")
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_xticks(range(M.shape[1]), M.columns, fontsize=7)
    ax.set_xlabel("layer")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="-log10 max(p_A, p_B, p_rep)", shrink=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(1, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="phase1_families")
    ap.add_argument("--unit", default="1.4b_s0_step143000")
    ap.add_argument("--groups", type=int, nargs="*", default=[0, 4, 16])
    ap.add_argument("--ds", type=int, nargs="*", default=[3, 5, 8])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--limit", type=int, default=None, help="test pass: first N layers, small nulls")
    a = ap.parse_args()
    unit_dir = runs.run_dir(a.experiment, a.unit)
    tag = "ring_detect" + ("_dryrun" if a.limit else "")
    out_dir = runs.run_dir(tag, a.unit)
    rep_dir = runs.REPO_ROOT / "reports" / tag / a.unit
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    runs.setup_logging(out_dir / "ring_detect.log")
    kw = {"n_null": 10, "n_perm": 200} if a.limit else {"n_null": 40, "n_perm": 1000}

    fams = a.families or [p.name[5:-4] for p in sorted(unit_dir.glob("acts_*.npz")) if "@" not in p.name]
    tasks = []
    for f in fams:
        L = len(FamilyActs.load(unit_dir / f"acts_{f}.npz").layers)
        lis = range(min(a.limit, L) if a.limit else L)
        plants = [None] + ([(s, amp) for s in ("ring", "arc") for amp in PLANT_AMPS] if f in PLANT_BASES else [])
        tasks += [(str(unit_dir), f, li, g, d, pl) for li, g, d, pl in itertools.product(lis, a.groups, a.ds, plants)]
    log.info("%d tasks on %d workers", len(tasks), a.workers)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_task, *t, kw) for t in tasks]
        for k, fut in enumerate(as_completed(futs)):
            try:
                rows.append(fut.result())
            except Exception as e:  # one bad cell must not lose the run
                log.exception("task failed: %s", e)
            if k % 200 == 0:
                log.info("done %d/%d", k + 1, len(tasks))
    df = pd.DataFrame(rows).sort_values(["family", "group", "d_proj", "layer"]).reset_index(drop=True)
    with runs.atomic_path(out_dir / "ring_scores.parquet") as tmp:
        df.to_parquet(tmp, index=False)

    cal = calibrate(df)
    best = cal.index[0]
    sel = df[(df.group == best[0]) & (df.d_proj == best[1])]
    heatmap(sel, rep_dir / "detection_best_dial.png",
            f"{a.unit}: label-free ring detection, group={best[0]} d={best[1]} (dial chosen on calibration only)")
    fam_tab = (sel.groupby(["role", "family"])
               .agg(layers_detected=("detected", "sum"), n_layers=("layer", "nunique"),
                    best_layer=("layer", lambda s: int(sel.loc[s.index].assign(
                        m=sel.loc[s.index][["pA", "pB", "rep_p"]].max(axis=1)).sort_values(["m", "top_life_A"],
                        ascending=[True, False]).layer.iloc[0])),
                    min_p=("pA", "min"), median_coherence=("coherence_A", "median"),
                    val_order_at_detected=("val_order", lambda s: float(s[sel.loc[s.index].detected].median())
                                           if sel.loc[s.index].detected.any() else np.nan))
               .reset_index())
    detected_layers = (sel[sel.detected].groupby("family").layer.apply(lambda s: ",".join(map(str, sorted(s)))))
    fam_tab["detected_at"] = fam_tab.family.map(detected_layers).astype(object).fillna("")
    lines = [f"# Label-free ring detection: {a.unit}", "",
             "Discover on half A of each item's contexts, confirm on half B; the true order is used only for the",
             "`val_order` column (validation). See src/topodur/ring.py. Exploratory: not part of the frozen analysis.", "",
             "## Calibration (dial = context group size x projection dim)", "",
             "Detection rate over all layers. Objective = planted_ring - max(line, planted_arc).", "",
             md_table(cal), "",
             f"Chosen dial: group={best[0]}, d={best[1]}.", "",
             "## Families at the chosen dial", "",
             md_table(fam_tab, index=False), "",
             "![detection](detection_best_dial.png)", ""]
    runs.atomic_write_text(rep_dir / "summary.md", "\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
