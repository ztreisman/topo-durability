"""Phase 2 scoring for one extraction unit -> tda_scores.parquet (tidy; one row per
family x layer x preprocessing). CPU only; parallel over tasks.

  uv run python scripts/score.py --config configs/phase0_replication.yaml --unit 1.4b_s0_step143000
  ... --limit 4 --dry-run    # first 4 layers, small bootstrap; reads the _dryrun experiment
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402
from topodur.concepts import load_family  # noqa: E402
from topodur.extract import Background, FamilyActs  # noqa: E402
from topodur.geometry import prep_grid  # noqa: E402
from topodur.scoring import gate_pass, score_layer  # noqa: E402

log = logging.getLogger("score")
_CACHE: dict = {}


def _task(unit_dir: str, fam_name: str, li: int, prep, sc: dict, seed: int) -> dict:
    key = (unit_dir, fam_name)
    if key not in _CACHE:
        _CACHE[key] = (FamilyActs.load(Path(unit_dir) / f"acts_{fam_name}.npz"),
                       Background.load(Path(unit_dir) / "background.npz"), load_family(fam_name))
    acts, bg, fam = _CACHE[key]
    r = score_layer(fam, acts, bg, li, prep, n_boot=sc["n_boot"], n_perm=sc["n_perm"],
                    n_gauss=sc["n_gauss"], seed=seed,
                    order_max_items=sc.get("order_primary_max_items", 30)).row
    r["gate_pass"] = gate_pass(r, sc.get("alpha", 0.05))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", required=True, help="e.g. 1.4b_s0_step143000")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="score only the first N layers, small nulls")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run or a.limit else "")
    unit_dir = runs.run_dir(exp, a.unit)
    runs.setup_logging(unit_dir / "score.log")
    out = unit_dir / "tda_scores.parquet"
    if runs.is_done(out) and not a.force:
        log.info("%s already scored", a.unit)
        return 0

    sc = dict(cfg["scoring"])
    if a.limit:
        sc.update(n_boot=20, n_perm=200, n_gauss=20)
    preps = prep_grid(sc["ks"], sc["ds"], sc["metrics"], sc["variants"])
    fams = a.families or cfg["families"]
    tasks = []
    for f in fams:
        acts = FamilyActs.load(unit_dir / f"acts_{f}.npz")
        n_layers = len(acts.layers) if not a.limit else min(a.limit, len(acts.layers))
        tasks += [(str(unit_dir), f, li, p) for li in range(n_layers) for p in preps]
    log.info("%d scoring tasks on %d workers", len(tasks), a.workers)

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_task, *t, sc, cfg.get("seed", 0)) for t in tasks]
        for k, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if k % 100 == 0:
                log.info("scored %d/%d", k + 1, len(tasks))

    size, s, step = a.unit.split("_")
    df = pd.DataFrame(rows).assign(model=size, seed=int(s[1:]), step=int(step.removeprefix("step")))
    df = df.sort_values(["family", "layer", "variant", "k_remove", "d_proj"]).reset_index(drop=True)
    with runs.atomic_path(out) as tmp:
        df.to_parquet(tmp, index=False)
    if not a.dry_run:
        runs.mark_done(out, {"rows": len(df)})
    log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
