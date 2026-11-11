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
from topodur.concepts import load_family, subset  # noqa: E402
from topodur.extract import Background, FamilyActs  # noqa: E402
from topodur.geometry import prep_grid  # noqa: E402
from topodur.scoring import circle_pass, gate_pass, score_layer  # noqa: E402

log = logging.getLogger("score")
_CACHE: dict = {}


def acts_name(family: str, source: str) -> str:
    return f"acts_{family}.npz" if source == "template" else f"acts_{family}@{source}.npz"


def family_for(acts: FamilyActs, name: str):
    """The family restricted to the items present in acts (mined sources drop rare items)."""
    fam = load_family(name)
    if acts.items == fam.items:
        return fam
    return subset(fam, [fam.items.index(x) for x in acts.items])


def _task(unit_dir: str, fam_name: str, source: str, li: int, prep, sc: dict, seed: int) -> dict:
    key = (unit_dir, fam_name, source)
    if key not in _CACHE:
        if len(_CACHE) > 4:
            _CACHE.clear()
        acts = FamilyActs.load(Path(unit_dir) / acts_name(fam_name, source))
        _CACHE[key] = (acts, Background.load(Path(unit_dir) / "background.npz"), family_for(acts, fam_name))
    acts, bg, fam = _CACHE[key]
    big = fam.n_items > sc.get("order_primary_max_items", 30)
    nb = sc.get("n_boot_large", sc["n_boot"]) if big else sc["n_boot"]
    ng = sc.get("n_gauss_large", sc["n_gauss"]) if big else sc["n_gauss"]
    r = score_layer(fam, acts, bg, li, prep, n_boot=nb, n_perm=sc["n_perm"], n_gauss=ng, seed=seed,
                    order_max_items=sc.get("order_primary_max_items", 30),
                    closure_max=sc.get("closure_max", 2.0)).row
    r["source"] = source
    r["gate_pass"] = gate_pass(r, sc.get("alpha", 0.05))
    r["circle_pass"] = circle_pass(r, sc.get("alpha", 0.05), sc.get("closure_max", 2.0))
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
    sources = cfg["contexts"].get("sources", ["template"])
    prim = [p for p in preps if all(getattr(p, k) == v for k, v in sc["primary"].items())]
    tasks = []
    for f in fams:
        for src in sources:
            path = unit_dir / acts_name(f, src)
            if not path.exists():
                log.warning("no %s; skipped", path.name)  # e.g. a mined family with too few items
                continue
            acts = FamilyActs.load(path)
            lis = list(range(len(acts.layers) if not a.limit else min(a.limit, len(acts.layers))))
            if sc.get("layers"):
                lis = [li for li in lis if acts.layers[li] in sc["layers"]]
            # full prep grid for template contexts; mined contexts get the primary prep only
            # unless scoring.pile_full_grid is set
            ps = preps if src == "template" or sc.get("pile_full_grid") else prim
            tasks += [(str(unit_dir), f, src, li, p) for li in lis for p in ps]
    log.info("%d scoring tasks on %d workers", len(tasks), a.workers)

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_task, *t, sc, cfg.get("seed", 0)) for t in tasks]
        for k, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if k % 100 == 0:
                log.info("scored %d/%d", k + 1, len(tasks))

    base, _, variant = a.unit.partition("+")
    size, s, step = base.split("_")
    df = pd.DataFrame(rows).assign(model=size, seed=int(s[1:]), step=int(step.removeprefix("step")),
                                   variant=variant or "base")
    df = df.sort_values(["family", "source", "layer", "variant", "metric", "k_remove", "d_proj"]
                        ).reset_index(drop=True)
    with runs.atomic_path(out) as tmp:
        df.to_parquet(tmp, index=False)
    if not a.dry_run:
        runs.mark_done(out, {"rows": len(df)})
    log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
