"""Phase 4 sweep: train toy models for a grid of (structure, n, m, q), 20 seeds per cell trained as one
batch, then measure geometry, capacity, durability, and LLC (whole model + refined per substructure).
CPU only; parallel over cells; resumable (one parquet per cell with a .done marker).

  uv run python scripts/toy_sweep.py --config configs/phase4_toy.yaml --workers 48
  ... --limit 2 --dry-run      # 2 cells, short training, few SGLD steps

Output: data/results/<experiment>/cells/<cell>.parquet, one row per seed x substructure; then
scripts/toy_report.py aggregates.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from topodur import runs  # noqa: E402

log = logging.getLogger("toy_sweep")


def cell_id(c: dict) -> str:
    return f"{c['structure']}_n{c['n']}_m{c['m']}_q{c['q']:g}"


def run_cell(c: dict, opts: dict, out_dir: str) -> str:
    import torch

    torch.set_num_threads(1)
    from topodur.toy.measure import capacity, durability, geometry
    from topodur.toy.tms import ToyConfig, build_groups, sgld_llc, train

    path = Path(out_dir) / f"{cell_id(c)}.parquet"
    if runs.is_done(path):
        return f"skip {path.name}"
    t0 = time.time()
    cfg = ToyConfig(structure=c["structure"], n=c["n"], m=c["m"], q=c["q"], seeds=opts["seeds"],
                    steps=opts["steps"], batch=opts["batch"], lr=opts["lr"], seed0=opts.get("seed", 0))
    groups = build_groups(cfg)
    W, b, snaps = train(cfg, groups=groups, record_every=opts.get("record_every", 0))
    dur = durability(cfg, W, b, groups.sub, retrain_steps=opts["retrain_steps"], seed=cfg.seed0)
    W2, b2 = dur.pop("_retrained", (None, None))
    llc_whole = sgld_llc(W, b, cfg, groups, steps=opts["sgld_steps"], eps=opts["sgld_eps"],
                         gamma=opts["sgld_gamma"], seed=cfg.seed0)
    llc_sub = {k: sgld_llc(W, b, cfg, groups, mask_feats=f, steps=opts["sgld_steps"], eps=opts["sgld_eps"],
                           gamma=opts["sgld_gamma"], seed=cfg.seed0) for k, f in groups.sub.items()}
    # learning time of each substructure's geometry: first snapshot where its order score >= 0.8
    rows = []
    for s in range(cfg.seeds):
        Wn = W[s].numpy()
        for k, f in groups.sub.items():
            r = {**c, "cell": cell_id(c), "seed": s, "sub": k, "kind": groups.order[k] or "none",
                 "llc_whole": float(llc_whole[s]), "llc_refined": float(llc_sub[k][s])}
            r.update(geometry(Wn, f, groups.order[k], seed=s))
            r.update(capacity(Wn, f))
            for mname, arr in dur[k].items():
                r[mname] = float(arr[s])
            if W2 is not None and groups.order[k]:
                g2 = geometry(W2[s].numpy(), f, groups.order[k], seed=s, n_perm=200)
                r["order_score_retrain"] = g2.get("order_score", np.nan)
                r["h1_ratio_retrain"] = g2.get("h1_ratio", np.nan)
            if snaps and groups.order[k]:
                first = np.nan
                for t, Ws, _ in snaps:
                    g = geometry(Ws[s].numpy(), f, groups.order[k], seed=s, n_perm=0)
                    if g.get("order_score", 0) >= 0.8:
                        first = t
                        break
                r["order_emerge_step"] = first
            rows.append(r)
    df = pd.DataFrame(rows)
    with runs.atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    runs.mark_done(path, {"seconds": time.time() - t0})
    return f"{path.name} {time.time() - t0:.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="number of cells (test pass)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run or a.limit else "")
    out = runs.start_run(exp, "cells", cfg, seed=cfg.get("seed", 0))
    runs.setup_logging(out / "sweep.log")
    g = cfg["grid"]
    cells = [dict(structure=s, n=n, m=m, q=q) for s, n, m, q in
             itertools.product(a.structures or g["structures"], g["n"], g["m"], g["q"])]
    opts = dict(cfg["train"], **cfg["measure"])
    if a.limit:
        cells = cells[: a.limit]
        opts.update(steps=300, retrain_steps=100, sgld_steps=60, seeds=4, record_every=100)
    log.info("%d cells on %d workers -> %s", len(cells), a.workers, out)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_cell, c, opts, str(out)): c for c in cells}
        for k, f in enumerate(as_completed(futs)):
            log.info("[%d/%d] %s", k + 1, len(cells), f.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())
