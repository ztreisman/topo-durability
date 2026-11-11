"""Mine natural Pile contexts for concept-family items (Phase 1). CPU only.

  uv run python scripts/mine.py --source mirror:pile_mine_1000000.jsonl --out data/contexts/pile_v1
  ... --limit-docs 2000 --dry-run     # quick pass on the first 2000 docs, writes to <out>_dryrun

Writes <out>/<family>.jsonl (one context per line) and <family>.summary.json (contexts and total
occurrences per item). Several --source files are mined in turn and merged (per-item cap applies
across them). Families are re-mined only if their output is missing (or with --force).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # many single-threaded worker processes; BLAS threads would oversubscribe

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topodur import runs  # noqa: E402
from topodur.concepts import list_families, load_family  # noqa: E402
from topodur.mine import MineOpts, mine_file, save_mined  # noqa: E402
from topodur.pile import resolve_source  # noqa: E402

log = logging.getLogger("mine")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--max-per-item", type=int, default=500)
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="alias of --limit-docs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    limit = a.limit_docs or a.limit
    out = Path(a.out + ("_dryrun" if a.dry_run or limit else ""))
    names = a.families or list_families()
    todo = [n for n in names if a.force or not (out / f"{n}.summary.json").exists()]
    if not todo:
        log.info("all families mined in %s", out)
        return 0
    opts = MineOpts(max_per_item=a.max_per_item, per_doc=a.per_doc)
    found = {n: defaultdict(list) for n in todo}
    totals = {n: Counter() for n in todo}
    nd = nc = 0
    for src in a.source:
        path = resolve_source(src)
        t0 = time.time()
        f, t, d, c = mine_file(path, todo, opts, workers=a.workers, limit_docs=limit)
        log.info("%s: %d docs, %.2f GB text in %.0fs", path, d, c / 1e9, time.time() - t0)
        nd += d
        nc += c
        for n in todo:
            totals[n].update(t[n])
            for i, lst in f[n].items():
                found[n][i].extend(lst[: max(0, opts.max_per_item - len(found[n][i]))])
    for n in todo:
        fam = load_family(n)
        save_mined(out, fam, found[n], totals[n], nd, nc, " + ".join(a.source))
        cnt = [len(found[n].get(i, [])) for i in range(fam.n_items)]
        log.info("%-18s contexts/item min %d median %d max %d; items with >=20: %d/%d", n, min(cnt),
                 sorted(cnt)[len(cnt) // 2], max(cnt), sum(c >= 20 for c in cnt), fam.n_items)
    (out / "_run.json").write_text(json.dumps({"sources": a.source, "opts": vars(opts), "n_docs": nd,
                                               "provenance": runs.provenance(None)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
