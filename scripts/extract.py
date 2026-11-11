"""Extract background stats and family activations for one (model size, step, seed).

Outputs (in data/results/<experiment>/<size>_s<seed>_step<step>/):
  background.npz          mean + top PCs of the residual stream per layer
  acts_<family>.npz       per-item centroids and instance vectors per layer (template contexts)
  acts_<family>@pile.npz  same for mined Pile contexts (contexts.sources includes "pile"); only items
                          with at least contexts.pile_min_per_item mined contexts are kept
Each file gets a .done marker; finished files are skipped on rerun (resumable).

  uv run python scripts/extract.py --config configs/phase0_replication.yaml --size 1.4b --step 143000
  ... --limit 8 --dry-run      # small test pass: 8 contexts/item, 200 background texts, no .done markers
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topodur import runs  # noqa: E402
from topodur.concepts import load_family, template_contexts  # noqa: E402
from topodur.mine import load_mined  # noqa: E402
from topodur.extract import background_stats, extract_family  # noqa: E402
from topodur.models import load_model, unit_id  # noqa: E402
from topodur.pile import stream_texts  # noqa: E402

log = logging.getLogger("extract")


def acts_name(family: str, source: str) -> str:
    return f"acts_{family}.npz" if source == "template" else f"acts_{family}@{source}.npz"


def family_cx(cfg: dict, family: str) -> dict:
    """contexts settings with per-family overrides (contexts.overrides.<family>)."""
    cx = {k: v for k, v in cfg["contexts"].items() if k != "overrides"}
    cx.update((cfg["contexts"].get("overrides") or {}).get(family, {}))
    return cx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="PolyPythia seed (0 = original run)")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--variant", default=None, help="model variant: lora:<dir> | int8 | nf4 | prune:<f> | noise:<s>[:<seed>]")
    ap.add_argument("--limit", type=int, default=None, help="contexts per item (test pass)")
    ap.add_argument("--dry-run", action="store_true", help="write to <experiment>_dryrun, no .done markers")
    a = ap.parse_args()

    cfg = runs.load_config(a.config)
    exp = cfg["experiment"] + ("_dryrun" if a.dry_run or a.limit else "")
    dtype = a.dtype or cfg.get("dtype", "fp16")
    fams = a.families or cfg["families"]
    uid = unit_id(a.size, a.step, a.seed, a.variant)
    cfg_resolved = {**cfg, "unit": {"size": a.size, "step": a.step, "seed": a.seed, "dtype": dtype,
                                    "families": fams, "limit": a.limit,
                                    "variant": a.variant}}
    out = runs.start_run(exp, uid, cfg_resolved, seed=cfg.get("seed", 0))

    sources = cfg["contexts"].get("sources", ["template"])
    targets = [out / "background.npz"] + [out / acts_name(f, src) for f in fams for src in sources]
    if all(runs.is_done(t) or (t.name.endswith("@pile.npz") and (out / (t.name + ".skipped")).exists())
           for t in targets):
        log.info("all outputs done for %s; nothing to do", uid)
        return 0

    # Disk estimate: L * I * (store+1) * d * 2 bytes * 2 variants per family; checked against budget.
    t0 = time.time()
    lm = load_model(a.size, a.step, dtype=dtype, seed=a.seed or None, multi_gpu=a.multi_gpu, variant=a.variant)
    log.info("model loaded in %.0fs: %d layers, d=%d, dtype=%s", time.time() - t0, lm.n_layers, lm.d_model, lm.dtype)
    layers = cfg["extract"].get("layers")  # None = all layers
    n_l = len(layers) if layers else lm.n_layers
    est = sum(n_l * len(load_family(f).items) * (family_cx(cfg, f)["store_instances"] + 1) * lm.d_model * 4
              for f in fams) * len(sources)
    runs.check_cache_budget(0)  # results are not cache, but fail early if the disk is already over budget
    log.info("estimated output size %.2f GB", est / 1e9)

    bg_path = out / "background.npz"
    if not runs.is_done(bg_path):
        b = cfg["background"]
        n_texts = 200 if a.limit else b["n_texts"]
        bg = background_stats(lm, stream_texts(b["source"]), n_texts=n_texts, max_len=b["max_len"],
                              positions_per_text=b["positions_per_text"], k=b["k"],
                              batch_size=b.get("batch_size", 16), seed=cfg.get("seed", 0))
        bg.save(bg_path)
        if not a.dry_run:
            runs.mark_done(bg_path, {"n_vectors": bg.n_vectors})

    for f in fams:
        cx = family_cx(cfg, f)
        for src in sources:
            p = out / acts_name(f, src)
            if runs.is_done(p):
                log.info("skip %s/%s (done)", f, src)
                continue
            fam = load_family(f)
            if src == "template":
                ctxs = template_contexts(fam, cx["per_item"], seed=cfg.get("seed", 0))
            elif src == "pile":
                ctx_dir = Path(cx["pile_dir"])
                if not (ctx_dir / f"{f}.jsonl").exists():
                    raise FileNotFoundError(f"{ctx_dir}/{f}.jsonl missing: run scripts/mine.py first")
                fam, ctxs, kept = load_mined(ctx_dir, fam, cx.get("pile_min_per_item", 20),
                                             cx.get("pile_per_item", 500))
                if fam.n_items < cx.get("pile_min_items", 5):
                    log.warning("%s/pile: only %d items with enough contexts; skipped", f, fam.n_items)
                    (out / (p.name + ".skipped")).write_text(str(fam.n_items))
                    continue
            else:
                raise ValueError(f"unknown context source {src!r}")
            t1 = time.time()
            acts = extract_family(lm, ctxs, fam.items, layers=layers, batch_size=cfg["extract"]["batch_size"],
                                  max_len=cx["max_len"], store_instances=cx["store_instances"],
                                  seed=cfg.get("seed", 0), limit=a.limit)
            acts.meta.update(family=f, source=src, n_contexts=len(ctxs))
            acts.save(p)
            if not a.dry_run:
                runs.mark_done(p, {"n_contexts_per_item": int(acts.counts.min()), "n_items": fam.n_items})
            log.info("%s/%s: %d items x >=%d contexts in %.0fs", f, src, fam.n_items, int(acts.counts.min()),
                     time.time() - t1)
    log.info("unit %s complete", uid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
