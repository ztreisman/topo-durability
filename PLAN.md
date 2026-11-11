# Implementation plan

Companion to `../Topological Feature Durability Experiment Spec.md` (the spec). This file records how the
spec is being implemented, decisions taken where the spec left a choice, and open questions for Zack.
Results go in `RESULTS_LOG.md`, not here.

## Status (2026-11-10)

| Milestone | State |
| --- | --- |
| M0: environment + CPU tests | Built and passing on the laptop (23 tests, incl. pythia-70m extraction on CPU). Not yet run on lambda01: it was unreachable from off-site. |
| M1: replication gate | Pipeline built and smoke-tested end to end on pythia-70m. Waiting for the lambda01 run on 1.4B. |
| M2 to M7 | Not started. Roadmap below. |

## Phase 0: what to run on lambda01

Hugging Face is blocked on lambda01, so models and Pile text go laptop -> lambda01 through a plain
directory mirror (`data/hf_mirror` locally, `~/hf_mirror` on the server). Code goes through GitHub.

```bash
# on the laptop (any network)
uv run python scripts/fetch_hf.py models --sizes 70m 1.4b 2.8b --steps 143000   # ~9 GB
uv run python scripts/fetch_hf.py pile --n-docs 200000                          # ~1 GB
git push                                                                        # code

# on the laptop, on-site (lambda01 resolves only there)
bash scripts/push_data.sh                     # resumable; skips items already complete on the server

# on lambda01
git clone git@github.com:<you>/topo-durability.git ~/topo-durability   # first time; later: git pull
cd ~/topo-durability
bash jobs/setup_lambda.sh                     # uv, Python 3.12, torch cu126, offline env, check_env, pytest
uv run python scripts/make_jobs.py --config configs/phase0_replication.yaml --size 1.4b --steps 143000 --dry-run-first
bash jobs/start_runner.sh                     # tmux session "topodur-runner"
crontab -e                                    # add the two lines from jobs/start_runner.sh (reboot survival)
python3 jobs/runner.py status                 # the runner itself is stdlib-only
```

The queue runs: dry extract (8 contexts/item, 200 background texts) -> dry score -> full extract -> full
score -> report. The report lands in `reports/phase0_replication/1.4b_s0_step143000/gate.md` with figures;
`bash scripts/deploy.sh --pull` brings it back to the laptop. (`scripts/deploy.sh` without
`--pull` is a tar-over-ssh fallback for pushing code if GitHub is unreachable.)

`jobs/env.sh` sets `HF_HUB_OFFLINE=1` and `TOPODUR_HF_MIRROR`. With the mirror variable set, a
missing checkpoint fails immediately with a "fetch it on the laptop" message instead of trying the
network. Phase 3 needs the trajectory checkpoints first:
`fetch_hf.py models --sizes 1.4b --steps logspaced` (27 checkpoints, ~80 GB), plus
`--sizes 410m --seeds 1 2 3 4 5 6 7 8 9` for the seed measure.

Not verified: whether lambda01 reaches PyPI, download.pytorch.org, and astral.sh (needed by
`setup_lambda.sh`). If those are blocked as well, build a wheelhouse on the laptop
(`uv export` + `pip download --platform manylinux...`) and install from it.

If 1.4B fails the gate: `make_jobs.py --config configs/phase0_replication_2p8b.yaml --size 2.8b ...`.
Per the spec, if both fail, stop and decide.

**Expected cost:** extraction is about 7,250 short contexts plus 2,000 background documents, roughly
10 to 20 minutes on one 2080 Ti for 1.4B, plus a 2.9 GB checkpoint download. Scoring is
24 layers x 36 settings x 3 families, with 200 bootstraps, 200 Gaussian nulls, and 2,000 permutations each,
about 1 to 2 CPU hours on 16 workers.

## What exists

| Module | Contents |
| --- | --- |
| `models.py` | Load Pythia at any published step (fp16/fp32), PolyPythia seed ids, NaN/overflow check on a small batch with automatic fp32 fallback, the log-spaced checkpoint list (27 steps). |
| `concepts/` | Family YAML format (items, ground-truth order or positions, templates with filler slots). `template_contexts` gives every item the same set of distinct prefixes. Families written so far: days, months, and ordinals (line control). |
| `extract.py` | Block-output hooks, last-token and span-mean variants, per-item centroids, stored instances for the bootstrap, background mean + top-10 PCs (first 8 positions skipped for attention-sink tokens). |
| `geometry.py` | The four preprocessing steps, distance matrices, Procrustes, linear CKA, participation ratio. |
| `tda.py` | Ripser diagrams, scores (`h1_max`, `h1_ratio`, H2), Fasy-style bootstrap band, circular coordinates, circular rank correlation + permutation test, representative cycles. |
| `nulls.py` | Gaussian null with the family's PCA spectrum, frequency/token-count matched item sampler. |
| `scoring.py` | One tidy row per family x layer x preprocessing; the gate rule. |
| `runs.py` | Run dirs with config + provenance (git hash, versions, GPUs), atomic writes, `.done` markers, cache budget check (1.5 TB). |
| `jobs/runner.py` | Stdlib GPU queue: JSONL queue, `after` dependencies, retries, CPU slots, external-load detection, survives its own restart, STOP file. |
| `scripts/` | `check_env`, `extract`, `score`, `report`, `make_jobs`, `deploy.sh`. All support `--limit`/`--dry-run`; extract and score are resumable. |

## Decisions taken (flag any you disagree with)

1. **Standard, not deduplicated, Pythia.** PolyPythias (van der Wal et al. 2025, arXiv:2503.09543)
   trains on the standard Pile, so seed comparisons stay on the same data. Their extra seeds cover
   sizes 14M to 410M only, so **the seed measure (Phase 3.2) uses 410M** (approved). `fetch_hf.py`
   will confirm the seed repos exist when it downloads them.
2. **Pile source:** `monology/pile-uncopyrighted` (standard Pile minus Books3, BookCorpus2,
   OpenSubtitles, YTSubtitles, OWT2). The original standard-Pile mirror is no longer on the hub. It is
   used for the background statistics and, from Phase 1, for context mining.
3. **Circular coordinates are implemented in `tda.py`** (de Silva et al. 2011: persistent cocycle,
   integer lift, harmonic least squares), not taken from dreimac. Families are small, the algorithm is
   about 40 lines, and dreimac's API has changed between versions. Tests check that it recovers the order
   of a shuffled noisy circle and of 7 points. dreimac remains an optional cross-check.
4. **Gate rule:** at least one H1 class with lifetime > 2c, the circular coordinate in the *exact*
   cyclic order (any rotation or reflection), and a label-permutation p < 0.05. `order_score` is the
   Fisher-Lee-style circular rank correlation, max over the two orientations (1 = perfect).
5. **The bootstrap uses every context.** `store_instances` = contexts per item (250), so resampling
   draws from the same set that formed the centroid. The spec's 50 would overstate the band.
6. **Representative cycles** are the shortest loop near the class's birth radius that pairs with the
   cocycle. This is enough for Phase 2 plots. Phase 5 needs optimal cycles on graphs with thousands of
   nodes, so a library check (Gudhi, Oineus, HomCloud) is deferred to then; the function signature
   stays the same.
7. **Euclidean distance runs alongside cosine.** After step 4, cosine distance measures the angle
   around the family's own mean. That can create angular structure in a blob. The Gaussian null goes
   through the same pipeline, so it is controlled, but the Euclidean variant is in the Phase 0 grid as
   a check.

## Findings from the pythia-70m smoke run (not a gate result)

These come from 40 contexts per item and 30 bootstraps on CPU. They are worth knowing before the
1.4B run.

- Days and months pass the gate at some layers. The ordinals line control never does. The months
  centroids at layer 2 form a clean ring in the correct order (`reports/test_cpu/.../best_months.png`).
- **The Gaussian null does not reject for any family** (p = 0.13 to 0.71 on `h1_ratio`). With 7 to 12
  points, a Gaussian cloud with the same spectrum often has an H1 bar as long as the family's.
  `h1_ratio` has little power at this size. The ordering is what separates circles from lines here,
  not the bar length.
- With template contexts, every item shares the same prefixes, so early-layer structure may be mostly
  token-embedding geometry. The Pile-mined contexts in Phase 1 test this.

- **Closure ratio** (added after the decision below): distance(last item, first item) / median
  neighbour distance in the true order. Ordinals stay at 3.6 to 4.7 at every layer; days at 1.1 to
  1.5. Months is ordered perfectly at layers 0 to 2 but has a wide December -> January gap (3.8 at
  layer 0), and the gap closes by layers 4 and 5 (1.1, 0.95). In this tiny model, the closing edge
  appears deeper in the network than the rest of the ring. Watch for this on 1.4B: it bears on H-nov.

## Roadmap after the gate

| Phase | Build | Compute on lambda01 |
| --- | --- | --- |
| 1 | Remaining 14 families + 6 line controls (YAML). `mine.py` streams Pile contexts (case-sensitive whole word, item-ambiguity filters for May/March/etc., resumable). City longitude list (about 300 cities, from GeoNames). Torus order score per circle factor. | One unit per (family set, model): minutes each |
| 2 | Frequency-matched random sets: unigram counts over the Pile shard, contexts mined for random words. `tda_scores.parquet` across families. | CPU |
| 3 | `PREREGISTRATION.md` frozen first (M3). Trajectory: 27 checkpoints x 1.4B, one per GPU (`make_jobs --steps logspaced`). PolyPythia 410M seeds. `finetune.py` LoRA on code and non-English text, saves at 0/1/5/10/20M tokens, with held-out Pile perplexity. `compress.py`: bitsandbytes 8/4-bit (check sm_75 first), magnitude pruning, weight noise. `behavior.py`: offset queries with a wraparound split, subspace rotation and projection. | Trajectory about 1 GPU-day. LoRA: 2 corpora x about 20M tokens on 1.4B, about 1 to 2 GPU-days each. |
| 4 | `toy/`: TMS with the 6 correlation structures, ported from `arc-llc`. Sweep m x n x sparsity x 20 seeds. devinterp LLC. | CPU or 1 GPU per slice. Can start now, in parallel. |
| 5 | Check the hub for EleutherAI Pythia SAEs first. Otherwise sparsify TopK on 1.4B (timed on a short run first). NPMI co-activation graph, sparse flag-complex H1, two rewiring nulls, matched cycles/paths/cliques, ablations. | SAE: days per layer on a 2080 Ti (estimate before committing) |

Disk: 27 checkpoints x 2.9 GB of 1.4B weights is about 80 GB of HF cache. Activation outputs for
Phases 0 to 3 are centroids plus instances (well under 100 GB). Only Phase 5 approaches the 1.5 TB
budget.

## Decisions from Zack (2026-11-10)

1. **Primary topology measure:** `order_score` for families with at most 30 items
   (`scoring.order_primary_max_items`), and `h1_ratio` above that (cities). Written to each row as
   `topology_score` / `topology_measure`. The Gaussian null is reported but is not part of the gate.
   Caveat: an ordered *line* can also get a high order score (ordinals reached 0.93 at two layers in
   the smoke run). For the circle-versus-line contrast, `closure_ratio` is reported next to it.
   Proposal for the preregistration: the primary topology measure for small families is
   order_score x 1[closure_ratio < 2]. Not adopted until Zack agrees.
2. **Seed measure** uses PolyPythia 410M (seeds 1 to 9 plus the original run).
3. **Code on GitHub** (private repo); lambda01 pulls from it.
4. **Hugging Face is blocked on lambda01** -> the mirror workflow above.

## Open questions

1. Can lambda01 reach PyPI, download.pytorch.org, and astral.sh? Only Hugging Face is known to be
   blocked.
2. The order x closure proposal in decision 1.

## Citations

Verified so far: PolyPythias (van der Wal et al. 2025): uses the standard Pile, 9 new seeds, sizes up
to 410M. The other eleven in the spec's list are unchecked. They will be checked before any writeup
relies on them.
