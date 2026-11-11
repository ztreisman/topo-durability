# Results log

Dated entries, newest first.

## 2026-11-10 (later): decisions, offline mirror, closure ratio

- Decided with Zack: order_score is the primary topology measure for families of 30 items or fewer
  (h1_ratio above). Seeds come from PolyPythia 410M. Code goes on GitHub. Hugging Face is blocked on
  lambda01.
- Offline workflow built (`fetch_hf.py`, `push_data.sh`, `jobs/env.sh`, `topodur/mirror.py`).
  Tested on the laptop with all HF network access disabled: the tests pass, a step-1000 checkpoint
  loads from the mirror, and a missing checkpoint or a hub stream fails with a clear message.
- `closure_ratio` added. On the 70m smoke data (primary prep):
  - ordinals: 3.6 to 4.7 at every layer.
  - days: 1.1 to 1.5.
  - months: order 1.00 but closure 3.8 at layer 0, falling to 1.10 and 0.95 at layers 4 and 5. The
    ring's closing edge appears deeper in the network. This is a smoke result only.
- 24 tests passed, 1 skipped (the POSIX-only runner test).

## 2026-11-10: M0 (laptop), Phase 0 pipeline smoke test

- Repo scaffolded. Laptop: `uv sync` (Python 3.12.14, torch 2.15.0 CPU). The lock resolves
  torch 2.14.1+cu126 for Linux.
- `pytest`: 23 passed, 1 skipped. The skip is the runner test, which is POSIX-only and runs on
  lambda01. Covered: synthetic circle (one H1), Clifford torus (two H1, one H2), circular coordinates
  recovering a shuffled order, the bootstrap band flagging a real loop, and pythia-70m extraction
  (shapes, centroid = instance mean, padding invariance, background PCs).
- End-to-end CLI on pythia-70m (`configs/test_cpu.yaml`: 40 contexts/item, 30 bootstraps). This is
  a smoke test, not a gate result.
  - days: passes the gate at layers 3 and 4. months: passes at layers 0 and 2, where the ring is in
    the correct order. ordinals (line control): never passes; no significant H1 at any layer.
  - The Gaussian null does not reject `h1_ratio` for any family (p 0.13 to 0.71). See PLAN.md, Q1.
- lambda01 was not reachable from off-site, so M0 on GPUs is still pending.
