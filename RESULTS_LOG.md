# Results log

Dated entries, newest first.

## 2026-11-17: sense topology vs durability (exploratory) and the confirmatory test, specified before its data

- Hypothesis (Zack, 2026-11-17): splitting into several context clusters is normal (a concept anchored in
  many contexts). A concept whose clusters are more numerous, persist longer across layers, or fork and
  rejoin more (rank of H1) than comparable concepts is more durable. Long-term goal: the same test for persona
  vectors and other model-level traits.
- Exploratory, 229 single-token panel words (1.4B, 224 contexts each; `reports/neurons/1.4b_s0_step143000/
  durab_topology.md`). Durability = change in log p of the token in its own contexts after the 20M-token LoRA
  fine-tunes (code, German Wikipedia). Excess = residual after log frequency + class.
  - Excess b1 (strict fork-then-rejoin loops, containment >= 0.7): beta +0.19 [0.06, 0.33], p 0.003 in the
    joint model; same sign for each fine-tune and for 25% pruning.
  - Excess number of senses and excess persistence: no effect (longest bar slightly negative, p 0.047).
  - Durability is a weak per-token trait (code vs German fine-tune losses correlate 0.21).
- Confirmatory test, fixed now, before the data: 969 new concepts (`configs/concept_panel.json`; WordNet words
  of 1-4 tokens, frequency-stratified, disjoint from the exploratory panel), 224 contexts each from
  pile_mine. Same instrument (gap-statistic senses, 3 reference restarts, strict links 0.7). Excess =
  residual after log frequency + number of tokens. Primary: standardised coefficient of excess b1 in
  durability ~ excess senses + excess total persistence + excess b1 + log frequency + base log p
  + log WordNet synsets + number-of-tokens class; durability = mean of the two 20M LoRA fine-tunes.
  Confirmed if the 95% bootstrap CI is above 0. Secondary: 25% and 50% pruning, each measure alone.

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
