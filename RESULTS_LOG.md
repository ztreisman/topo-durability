# Results log

Dated entries, newest first.

## 2026-11-19: test 3 results (`reports/neurons/1.4b_s0_step143000/traits.md`)

- Base: persona vectors are clean half-spaces (statement mean, layer 12: AUC 0.973 vs shrinkage LDA 0.996).
  Base behavior (Yes-No) barely follows the persona (median separation 0.14 log-odds; 4% of traits > 0.5).
- Damage is tiny: median retention -0.3% to -1% of d' for every LoRA, noise and 25% pruning; 50% pruning
  -10% at layer 12, -17% at layer 20.
- (1) Retention is reliable across traits (split-half 0.69-0.85). Within a corpus it is reproducible
  (milestones and seed replicates 0.5-0.83 disattenuated).
- (2) Across corpora agreement is higher than for concepts: code vs German 0.28-0.49 (concepts ~0.18).
  LoRA vs pruning 0-0.29, noise seeds ~0 (-0.08 to 0.20): same structure as concepts, larger shared part
  between the two fine-tunes.
- (3) Behavior does not follow representation (retention vs dbeh -0.10 to +0.14). dbeh is very reliable
  (0.96-0.99) but agrees +-0.5-0.9 across unrelated perturbations with variant-specific signs (noise s1 and
  50% pruning opposite to the rest): looks like one trait-level factor times a global Yes/No shift, not
  durability. Unchecked.
- Exploratory: traits with higher effective dimension have more consistent retention across perturbations
  (dispersion vs eff. dim -0.44); cleaner half-spaces less consistent (+0.27). For concepts the sign was the
  opposite (consistency fell with dimension). Drift along the trait axis does not predict damage.

## 2026-11-18 (night): test 3, persona-trait durability, specified before its data

- Data: Anthropic model-written evals, persona/ (135 traits, 133k statements, balanced matching / not
  matching; `data/traits/persona`, source commit in SOURCE.txt). Asked as "Is the following statement
  something you would say?" in Human/Assistant format; activations at layers 4/8/12/16/20 (mean over the
  statement, last token) and the Yes-No log-odds (`scripts/trait_pass.py`).
- Same 15 perturbations as the reliability study (LoRA code / German at 1M-20M, seed-1 replicates, three
  noise seeds, 25% and 50% pruning).
- Primary measure: retention = log(d' variant / d' base) along the base persona vector (diff of means, fit on
  the other half of the statements), layer 12, statement mean. Questions, in order: (1) is retention reliable
  (split-half across traits, seed replicates)? (2) does it agree across perturbations more than concept
  durability did (~0.2)? (3) does behavior (Yes-No) change with the representation? Exploratory: retention
  vs drift along the trait axis, half-space fit (codimension-1-ness), effective dimension.
  No directional hypothesis is being confirmed here; this is the measurement step.

## 2026-11-18 (evening): reliability study (`reports/neurons/1.4b_s0_step143000/reliability_cfoot.md`)

- Durability is reliable within a perturbation (split-half 0.88-0.99) and reproducible across fine-tune
  seeds (true correlation, seed 0 vs seed 1 at 20M: code 0.82, German 0.89). Across milestones it drifts
  (1M vs 20M: code 0.45, German 0.70).
- Across perturbations the true correlation is small: code vs German ~0.18, LoRA vs pruning ~0.2,
  German vs 50% pruning negative (-0.07 to -0.18). Gaussian noise (sd 0.1): three seeds agree only ~0.07,
  so damage from a random direction is specific to that direction. Durability is a property of
  concept x perturbation direction, with a small shared component.
- Topology is reliable too (split-half 0.58-0.87; total persistence 0.87, senses 0.83, b1 0.62 gap /
  0.67 stable). Ceiling for an observable topology-durability correlation: 0.73-0.89. The null confirmatory
  result is therefore not measurement noise: both sides are measurable, the link is small (~0.04).

## 2026-11-18 (later): codimension study (`reports/neurons/1.4b_s0_step143000/codim.md`)

- Test 1 (first-order, MLP-output update only, direct path): predicted vs measured damage across concepts
  +0.22 (code), +0.37 (German), +0.24 (prune 25%), -0.11 (prune 50%, too large for first order). Within a
  concept, across contexts: ~0.06. Predicted damages do NOT reproduce the measured ~0.2 agreement between
  perturbations (code/German predicted +0.05 vs measured +0.16). More overlap with the update goes with
  slightly LESS damage (sign opposite to expectation; possibly a norm/frequency confound).
- Test 2: high effective dimension (low codimension) concepts have more uniform overlap across updates
  (-0.14, as predicted) but damage is not more consistent: agreement between perturbations falls with
  dimension (high minus low tercile -0.09 [-0.17, -0.03]), opposite to the prediction. They are slightly
  less damaged on average (excess effective dimension vs mean damage +0.12), which may also shrink the
  spread that the agreement correlations need.

## 2026-11-18: confirmatory test NOT confirmed; next: measurement reliability

- 969-concept panel (968 with 224 contexts; `reports/neurons/1.4b_s0_step143000/durab_topology_cfoot.md`).
  Primary: excess b1 beta +0.037 [-0.026, +0.099], p 0.25: NOT confirmed (exploratory panel: +0.19).
- Secondary: excess b1 is positive under 25% and 50% pruning (+0.07 [0.01, 0.13], +0.05 [0.01, 0.08]);
  excess mean bar length is negative under every perturbation (-0.05 to -0.08, CIs exclude 0). Senses and
  total persistence: no effect. Everything explains < 1% of the variance.
- Per-concept durability agrees only ~0.2 between perturbations (code vs German fine-tune, 25% pruning,
  50% pruning), as on the exploratory panel. Reading: the exploratory effect was inflated by selection; any
  real link is small relative to the noise in both measures.
- Decided (Zack): find out what is being measured before testing further. Reliability study
  (`scripts/measure_reliability.py`): split-half reliability of durability and of every topology measure,
  replicate fine-tunes (seed 1), LoRA milestones, three noise seeds, and stable clusters (prediction
  strength). Also open: model-level traits (persona vectors) may be measurable where token and concept
  level topology/durability are noise.

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
