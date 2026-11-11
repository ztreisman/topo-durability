# Preregistration: topological feature durability (Phase 3)

Status: **DRAFT**. It becomes frozen at the commit whose message starts with "FREEZE PREREGISTRATION",
made after the Phase 1/2 report (M2) and before any Phase 3 unit is scored. After freezing, this file is
never edited; later analyses go in the "Exploratory" section of RESULTS_LOG.md and the writeup.

Spec: `../Topological Feature Durability Experiment Spec.md`. Implementation decisions: PLAN.md.

## 1. Model and data

- Primary model: Pythia-1.4B, standard (non-deduplicated), final checkpoint step 143000 as the base.
  Replication of the trajectory on Pythia-2.8B is secondary. Seeds: PolyPythia 410M, seeds 0 (original)
  to 9, final checkpoint (no larger PolyPythia size has seeds).
- Families: the 22 families in `configs/phase3_durability.yaml`: 14 circle families, 2 torus families
  (keys, triads; scored on their first circle factor, the circle of fifths), 1 open family (political),
  and 7 line controls (ordinals, sizes, life_stages, centuries, temperature, city_latitude, numbers).
  `numbers` is an extra control (same tokens as clock_hours) and is reported but excluded from the
  primary contrast.
- Contexts: synthetic templates, 250 per item (64 for the two city families), context seed 1 in Phase 3.
  Mined Pile contexts are a Phase 1 robustness check only.
- Representation: residual stream after each block, last token of the item span.

## 2. Topology score (the predictor)

Primary preprocessing: center on background mean, remove 1 top background PC, unit-normalize, project
onto the family's top 5 PCs, cosine distance (`{k_remove: 1, d_proj: 5, metric: cosine, variant: last}`).

- Families of 30 items or fewer: **topology score = order_score x 1[closure_ratio < 2]**.
  order_score is the circular rank correlation between the circular coordinate of the most persistent
  H1 class and the ground-truth cyclic order (max over orientations, 1 = perfect). closure_ratio is
  D(last item, first item) / median D(consecutive items in the true order).
- Families over 30 items (cities_longitude, city_latitude): topology score = h1_ratio (largest H1
  death/birth ratio).
- A family "has a loop" at a layer (circle_pass) if it has at least one H1 class above the Fasy-style
  bootstrap band (lifetime > 2c, c = 95th percentile bottleneck distance over context resamples),
  its circular coordinate gives the exact cyclic order with label-permutation p < 0.05, and closure < 2.
  For the city families: n_sig_h1 >= 1 and closure < 2.

Layer: each family is analyzed at **L\***, the layer with the highest primary topology score in Phase 1
(1.4B, step 143000, template contexts, context seed 0; ties broken by h1_ratio, then the lower layer).
Because Phase 3 re-measures with context seed 1, retention is not biased by selecting on noise. The mean
over layers 12-16 is reported as a secondary, selection-free version.

## 3. Primary durability metric

**Retention after 20M tokens of unrelated LoRA fine-tuning**: for each family,
retention = topology score(fine-tuned) / topology score(base) at L\*, averaged over the two corpora
(Python code from codeparrot-clean; German Wikipedia). LoRA r=16, alpha=32, dropout 0.05, on the
attention and MLP projections of every block, lr 1e-4 cosine, 512-token sequences, batch 8, fp16
autocast. Families whose base score is 0 (no closed ordered loop at L\*) have undefined retention and are
excluded from retention analyses; they are counted and reported.

Held-out Pile perplexity (5000 Pile test documents, first 200 x 512 tokens) is reported at each save
(0, 1, 5, 10, 20M tokens). General damage is controlled by including log(perplexity ratio) as a
covariate when variants are pooled, and by reporting retention against perplexity.

Secondary durability measures (spec Phase 3): trajectory emergence step and post-emergence stability;
cross-seed agreement (Procrustes, CKA, share of seeds passing); retention under 8-bit and 4-bit
quantization, magnitude pruning (10, 25, 50%), and Gaussian weight noise (sigma = 0.01, 0.03, 0.1 x the
matrix's weight std); behavioral offset-query accuracy and log-probability margin, split into
wraparound and non-wraparound queries.

## 4. Primary contrasts and tests

1. **Circles vs line controls.** Unit: family. Compare mean retention of the circle families
   (excluding torus and political) with the six spec line controls. Effect: difference in mean
   retention, with a 95% bootstrap CI (10,000 resamples of families within each group). Also the
   mixed-effects model below with a circle indicator in place of the continuous score.
2. **Implicit vs explicit closure.** Among circle families: implicit (hues, cities_longitude) and
   implicit-long (krebs, anacyclosis) vs explicit. Same effect and CI.
3. Phases 4 and 5 (cycles vs paths vs cliques) are analyzed in their own reports with the same
   bootstrap CIs; they are not part of this file's decision rules beyond row 6 of the table below.

Mixed-effects model (statsmodels MixedLM), unit = family x layer x preprocessing at the 20M save, pooled
over the two corpora:

    retention ~ topology_score + log(token_freq) + n_items + participation_ratio + act_norm
                + (1 | layer) + (1 | preprocessing)

token_freq = mean over items of the item's whole-word count in the 1.2M-document Pile sample (mined
total counts). Continuous predictors are z-scored. The coefficient of interest is topology_score; its
95% CI comes from the model; families enter through their rows (no family random effect, because there
are too few families; a family-clustered bootstrap CI is reported alongside).

Multiple comparisons: Benjamini-Hochberg at q = 0.05 across families for per-family retention tests
(retention vs 1, bootstrap over contexts). Effect sizes with 95% CIs are the headline; p-values are
secondary.

## 5. Decision rules

| Result | Interpretation |
| --- | --- |
| Circle minus line retention CI includes 0, and the topology_score coefficient CI includes 0 | Topology per se not supported |
| The topology_score coefficient is positive alone but its CI includes 0 once log(token_freq) is in the model | H0: frequency explains it |
| Circle minus line retention > 0 (CI excludes 0) and the coefficient stays positive with covariates | Topology predicts durability |
| Toy models and Phase 5: cliques >= cycles > paths in absolute durability, no per-cost advantage for cycles | H-red |
| Toy models and Phase 5: cycles >= cliques per unit capacity cost, and cycles more common at smaller m | H-eff |
| Closing edges rare in the Pile, form late in training (closure ratio falls after the rest of the ring orders), are retained, and wraparound accuracy tracks them | H-nov |

"Forms late" for H-nov: in the trajectory, the first checkpoint where closure_ratio < 2 is later than the
first checkpoint where the family's items are ordered (order_score >= 0.8 with p < 0.05), at L\*.

## 6. What is fixed and what is not

Fixed now: the families and their item lists, the primary preprocessing, the topology score, the layer
rule, the fine-tuning recipe and corpora, the primary metric, the contrasts, the model formula, the
decision rules. Not fixed (exploratory): the 2.8B replication, the full preprocessing grid, mined-context
results, torus second-factor scores, subspace interventions, and anything added after the freeze.
