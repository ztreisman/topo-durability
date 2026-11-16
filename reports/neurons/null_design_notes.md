# Neuron-graph nulls: what was tried (2026-11-15)

Within-layer neuron loops were first tested against degree- and triangle-preserving rewiring. On 1.4B no
cycle beat it in any layer (null 95th-percentile score 4-16 vs real max 1.6-6; SAE L14: 82). Random wiring
is far loopier than any geometric graph, so that null asks the wrong question.

A geometric null (Gaussian cloud with the real points' principal variances) was tried in three forms on a
synthetic blob (800 points, 30 decaying dimensions) with a planted noisy ring of 40 points:

| form | blob false positives | planted ring found |
|---|---|---|
| top-15-neighbour flag complex, quantile filtration | 0 | only when nearly noise-free (~900 of 2,300 H1 classes never die: the pruning leaves holes unfilled) |
| global similarity threshold, same edge count | 0 | no: ring ranks 44-144 by lifetime, below blob holes |
| local scan (seed's neighbours, top-d PCs, ripser, Gaussian null) | 5-25% of seeds | 0-30%, no better than blob seeds (nearest neighbours in high dimension form a hollow shell) |

None is usable, so within-layer discovery is not wired to a new null yet; the rewiring numbers stay in the
reports, labelled as such. The calibration was replaced by a closing-edge test (scripts/neuron_mine.py,
closing_scores / calibrate; synthetic check tests/test_calib_closing.py: no wrong calls in 72 ring/line
families, no calls on random tuning). Cross-layer: the typology comparison against the layered null is the
test; individual loops are listed as examples only.
