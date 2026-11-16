# Cross-layer neuron loops: 1.4b_s0_step143000

One graph over the MLP neurons of all layers; edges within and across layers from one correlation type.
Good = chordless, >= 4 neurons, beats layered rewiring nulls (degree- and triangle-preserving). Random wiring is
far loopier than geometric graphs, so 'good' is near-impossible; the test that matters is the typology table.
Pure diamond = fork at one root neuron, two unrelated arms, rejoin at one target; span = layers from root to target.

| graph | neurons | edges (cross) | H1 | good | good pure diamonds | ... span >= 2 |
|---|---|---|---|---|---|---|
| ctx | 196608 | 2867873 (1819132) | 1000 | 0 | 0 | 0 |
| tok | 196608 | 5418975 (2837893) | 1000 | 0 | 0 | 0 |
| total | 196608 | 3852132 (2340117) | 1000 | 0 | 0 | 0 |

## Typology of the top cycles: real vs layered null (triangle-preserving; p = share of nulls >= real)

### ctx

| type | real | null mean | null 95% | p |
|---|---|---|---|---|
| diamond span 1 | 0 | 3.1 | 6.5 | 1.000 |
| diamond span 2 | 0 | 1.1 | 3.0 | 1.000 |
| diamond span 3-4 | 0 | 0.7 | 2.0 | 1.000 |
| diamond span 5-8 | 0 | 1.8 | 4.5 | 1.000 |
| diamond span >=9 | 0 | 12.2 | 15.0 | 1.000 |
| flat | 988 | 799.2 | 820.8 | 0.091 |
| pure span 3-4 | 0 | 0.1 | 0.5 | 1.000 |
| pure span 5-8 | 0 | 0.1 | 0.5 | 1.000 |
| pure span >=9 | 0 | 0.3 | 1.6 | 1.000 |
| zigzag | 12 | 181.4 | 197.9 | 1.000 |

### tok

| type | real | null mean | null 95% | p |
|---|---|---|---|---|
| diamond span 1 | 10 | 48.1 | 58.6 | 1.000 |
| diamond span 2 | 9 | 15.2 | 23.1 | 0.909 |
| diamond span 3-4 | 21 | 19.6 | 27.7 | 0.273 |
| diamond span 5-8 | 58 | 16.7 | 24.0 | 0.091 |
| diamond span >=9 | 132 | 40.6 | 49.5 | 0.091 |
| flat | 0 | 29.4 | 38.0 | 1.000 |
| pure span 3-4 | 2 | 0.0 | 0.0 | 0.091 |
| pure span 5-8 | 18 | 0.0 | 0.0 | 0.091 |
| pure span >=9 | 112 | 0.0 | 0.0 | 0.091 |
| zigzag | 638 | 830.4 | 846.5 | 1.000 |

### total

| type | real | null mean | null 95% | p |
|---|---|---|---|---|
| diamond span 1 | 5 | 33.1 | 37.0 | 1.000 |
| diamond span 2 | 1 | 8.2 | 11.6 | 1.000 |
| diamond span 3-4 | 8 | 13.5 | 19.5 | 1.000 |
| diamond span 5-8 | 14 | 19.7 | 23.5 | 1.000 |
| diamond span >=9 | 126 | 78.1 | 89.1 | 0.091 |
| flat | 82 | 25.4 | 40.2 | 0.091 |
| pure span 5-8 | 6 | 0.0 | 0.0 | 0.091 |
| pure span >=9 | 25 | 0.1 | 0.5 | 0.091 |
| zigzag | 733 | 821.9 | 837.5 | 1.000 |

## Chordless pure diamonds among the top cycles: residual-path check (medians)

| graph | n | median span | direct_pct | arm_pct | root r2_tok | target r2_tok | tok_share |
|---|---|---|---|---|---|---|---|
| tok | 132 | 14 | 0.62 | 0.75 | 0.26 | 0.19 | 3.24 |
| total | 31 | 13 | 0.70 | 0.96 | 0.25 | 0.19 | 1.45 |

Examples: crosslayer_<graph>_examples.md. ![spans](crosslayer.png)
