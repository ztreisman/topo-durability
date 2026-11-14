#!/bin/bash
# Queue the neuron-space Phase 5 sweep (MLP neurons, all 24 layers of 1.4B: within-layer and cross-layer
# loops) and the secondary SAE L14 graph mining. Run once on lambda01 after `git pull` (or from the inbox).
# Dry runs (1M tokens, layers 20-23 = full model depth, the GPU-memory worst case) gate the full jobs:
#   nstats-dryrun-a/b  within-layer stats, layers 20-21 and 22-23
#   xstats-dryrun      cross-layer stats, all 10 pairs among 20-23 incl. self-checks (l,l), which must
#                      reproduce the within-layer correlations or the job fails
#   nmine-dryrun / xmine-dryrun   CPU mining on the dry-run data
# Full sweep:
#   nstats-L<a>-<b>    2 layers per GPU job, 20M Pile tokens (also saves per-token means for xstats)
#   xstats-B<i>        12 layer pairs per GPU job, same token stream; all 276 pairs + self-check (12,12)
#   nmine-L<l>         within-layer loops, CPU          -> nsummary: report, GitHub, ntfy
#   xmine-<graph>      cross-layer fork/rejoin, CPU     -> xsummary: report, GitHub, ntfy
#   gmine-dryrun / gmine-L14   SAE secondary (scripts/graph_mine.py)
set -e
cd "$(dirname "$0")/.."
python3 - <<'EOF'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
J = [{"id": "nstats-dryrun-a", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/neuron_stats.py --layers 20 21 --limit 1"},
     {"id": "nstats-dryrun-b", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/neuron_stats.py --layers 22 23 --limit 1"},
     {"id": "xstats-dryrun", "gpus": 1, "retries": 0, "after": ["nstats-dryrun-a", "nstats-dryrun-b"],
      "cmd": f"{R} scripts/neuron_xstats.py --limit 1 --pairs " + " ".join(f"{l},{m}" for l in range(20, 24) for m in range(l, 24))},
     {"id": "nmine-dryrun", "gpus": 0, "retries": 0, "after": ["nstats-dryrun-b"],
      "cmd": f"{R} scripts/neuron_mine.py --unit {U} --layer 23 --limit --workers 8 && "
             f"{R} scripts/neuron_mine.py --unit {U} --summarize --limit"},
     {"id": "xmine-dryrun", "gpus": 0, "retries": 0, "after": ["xstats-dryrun"],
      "cmd": f"{R} scripts/neuron_xmine.py --unit {U} --limit --graph ctx --workers 4 && "
             f"{R} scripts/neuron_xmine.py --unit {U} --limit --summarize"}]
GATE = ["nmine-dryrun", "xmine-dryrun"]
nst = {}
for a in range(0, 24, 2):
    b = a + 1
    nst[a] = nst[b] = f"nstats-L{a}-{b}"
    J.append({"id": f"nstats-L{a}-{b}", "gpus": 1, "retries": 1, "after": GATE,
              "cmd": f"{R} scripts/neuron_stats.py --layers {a} {b} --tokens 20e6"})
    for l in (a, b):
        J.append({"id": f"nmine-L{l}", "gpus": 0, "retries": 1, "after": [f"nstats-L{a}-{b}"],
                  "cmd": f"{R} scripts/neuron_mine.py --unit {U} --layer {l} --workers 12"})
J.append({"id": "nsummary", "gpus": 0, "retries": 0, "after": [f"nmine-L{l}" for l in range(24)],
          "cmd": f"{R} scripts/neuron_mine.py --unit {U} --summarize && {P} && {N} \"neuron-space loops: all 24 layers\" "
                 f"\"$(sed -n 8,40p reports/neurons/{U}/summary.md | cut -c1-200)\" reports/neurons/{U}/neuron_layers.png"})
pairs = sorted([(l, m) for l in range(24) for m in range(l + 1, 24)] + [(12, 12)], key=lambda p: (p[1], p[0]))
nb = -(-len(pairs) // 12)                                   # <= 12 pairs per block, sizes as even as possible
cut = [round(i * len(pairs) / nb) for i in range(nb + 1)]
blocks = [pairs[cut[i]:cut[i + 1]] for i in range(nb)]
for i, bl in enumerate(blocks):
    J.append({"id": f"xstats-B{i:02d}", "gpus": 1, "retries": 1, "after": sorted({nst[l] for p in bl for l in p}),
              "cmd": f"{R} scripts/neuron_xstats.py --tokens 20e6 --pairs " + " ".join(f"{l},{m}" for l, m in bl)})
XS = [f"xstats-B{i:02d}" for i in range(len(blocks))]
for g in ("ctx", "tok", "total"):
    J.append({"id": f"xmine-{g}", "gpus": 0, "retries": 1, "after": XS,
              "cmd": f"{R} scripts/neuron_xmine.py --unit {U} --graph {g} --workers 8"})
J.append({"id": "xsummary", "gpus": 0, "retries": 0, "after": [f"xmine-{g}" for g in ("ctx", "tok", "total")],
          "cmd": f"{R} scripts/neuron_xmine.py --unit {U} --summarize && {P} && {N} \"cross-layer fork/rejoin loops\" "
                 f"\"$(sed -n 6,12p reports/neurons/{U}/crosslayer.md | cut -c1-200)\" reports/neurons/{U}/crosslayer.png"})
J.append({"id": "gmine-dryrun", "gpus": 1, "retries": 0,
          "cmd": f"{R} scripts/graph_mine.py --sae data/sae/1.4b_L14_32768 --layer 14 --limit 2"})
J.append({"id": "gmine-L14", "gpus": 1, "retries": 1, "after": ["gmine-dryrun"],
          "cmd": f"{R} scripts/graph_mine.py --sae data/sae/1.4b_L14_32768 --layer 14 && {P} && "
                 f"{N} \"SAE L14 graph mined\" \"$(cat data/sae/1.4b_L14_32768/graph/summary.json)\""})
have = set()
try:
    have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
except FileNotFoundError:
    pass
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(f"{len(new)} jobs queued ({len(J) - len(new)} already present); {len(blocks)} cross-layer GPU blocks")
EOF
bash jobs/notify.sh "neuron sweep queued" "dry runs first (layers 20-23 within + cross, SAE L14), then 24 layers within + 276 layer pairs; reports + ntfy at the end"
