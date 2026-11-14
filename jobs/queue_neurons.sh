#!/bin/bash
# Queue the neuron-space Phase 5 sweep (MLP neurons, all 24 layers of 1.4B) and the secondary SAE L14
# graph mining. Run once on lambda01 after `git pull`. Dry runs gate the full jobs.
#   nstats-dryrun / nmine-dryrun   layers 22-23 at 1M tokens (full model: worst case for GPU memory)
#   nstats-L<a>-<b>                 2 layers per GPU job, 20M Pile tokens
#   nmine-L<l>                      CPU, 12 workers each; nsummary -> report, GitHub, ntfy
#   gmine-dryrun / gmine-L14        SAE secondary (scripts/graph_mine.py)
set -e
cd "$(dirname "$0")/.."
python3 - <<'EOF'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
J = [{"id": "nstats-dryrun", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/neuron_stats.py --layers 22 23 --limit 1"},
     {"id": "nmine-dryrun", "gpus": 0, "retries": 0, "after": ["nstats-dryrun"],
      "cmd": f"{R} scripts/neuron_mine.py --unit {U} --layer 23 --limit --workers 8 && "
             f"{R} scripts/neuron_mine.py --unit {U} --summarize --limit"}]
for a in range(0, 24, 2):
    b = a + 1
    J.append({"id": f"nstats-L{a}-{b}", "gpus": 1, "retries": 1, "after": ["nmine-dryrun"],
              "cmd": f"{R} scripts/neuron_stats.py --layers {a} {b} --tokens 20e6"})
    for l in (a, b):
        J.append({"id": f"nmine-L{l}", "gpus": 0, "retries": 1, "after": [f"nstats-L{a}-{b}"],
                  "cmd": f"{R} scripts/neuron_mine.py --unit {U} --layer {l} --workers 12"})
J.append({"id": "nsummary", "gpus": 0, "retries": 0, "after": [f"nmine-L{l}" for l in range(24)],
          "cmd": f"{R} scripts/neuron_mine.py --unit {U} --summarize && {P} && {N} \"neuron-space loops: all 24 layers\" "
                 f"\"$(sed -n 8,40p reports/neurons/{U}/summary.md | cut -c1-200)\" reports/neurons/{U}/neuron_layers.png"})
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
print(f"{len(new)} jobs queued ({len(J) - len(new)} already present)")
EOF
bash jobs/notify.sh "neuron sweep queued" "dry runs first (layers 22-23, SAE L14), then 24 layers; report + ntfy at the end"
