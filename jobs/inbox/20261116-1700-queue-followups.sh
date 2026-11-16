# Follow-ups to the 2026-11-15 results:
#   xmine100-<g>   cross-layer typology with 100 triangle-preserving layered nulls (was 10: p floor 0.091)
#   treeb-ctx-eq224  version B with the same 224 contexts for every token (frequency control)
#   tsenses-eq224    senses: clusters of each token's contexts per layer (gap statistic), tracked across layers
nproc; free -g | sed -n 2p
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
J = [{"id": f"xmine100-{g}", "gpus": 0, "retries": 1,
      "cmd": f"{R} scripts/neuron_xmine.py --unit {U} --graph {g} --null-kinds triangle --n-null 100 --workers 12 --force"}
     for g in ("tok", "total", "ctx")]
J.append({"id": "xsummary100", "gpus": 0, "retries": 0, "after": [f"xmine100-{g}" for g in ("tok", "total", "ctx")],
          "cmd": f"{R} scripts/neuron_xmine.py --unit {U} --summarize && {P} && {N} \"cross-layer loops, 100 nulls\" "
                 f"\"$(grep -A12 '^### tok' reports/neurons/{U}/crosslayer.md | grep pure)\" reports/neurons/{U}/crosslayer.png"})
J.append({"id": "treeb-ctx-eq224", "gpus": 0, "retries": 1,
          "cmd": f"{R} scripts/token_reeb.py --unit {U} --mode ctx --n-ctx-eq 224 && {P} && "
                 f"{N} \"token fork/rejoin, 224 contexts each\" \"$(grep 'Known answer\|Frequency' reports/neurons/{U}/reeb_ctx_eq224.md | head -6)\" reports/neurons/{U}/reeb_ctx_eq224.png"})
J.append({"id": "tsenses-eq224", "gpus": 0, "retries": 1,
          "cmd": f"{R} scripts/token_senses.py --unit {U} --n-ctx-eq 224 --workers 16 && {P} && "
                 f"{N} \"token senses (context clusters)\" \"$(grep 'Known answer\|Frequency' reports/neurons/{U}/senses_eq224.md)\" reports/neurons/{U}/senses_eq224.png"})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
