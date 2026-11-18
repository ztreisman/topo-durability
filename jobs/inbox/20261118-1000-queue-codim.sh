# Codimension study (2026-11-18): first-order update overlap per concept (GPU, base model), then tests 1 and 2.
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
J = [{"id": "codim-dryrun", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/codim_pass.py --size 1.4b --max-concepts 3"},
     {"id": "codim-pass", "gpus": 1, "retries": 1, "after": ["codim-dryrun"], "cmd": f"{R} scripts/codim_pass.py --size 1.4b"},
     {"id": "codim-analysis", "gpus": 0, "retries": 0, "after": ["codim-pass"],
      "cmd": f"{R} scripts/codim_analysis.py --unit {U} --workers 32 && {P} && {N} \"codimension study\" "
             f"\"$(sed -n '/## Test 1/,/^Overlap/p' reports/neurons/{U}/codim.md | tail -6)\" reports/neurons/{U}/codim.png"}]
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
