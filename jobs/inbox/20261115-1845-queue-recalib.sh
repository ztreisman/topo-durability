# Queue the closing-edge calibration (replaces the H1 calibration) for all 24 layers, then a fresh summary.
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
J = [{"id": f"ncal-L{l}", "gpus": 0, "retries": 1, "after": [f"nmine-L{l}"],
      "cmd": f"{R} scripts/neuron_mine.py --unit {U} --layer {l} --recalibrate"} for l in range(24)]
J.append({"id": "nsummary2", "gpus": 0, "retries": 0, "after": [f"ncal-L{l}" for l in range(24)] + ["nsummary"],
          "cmd": f"{R} scripts/neuron_mine.py --unit {U} --summarize && bash jobs/publish_results.sh && "
                 f"bash jobs/notify.sh \"neuron loops + closing-edge calibration\" \"$(grep -A3 'calls made' reports/neurons/{U}/summary.md | head -3)\""})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
