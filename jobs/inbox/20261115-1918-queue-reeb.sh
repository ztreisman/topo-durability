# Queue the token fork/rejoin study: version A (saved means; after the cross-layer stats), footprint GPU pass
# (dry run first), version B (per-token contexts).
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
J = [{"id": "treeb-mean", "gpus": 0, "retries": 1,
      "after": [f"xstats-B{i:02d}" for i in range(24)] + ["nstats-L22-23"],
      "cmd": f"{R} scripts/token_reeb.py --unit {U} --mode mean && {P} && "
             f"{N} \"token fork/rejoin (version A, means)\" \"$(grep 'Known answer' reports/neurons/{U}/reeb_mean.md | head -3)\" reports/neurons/{U}/reeb_mean.png"},
     {"id": "tfoot-dryrun", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/token_footprint.py --limit 1"},
     {"id": "tfoot", "gpus": 1, "retries": 1, "after": ["tfoot-dryrun"], "cmd": f"{R} scripts/token_footprint.py"},
     {"id": "treeb-ctx", "gpus": 0, "retries": 1, "after": ["tfoot", "nstats-L22-23"],
      "cmd": f"{R} scripts/token_reeb.py --unit {U} --mode ctx && {P} && "
             f"{N} \"token fork/rejoin (version B, contexts)\" \"$(grep 'Known answer' reports/neurons/{U}/reeb_ctx.md | head -3)\" reports/neurons/{U}/reeb_ctx.png"}]
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
