# Test 3 (RESULTS_LOG 2026-11-18 night): persona-trait durability. GPU dry run, then one pass per variant, then
# the analysis (CPU, one thread per worker) with ntfy.
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
TP = f"{R} scripts/trait_pass.py --size 1.4b"
V = {"base": ""}
V.update({f"lora-{c}-{m}": f"--variant lora:data/finetune/1.4b_{c}/adapters/{m}"
          for c in ("code", "wiki_de") for m in ("1M", "5M", "10M", "20M")})
V.update({f"lora-{c}_s1-20M": f"--variant lora:data/finetune/1.4b_{c}_s1/adapters/20M" for c in ("code", "wiki_de")})
V.update({f"noise01-s{s}": f"--variant noise:0.1:{s}" for s in range(3)})
V.update({"prune25": "--variant prune:0.25", "prune50": "--variant prune:0.5"})
J = [{"id": "trait-dryrun", "gpus": 1, "retries": 0,
      "cmd": f"{TP} --max-traits 5 --per-trait 40 && {TP} --max-traits 5 --per-trait 40 --variant prune:0.25"
             f" && {R} scripts/trait_analysis.py --size 1.4b --dry --workers 4"}]
J += [{"id": f"trait-{k}", "gpus": 1, "retries": 1, "after": ["trait-dryrun"], "cmd": f"{TP} {v}"} for k, v in V.items()]
J.append({"id": "trait-analysis", "gpus": 0, "retries": 0, "after": [f"trait-{k}" for k in V],
          "cmd": f"{R} scripts/trait_analysis.py --size 1.4b --workers 48 && {P} && {N} \"persona-trait durability\" "
                 f"\"$(grep -A4 'Base model' reports/neurons/{U}/traits.md | tail -1)\" reports/neurons/{U}/traits.png"})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done"
