# Per-token durability vs sense topology (decided 2026-11-17): recover the footprint contexts (CPU), dry run on
# 4 tokens (base: activations must match the footprint), then one pass per model variant, then the analysis.
ls -d data/finetune/1.4b_code/adapters/20M data/finetune/1.4b_wiki_de/adapters/20M; df -h data | tail -1
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
V = {"base": "", "lora-code": "--variant lora:data/finetune/1.4b_code/adapters/20M --save-acts",
     "lora-wiki": "--variant lora:data/finetune/1.4b_wiki_de/adapters/20M --save-acts",
     "nf4": "--variant nf4 --save-acts", "noise": "--variant noise:0.03 --save-acts", "prune": "--variant prune:0.25 --save-acts"}
J = [{"id": "tctx", "gpus": 0, "retries": 1, "cmd": f"{R} scripts/token_contexts.py --size 1.4b"},
     {"id": "tdur-dryrun", "gpus": 1, "retries": 0, "after": ["tctx"],
      "cmd": f"{R} scripts/token_durability.py --size 1.4b --max-tokens-panel 4"}]
J += [{"id": f"tdur-{k}", "gpus": 1, "retries": 1, "after": ["tdur-dryrun"],
       "cmd": f"{R} scripts/token_durability.py --size 1.4b {v}"} for k, v in V.items()]
J.append({"id": "tdur-analysis", "gpus": 0, "retries": 0, "after": [f"tdur-{k}" for k in V],
          "cmd": f"{R} scripts/token_durability_analysis.py --unit {U} --senses eq224 && {P} && "
                 f"{N} \"sense topology vs durability\" \"$(sed -n '/## Joint/,/^$/p;/## Joint/,+8p' reports/neurons/{U}/durab_topology.md | head -9)\" "
                 f"reports/neurons/{U}/durab_topology.png"})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
