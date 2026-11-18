# Confirmatory durability test on the 969-concept panel (RESULTS_LOG 2026-11-17): footprint pass (dry run first),
# senses, durability passes (base, two 20M LoRA fine-tunes, 25% and 50% pruning), analysis.
ls -la ~/hf_mirror/pile/pile_mine_1000000.jsonl* 2>&1 | head -2; df -h data | tail -1
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
V = {"base": "", "lora-code": "--variant lora:data/finetune/1.4b_code/adapters/20M",
     "lora-wiki": "--variant lora:data/finetune/1.4b_wiki_de/adapters/20M",
     "prune25": "--variant prune:0.25", "prune50": "--variant prune:0.5"}
J = [{"id": "cfoot-dryrun", "gpus": 1, "retries": 0, "cmd": f"{R} scripts/concept_footprint.py --size 1.4b --limit 3"},
     {"id": "cfoot", "gpus": 1, "retries": 1, "after": ["cfoot-dryrun"], "cmd": f"{R} scripts/concept_footprint.py --size 1.4b"},
     {"id": "csenses", "gpus": 0, "retries": 1, "after": ["cfoot"],
      "cmd": f"{R} scripts/token_senses.py --unit {U} --fp-dir cfoot --n-ctx-eq 224 --workers 8 && {P}"},
     {"id": "cdur-dryrun", "gpus": 1, "retries": 0, "after": ["cfoot"],
      "cmd": f"{R} scripts/token_durability.py --size 1.4b --fp-dir cfoot --max-tokens-panel 4"}]
J += [{"id": f"cdur-{k}", "gpus": 1, "retries": 1, "after": ["cdur-dryrun"],
       "cmd": f"{R} scripts/token_durability.py --size 1.4b --fp-dir cfoot {v}"} for k, v in V.items()]
J.append({"id": "canalysis", "gpus": 0, "retries": 0, "after": ["csenses"] + [f"cdur-{k}" for k in V],
          "cmd": f"{R} scripts/token_durability_analysis.py --unit {U} --fp-dir cfoot --senses cfoot_eq224 && {P} && "
                 f"{N} \"CONFIRMATORY: concept topology vs durability\" \"$(grep 'Result:' reports/neurons/{U}/durab_topology_cfoot.md)\" "
                 f"reports/neurons/{U}/durab_topology_cfoot.png"})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
