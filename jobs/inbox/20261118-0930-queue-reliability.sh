# Measurement reliability study (RESULTS_LOG 2026-11-18): durability reliability now from the existing passes,
# topology split-half (CPU), replicate fine-tunes (seed 1), LoRA milestones and noise seeds, then the full report.
python3 - <<'PY'
import json
U = "1.4b_s0_step143000"; R = "uv run --frozen python"
N = "bash jobs/notify.sh"; P = "bash jobs/publish_results.sh"
MR = f"{R} scripts/measure_reliability.py --unit {U} --fp-dir cfoot"
V = {f"lora-{c}-{m}": f"--variant lora:data/finetune/1.4b_{c}/adapters/{m}" for c in ("code", "wiki_de") for m in ("1M", "5M", "10M")}
V.update({f"noise01-s{s}": f"--variant noise:0.1:{s}" for s in range(3)})
J = [{"id": "rel-dur0", "gpus": 0, "retries": 0, "cmd": f"{MR} --skip-topology && {P}"},
     {"id": "rel-topo", "gpus": 0, "retries": 1, "cmd": f"{MR} --workers 48 && {P}"}]
J += [{"id": f"ft-{c}-s1", "gpus": 1, "retries": 1,
       "cmd": f"{R} scripts/finetune.py --name {c}_s1 --corpus mirror:corpora/{f} --seed 1"}
      for c, f in (("code", "code_py_30000.jsonl"), ("wiki_de", "wiki_de_40000.jsonl"))]
J += [{"id": f"cdur-{k}", "gpus": 1, "retries": 1, "cmd": f"{R} scripts/token_durability.py --size 1.4b --fp-dir cfoot {v}"}
      for k, v in V.items()]
J += [{"id": f"cdur-lora-{c}_s1-20M", "gpus": 1, "retries": 1, "after": [f"ft-{c}-s1"],
       "cmd": f"{R} scripts/token_durability.py --size 1.4b --fp-dir cfoot --variant lora:data/finetune/1.4b_{c}_s1/adapters/20M"}
      for c in ("code", "wiki_de")]
J.append({"id": "rel-final", "gpus": 0, "retries": 0,
          "after": ["rel-topo"] + [f"cdur-{k}" for k in V] + [f"cdur-lora-{c}_s1-20M" for c in ("code", "wiki_de")],
          "cmd": f"{MR} --skip-topology && {P} && {N} \"measurement reliability\" "
                 f"\"$(grep -A8 'Composite' reports/neurons/{U}/reliability_cfoot.md | tail -4)\" reports/neurons/{U}/reliability_cfoot.png"})
have = {json.loads(l)["id"] for l in open("jobs/queue.jsonl") if l.strip() and not l.startswith("#")}
new = [j for j in J if j["id"] not in have]
with open("jobs/queue.jsonl", "a") as f:
    for j in new:
        f.write(json.dumps(j) + "\n")
print(len(new), "jobs queued")
PY
python3 jobs/runner.py status 2>&1 | grep -v "^done\|^pending"
