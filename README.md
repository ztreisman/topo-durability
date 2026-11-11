# topo-durability

Tests whether LLM features with nontrivial topology (loops, tori) are more durable than comparable
features without it. The testbed is Pythia, run on lambda01. The spec is
`../Topological Feature Durability Experiment Spec.md`; the implementation plan and open decisions are
in `PLAN.md`; results are in `RESULTS_LOG.md`.

## Laptop

```bash
uv sync                                   # Python 3.12 venv; CPU torch on Windows/macOS
uv run pytest                             # includes pythia-70m on CPU (about 160 MB download)
uv run python scripts/extract.py --config configs/test_cpu.yaml --size 70m --step 143000
uv run python scripts/score.py   --config configs/test_cpu.yaml --unit 70m_s0_step143000
uv run python scripts/report.py  --config configs/test_cpu.yaml --unit 70m_s0_step143000
bash scripts/deploy.sh                    # push code to lambda01 (--pull fetches reports back)
```

## Getting models and data to lambda01 (Hugging Face is blocked there)

```bash
uv run python scripts/fetch_hf.py models --sizes 70m 1.4b 2.8b --steps 143000   # laptop -> data/hf_mirror
uv run python scripts/fetch_hf.py pile --n-docs 200000
bash scripts/push_data.sh                 # on-site: laptop -> lambda01:~/hf_mirror (resumable)
```

## lambda01

```bash
git clone git@github.com:<you>/topo-durability.git && cd topo-durability   # later: git pull
bash jobs/setup_lambda.sh                 # once: uv, Python 3.12, torch cu126, offline env, checks, tests
uv run python scripts/make_jobs.py --config configs/phase0_replication.yaml --size 1.4b --steps 143000 --dry-run-first
bash jobs/start_runner.sh                 # runner in tmux; add its cron lines to survive reboots
python3 jobs/runner.py status
```

Jobs are lines in `jobs/queue.jsonl` (`id`, `cmd`, `gpus`, `after`, `retries`). The runner gives each
job free GPUs through `CUDA_VISIBLE_DEVICES`, skips GPUs other users are loading, retries failures, and
re-adopts running jobs after it restarts. `touch jobs/STOP` holds new launches.

Outputs: `data/results/<experiment>/<size>_s<seed>_step<step>/` (config, provenance, npz activations,
`tda_scores.parquet`) and `reports/<experiment>/<unit>/gate.md`.

## Hardware rules (2080 Ti, sm_75)

fp16 or fp32 only; never bf16 or FlashAttention 2. Torch comes from the cu126 index (never above
CUDA 12.8). `load_model` checks every new model and step for NaN/overflow and falls back to fp32.
