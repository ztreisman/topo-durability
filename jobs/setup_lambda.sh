#!/usr/bin/env bash
# One-time (idempotent) environment setup on lambda01. No root needed.
#   cd ~/topo-durability && bash jobs/setup_lambda.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. uv in ~/.local/bin (user-level)
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
uv --version

# 2. Python 3.12 managed by uv (system 3.10 is EOL and untouched)
uv python install 3.12

# 3. Hugging Face is blocked on lambda01: models and Pile text come from ~/hf_mirror, filled from the
#    laptop with scripts/fetch_hf.py + scripts/push_data.sh. Offline flags stop any hub calls.
source jobs/env.sh
mkdir -p "$TOPODUR_HF_MIRROR"
grep -q 'topo-durability/jobs/env.sh' "$HOME/.bashrc" 2>/dev/null ||   echo "source $PWD/jobs/env.sh  # topo-durability/jobs/env.sh" >> "$HOME/.bashrc"
[ -f "$TOPODUR_HF_MIRROR/models/EleutherAI/pythia-70m/step143000/.complete" ] ||   echo "WARNING: pythia-70m not in the mirror yet; the model tests will fail until push_data.sh has run"

# 4. Project venv: torch from the cu126 index (pyproject pins the index for Linux)
uv sync --python 3.12
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())"

# 5. Environment checks (Phase 0 steps 1-2) and CPU tests (step 3)
uv run python scripts/check_env.py --expect-gpus 8
uv run pytest -q
echo "setup complete"
