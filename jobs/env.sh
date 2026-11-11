# Environment for topo-durability on lambda01 (sourced by setup, the runner, and ~/.bashrc).
export PATH="$HOME/.local/bin:$PATH"
export TOPODUR_HF_MIRROR="${TOPODUR_HF_MIRROR:-$HOME/hf_mirror}"   # models + Pile sample (offline)
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1          # Hugging Face is blocked here; never try the network
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
