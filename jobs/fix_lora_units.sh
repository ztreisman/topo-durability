#!/bin/bash
# One-shot repair (2026-11-12): LoRA unit ids used to omit the corpus, so code and wiki_de adapters
# collided on "+lora-<tokens>" and only the code variants were queued. Renames the existing (code)
# units to "+lora-code-<tokens>" and queues the wiki_de variants (extraction + eval only).
set -e
cd "$(dirname "$0")/.."
source jobs/env.sh
for t in 1M 5M 10M 20M; do
  for exp in phase3_durability; do
    d=data/results/$exp/1.4b_s0_step143000+lora-$t
    if [ -d "$d" ]; then
      grep -q "1.4b_code/adapters/$t" "$d/provenance.json" && mv "$d" "data/results/$exp/1.4b_s0_step143000+lora-code-$t" \
        && echo "renamed $d -> +lora-code-$t"
    fi
  done
done
V=""; for c in code wiki_de; do for t in 1M 5M 10M 20M; do V="$V lora:data/finetune/1.4b_$c/adapters/$t"; done; done
uv run --frozen python scripts/make_jobs.py --config configs/phase3_durability.yaml --size 1.4b --steps 143000 \
    --extract-only --eval --variant $V
echo "code units are re-queued under the new ids but their extraction is already .done, so they finish at once"
