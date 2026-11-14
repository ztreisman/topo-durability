#!/bin/bash
# Commit small results from lambda01 and push them to GitHub (deploy key, see README notes below).
# Runs hourly from cron and when the runner goes idle; a no-op when nothing changed.
# Pushed: reports/ (markdown, figures, csv), and per unit tda_scores.parquet, eval.json,
# behavior.parquet, provenance.json, config.yaml; SAE / toy logs and summaries. NOT pushed:
# activations (acts_*.npz, background.npz), model weights, adapters, SAE weights (too large).
cd "$(dirname "$0")/.." || exit 1
exec 9>/tmp/topodur_publish.lock; flock -n 9 || exit 0
log=jobs/logs/publish.log; mkdir -p jobs/logs
{
  echo "== $(date)"
  git add -A reports/ 2>/dev/null
  find data/results -maxdepth 3 \( -name tda_scores.parquet -o -name ring_scores.parquet -o -name eval.json -o -name behavior.parquet \
       -o -name provenance.json -o -name config.yaml -o -name '*.md' \) -size -50M -print0 2>/dev/null \
       | xargs -0 -r git add -f
  find data/results -maxdepth 3 -path '*cells*' -name '*.parquet' -size -20M -print0 2>/dev/null | xargs -0 -r git add -f
  find data/sae -maxdepth 3 \( -name train_log.jsonl -o -name cfg.json -o -name summary.json -o -name '*.parquet' \
       -o -name '*.md' \) -size -50M -print0 2>/dev/null | xargs -0 -r git add -f
  find data/neurons -maxdepth 2 \( -name 'mine_L*.json' -o -name 'closing_L*.parquet' -o -name 'calib_L*.parquet' \
       -o -name 'cycles_L*.parquet' -o -name 'calib.json'        -o -name 'xmine_*.json' -o -name 'xcycles_*.parquet' -o -name 'xnull_*.parquet' \) -size -20M -print0 2>/dev/null | xargs -0 -r git add -f
  find data/finetune -maxdepth 2 \( -name train_log.jsonl -o -name ft_config.json -o -name '*.json' \) -size -20M \
       -print0 2>/dev/null | xargs -0 -r git add -f
  if git diff --cached --quiet; then echo "nothing new"; else
    git commit -q -m "results from lambda01 $(date '+%Y-%m-%d %H:%M')" && echo committed
  fi
  if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && echo pushed \
      || echo "push failed (deploy key not added yet?)"
  fi
} >> "$log" 2>&1
