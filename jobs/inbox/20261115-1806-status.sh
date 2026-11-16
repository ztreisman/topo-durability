# Read-only status: runner, failures, SAE dry run, GPUs.
date; python3 jobs/runner.py status 2>&1 | grep -v "^done"
for f in $(ls -t jobs/logs/*.log | head -60); do grep -l "Traceback\|Error" $f; done 2>/dev/null | head
L=$(ls -t jobs/logs/gmine-dryrun* | head -1); echo "== $L"; tail -6 $L
for f in $(ls -t jobs/logs/xstats-B*.log 2>/dev/null | head -3); do echo "== $f"; tail -2 $f; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; free -g | sed -n 2p
