# Inbox round-trip test: read-only status report.
echo "host: $(hostname)  commit: $(git log -1 --format='%h %s')"
python3 jobs/runner.py status 2>&1 | grep -v "^done" | head -30
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
