#!/usr/bin/env bash
# Start the job runner in a detached tmux session if it is not already running.
# Safe to call repeatedly, so it doubles as a watchdog. To survive reboots without root:
#   crontab -e
#   @reboot      bash $HOME/topo-durability/jobs/start_runner.sh
#   */15 * * * * bash $HOME/topo-durability/jobs/start_runner.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/env.sh
SESSION=topodur-runner
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exit 0
fi
mkdir -p jobs/logs
tmux new-session -d -s "$SESSION" \
  "python3 jobs/runner.py run ${RUNNER_ARGS:-} 2>&1 | tee -a jobs/logs/runner.log"
echo "started tmux session $SESSION (attach: tmux attach -t $SESSION; status: python3 jobs/runner.py status)"
