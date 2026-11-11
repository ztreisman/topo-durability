#!/usr/bin/env bash
# Laptop -> lambda01: copy the code (not data/, .venv/, jobs/state, jobs/logs) to ~/topo-durability.
# Uses Windows OpenSSH when present (its agent holds the key); plain ssh elsewhere.
#   bash scripts/deploy.sh            # push code
#   bash scripts/deploy.sh --pull     # fetch reports/ and RESULTS_LOG.md back
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=ssh
[ -x /c/Windows/System32/OpenSSH/ssh.exe ] && SSH=/c/Windows/System32/OpenSSH/ssh.exe
HOST=${LAMBDA_HOST:-lambda01}
DEST=${LAMBDA_DIR:-topo-durability}

if [ "${1:-}" = "--pull" ]; then
  "$SSH" "$HOST" "cd $DEST && tar czf - reports RESULTS_LOG.md 2>/dev/null" | tar xzf - -C .
  echo "pulled reports/ from $HOST:$DEST"
  exit 0
fi

tar czf - --exclude=./data --exclude=./.venv --exclude=./jobs/state --exclude=./jobs/logs \
  --exclude=./jobs/queue.jsonl --exclude='__pycache__' --exclude=./.pytest_cache --exclude=./reports . \
  | "$SSH" "$HOST" "mkdir -p $DEST && tar xzf - -C $DEST"
echo "deployed to $HOST:$DEST (queue, state, data, reports on the server were left alone)"
