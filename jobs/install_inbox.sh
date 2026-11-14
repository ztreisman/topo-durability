#!/bin/bash
# One-time setup on lambda01 (run once while connected):  cd ~/topo-durability && git pull && bash jobs/install_inbox.sh
# Adds the jobs/inbox.sh cron entry (every 10 min) and marks inbox items already in the repo as seen,
# so only items pushed after installation ever run. Undo: crontab -e and delete the inbox line,
# or pause with: touch jobs/INBOX_OFF
set -e
cd "$(dirname "$0")/.."
here=$(pwd)
mkdir -p jobs/state/inbox jobs/inbox
for f in jobs/inbox/*.sh; do [ -e "$f" ] && date > "jobs/state/inbox/$(basename "$f" .sh)"; done
line="*/10 * * * * bash $here/jobs/inbox.sh"
if crontab -l 2>/dev/null | grep -qF "jobs/inbox.sh"; then
  echo "cron entry already present"
else
  (crontab -l 2>/dev/null; echo "$line") | crontab -
  echo "cron entry added: $line"
fi
bash jobs/notify.sh "lambda01 inbox installed" "polling GitHub every 10 min; results in reports/inbox/"
