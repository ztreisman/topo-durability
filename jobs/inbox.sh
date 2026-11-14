#!/bin/bash
# GitHub inbox for lambda01: lets work be queued from off-site by pushing a script to jobs/inbox/.
# Cron runs this every 10 minutes (installed by jobs/install_inbox.sh). Each run:
#   1. pulls origin/main (rebase onto any unpushed local results commits, like publish_results.sh)
#   2. runs every jobs/inbox/*.sh not yet seen, AT MOST ONCE, oldest name first, from the repo root,
#      with jobs/env.sh sourced and a 2 h timeout. Inbox scripts should be short: queue long work into
#      jobs/queue.jsonl for the runner rather than doing it inline.
#   3. writes reports/inbox/<name>.txt (exit code + log tail), publishes it to GitHub, sends an ntfy.
# Seen state: jobs/state/inbox/<name> (gitignored). Kill switch: touch jobs/INBOX_OFF on lambda01.
# Trust: anyone who can push to main can run code here as this user. Push access is the owner plus
# lambda01's deploy key.
cd "$(dirname "$0")/.." || exit 1
[ -f jobs/INBOX_OFF ] && exit 0
exec 8>/tmp/topodur_inbox.lock; flock -n 8 || exit 0
source jobs/env.sh
mkdir -p jobs/logs jobs/state/inbox reports/inbox
log=jobs/logs/inbox.log

if [ "$1" != "--run" ]; then
  # Stage 1: pull, then re-exec the (possibly updated) script so bash never runs a file changed under it.
  {
    echo "== $(date) pull"
    if ! git pull -q --rebase --autostash origin main; then
      echo "pull failed"; git rebase --abort 2>/dev/null
      if [ ! -f jobs/state/inbox/.pull_failed ]; then
        touch jobs/state/inbox/.pull_failed
        bash jobs/notify.sh "lambda01 inbox: git pull failed" "see jobs/logs/inbox.log; inbox paused until a pull succeeds"
      fi
      exit 1
    fi
    rm -f jobs/state/inbox/.pull_failed
  } >> "$log" 2>&1 || exit 1
  flock -u 8
  exec bash jobs/inbox.sh --run
fi

ran=0
for f in $(ls jobs/inbox/*.sh 2>/dev/null | sort); do
  name=$(basename "$f" .sh)
  [ -e "jobs/state/inbox/$name" ] && continue
  date > "jobs/state/inbox/$name"            # mark first: at most once, even if it crashes
  out=jobs/logs/inbox_$name.log
  echo "== $(date) run $name ($(git log -1 --format='%h %an' -- "$f"))" >> "$log"
  timeout 2h bash "$f" > "$out" 2>&1; rc=$?
  echo "   exit $rc" >> "$log"
  { echo "inbox item: $name"; echo "commit: $(git log -1 --format='%h %ad' -- "$f")"
    echo "ran: $(cat "jobs/state/inbox/$name")  finished: $(date)"; echo "exit: $rc"; echo; echo "--- last 60 lines of output"
    tail -n 60 "$out"; } > "reports/inbox/$name.txt"
  bash jobs/notify.sh "lambda01 inbox: $name exit $rc" "$(tail -n 8 "$out" | cut -c1-200)"
  ran=1
done
[ $ran = 1 ] && bash jobs/publish_results.sh
exit 0
