# lambda01 inbox

Push a script here as `YYYYMMDD-HHMM-name.sh`. lambda01 pulls every 10 minutes (jobs/inbox.sh, cron) and
runs each new script once, from the repo root, with jobs/env.sh sourced and a 2 h timeout. The result
(exit code + log tail) appears in `reports/inbox/<name>.txt` on GitHub and as an ntfy message.

Keep scripts short: queue long work for the runner (append to jobs/queue.jsonl or call a jobs/queue_*.sh).
Items present when jobs/install_inbox.sh ran are never executed. Pause on lambda01: `touch jobs/INBOX_OFF`.
