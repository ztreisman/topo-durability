#!/bin/bash
# Push a notification to ntfy.sh. Topic from NTFY_TOPIC (set in jobs/runner.local.sh, gitignored).
#   jobs/notify.sh "title" "message"                 plain message
#   jobs/notify.sh "title" "message" file [file...]  one message per file, with the file attached
# Attachments on ntfy.sh expire after a few hours; the message text is kept.
here="$(cd "$(dirname "$0")" && pwd)"
[ -z "$NTFY_TOPIC" ] && [ -f "$here/runner.local.sh" ] && . "$here/runner.local.sh"
[ -z "$NTFY_TOPIC" ] && exit 0
url="https://ntfy.sh/$NTFY_TOPIC"
title="$1"; msg="$2"; shift 2 || true
if [ $# -eq 0 ]; then
  curl -s -m 30 -o /dev/null -H "Title: $title" -H "Tags: microscope" --data-binary "$msg" "$url"
else
  for f in "$@"; do
    [ -f "$f" ] || continue
    curl -s -m 60 -o /dev/null -H "Title: $title" -H "Filename: $(basename "$f")" \
         -H "Message: $(echo "$msg" | tr '\n' ' ' | cut -c1-300)" -T "$f" "$url"
  done
fi
exit 0
