#!/bin/bash
# Phase 1 report digest to ntfy: per family best layer, score, closure, pass layers; then the
# summary and profile figure as attachments.   jobs/notify_phase1.sh <unit>
cd "$(dirname "$0")/.." || exit 0
u="$1"; r="reports/phase1_families/$u"
digest=$(grep '^| [a-z_]* | \(circle\|torus\|interval\|unknown\)' "$r/summary.md" | cut -d'|' -f2,6,7,8,11,12 \
         | sed 's/  */ /g; s/^ //' | head -c 3600)
bash jobs/notify.sh "Phase 1 report: $u" "family | best L | score | closure | pass layers | robust
$digest"
bash jobs/notify.sh "Phase 1 $u files" "summary + profiles for $u (attachments expire in a few hours)" \
     "$r/summary.md" "$r/profiles.png"
