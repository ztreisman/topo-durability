#!/usr/bin/env bash
# Copy the local HF mirror (models + Pile sample, built by scripts/fetch_hf.py) to lambda01:~/hf_mirror.
# Item by item, skipping anything already complete on the server, so an interrupted push resumes.
#   bash scripts/push_data.sh            # push everything complete locally
#   bash scripts/push_data.sh --dry-run  # list what would be sent
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=ssh
[ -x /c/Windows/System32/OpenSSH/ssh.exe ] && SSH=/c/Windows/System32/OpenSSH/ssh.exe
HOST=${LAMBDA_HOST:-lambda01}
REMOTE=${LAMBDA_MIRROR:-hf_mirror}
LOCAL=${TOPODUR_HF_MIRROR:-data/hf_mirror}
DRY=${1:-}

[ -d "$LOCAL" ] || { echo "no local mirror at $LOCAL; run scripts/fetch_hf.py first"; exit 1; }
remote_have=$("$SSH" "$HOST" "cd $REMOTE 2>/dev/null && find . -name '.complete' -o -name '*.complete'" || true)

send() {  # $1 = path relative to the mirror root, $2 = its marker (relative)
  if grep -qxF "./$2" <<<"$remote_have"; then echo "skip  $1"; return; fi
  if [ "$DRY" = "--dry-run" ]; then echo "would send $1 ($(du -sh "$LOCAL/$1" | cut -f1))"; return; fi
  echo "send  $1 ($(du -sh "$LOCAL/$1" | cut -f1))"
  # payload first, marker last, so a broken transfer is never marked complete
  tar cf - -C "$LOCAL" --exclude='.complete' --exclude='*.complete' --exclude='.cache' "$1" \
    | "$SSH" "$HOST" "mkdir -p $REMOTE && tar xf - -C $REMOTE"
  tar cf - -C "$LOCAL" "$2" | "$SSH" "$HOST" "tar xf - -C $REMOTE"
}

cd "$LOCAL"
while IFS= read -r marker; do
  marker=${marker#./}
  if [[ "$marker" == */.complete ]]; then send "${marker%/.complete}" "$marker"
  else send "${marker%.complete}" "$marker"; fi
done < <(find . -name '.complete' -o -name '*.jsonl.complete' | sort)
cd - >/dev/null
echo "done"
