#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LABEL="com.local.mlx-openai-tts"
PLIST_SRC="$ROOT_DIR/launchd/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing venv at $ROOT_DIR/.venv. Create it first:" >&2
  echo "  cd \"$ROOT_DIR\" && uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e .[dev]" >&2
  exit 1
fi

sed \
  -e "s|/Users/user/mlx-openai-tts|$ROOT_DIR|g" \
  -e "s|/Users/user|$HOME|g" \
  "$PLIST_SRC" > "$PLIST_DST"

launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_DST" || true
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

if ! launchctl print "gui/$UID_NUM/$LABEL" >/dev/null 2>&1; then
  echo "Failed to load LaunchAgent: $LABEL" >&2
  echo "Plist: $PLIST_DST" >&2
  exit 1
fi

echo "Installed and started LaunchAgent: $LABEL"
echo "Logs: $HOME/Library/Logs/mlx-openai-tts.log"
