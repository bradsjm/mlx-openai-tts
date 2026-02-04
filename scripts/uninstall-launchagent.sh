#!/usr/bin/env bash
set -euo pipefail

LABEL="com.local.mlx-openai-tts"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
rm -f "$PLIST_DST"

echo "Uninstalled LaunchAgent: $LABEL"
