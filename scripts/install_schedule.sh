#!/usr/bin/env bash
# Install (or reinstall) a macOS launchd job that runs scripts/update_data.sh
# every 5 days. If the Mac is asleep when it's due, it runs on wake; it runs
# only while you're logged in. Remove with: scripts/install_schedule.sh --remove
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.strikeout-model.update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$PLIST"
  echo "Removed $LABEL"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/scripts/update_data.sh</string></array>
  <key>StartInterval</key><integer>432000</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$ROOT/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/launchd.log</string>
</dict>
</plist>
PLIST
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "Installed $LABEL: runs $ROOT/scripts/update_data.sh every 5 days"
echo "Run it now with: launchctl kickstart $DOMAIN/$LABEL"
