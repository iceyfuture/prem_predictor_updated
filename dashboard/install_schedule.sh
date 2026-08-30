#!/bin/zsh
# Install (or remove) the daily refresh as a launchd agent. macOS only.
# Paths are derived from this script's location, so it works from any clone.
set -e
DIR="${0:A:h}"
LABEL="com.prempredictor.refresh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

cat > "$PLIST" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>$DIR/refresh.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$DIR/refresh.launchd.log</string>
  <key>StandardErrorPath</key><string>$DIR/refresh.launchd.log</string>
  <!-- Standard, NOT Background: Background + LowPriorityIO throttled this so hard that a
       run taking seconds by hand took ~2 hours, and starved a network read into a reset. -->
  <key>ProcessType</key><string>Standard</string>
</dict>
</plist>
XML

plutil -lint "$PLIST" >/dev/null
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed $LABEL - runs daily at 06:30"
echo "  check:   launchctl list | grep ${LABEL}"
echo "  logs:    tail $DIR/refresh.log"
echo "  remove:  $0 --uninstall"
