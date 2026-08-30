#!/bin/zsh
# Weekly/daily refresh so the season rolls forward on its own.
#
# Runs the whole chain in dependency order and logs everything. Safe to run repeatedly:
# every step is idempotent (stat snapshots de-dupe on date, the ledger locks once per fixture,
# the fantasy snapshot seals once per gameweek and refuses to touch a started gameweek).
#
# Installed as a launchd agent - see com.samiakil.premrefresh.plist. Runs whether or not
# Claude is open; that is the point, since the dashboard going stale is what made the GW2
# squad show as locked a day after it had actually unlocked.

set -u
# Derive paths from THIS script's own location so a clone runs anywhere, on any machine.
DIR="${0:A:h}"
REPO="${DIR:h}"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
LOG="$DIR/refresh.log"
LOCK="$DIR/.refresh.lock"

exec >> "$LOG" 2>&1
echo "=========================================================="
echo "refresh start $(date '+%Y-%m-%d %H:%M:%S %Z')"

# don't let a slow run overlap the next trigger
if [ -d "$LOCK" ]; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +90 2>/dev/null)" ]; then
    echo "  stale lock (>90m) - clearing"; rmdir "$LOCK" 2>/dev/null
  else
    echo "  another refresh is running - skipping"; exit 0
  fi
fi
mkdir "$LOCK" 2>/dev/null || { echo "  could not take lock - skipping"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$DIR" || exit 1
fail=0
# `python x.py | grep ...` reports GREP's status, not python's. The first unattended run
# crashed inside build_dashboard.py and this script still logged "exit=0" - a scheduler that
# reports success while the build is broken is worse than no scheduler. pipefail + PIPESTATUS
# make the python exit code the one that counts.
setopt pipefail 2>/dev/null || set -o pipefail
run() {
  echo "--- $1"
  "$PY" "$1" 2>&1 | grep -viE '^warning|deprecat'
  local rc=${pipestatus[1]:-${PIPESTATUS[0]}}
  if [ "${rc:-1}" -ne 0 ]; then
    echo "  !! $1 FAILED (exit $rc)"; fail=1
  fi
}

run build_player_stats.py     # player xG/xA/xGC/DC + per-gameweek + dated history snapshot
run build_dashboard.py        # refits the model on finished results, relocks ledger + fantasy
run make_standalone.py        # regenerates the shareable HTML

# keep the log from growing without bound
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 4000 ]; then
  tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

if [ "$fail" -ne 0 ]; then
  echo "refresh FAILED $(date '+%Y-%m-%d %H:%M:%S %Z') - see the traceback above"
else
  echo "refresh done $(date '+%Y-%m-%d %H:%M:%S %Z')  exit=0"
fi
exit $fail
