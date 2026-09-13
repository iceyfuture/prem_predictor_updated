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

# WAIT FOR THE NETWORK. The 2026-09-11 run died at 06:35 with
#   socket.gaierror: nodename nor servname provided
# because the Mac was waking from sleep and DNS was not up yet. One transient blip took out
# build_player_stats, fotmob, compute_strength AND build_dashboard - four of eight steps - so
# there was no fantasy squad, no Kalshi settlement and no dashboard for a whole day.
net_ready() {
  for host in api.fotmob.com fantasy.premierleague.com site.api.espn.com; do
    if ping -c1 -t2 "$host" >/dev/null 2>&1 ||        /usr/bin/nc -z -G 3 "$host" 443 >/dev/null 2>&1; then return 0; fi
  done
  return 1
}
for i in $(seq 1 30); do          # up to ~5 minutes, then give up and let the retries handle it
  if net_ready; then
    [ "$i" -gt 1 ] && echo "  network came up after $((i*10))s"
    break
  fi
  [ "$i" -eq 1 ] && echo "  waiting for network..."
  /bin/sleep 10
done

fail=0
declare -A STEP_STATUS
# `python x.py | grep ...` reports GREP's status, not python's. The first unattended run
# crashed inside build_dashboard.py and this script still logged "exit=0" - a scheduler that
# reports success while the build is broken is worse than no scheduler. pipefail + PIPESTATUS
# make the python exit code the one that counts.
setopt pipefail 2>/dev/null || set -o pipefail
# Retry each step. Every script here is idempotent - predictions lock once per fixture, the
# fantasy snapshot seals once per gameweek, stat snapshots de-duplicate on date - so re-running
# a step costs nothing and rescues a transient network error instead of losing the day.
RETRIES=3
run() {
  local name="$(basename $1)" rc=1 attempt=1
  echo "--- $name"
  while [ "$attempt" -le "$RETRIES" ]; do
    "$PY" "$1" 2>&1 | grep -viE '^warning|deprecat'
    rc=${pipestatus[1]:-${PIPESTATUS[0]}}
    [ "${rc:-1}" -eq 0 ] && break
    if [ "$attempt" -lt "$RETRIES" ]; then
      echo "  .. $name failed (exit $rc), retry $attempt/$((RETRIES-1)) in $((attempt*20))s"
      /bin/sleep $((attempt*20))
      net_ready || /bin/sleep 30
    fi
    attempt=$((attempt+1))
  done
  if [ "${rc:-1}" -ne 0 ]; then
    echo "  !! $name FAILED after $RETRIES attempts (exit $rc)"; fail=1
    STEP_STATUS[$name]="failed"
  else
    [ "$attempt" -gt 1 ] && echo "  .. $name recovered on attempt $attempt"
    STEP_STATUS[$name]="ok"
  fi
}

# ORDER IS A DEPENDENCY CHAIN, not a preference:
#   build_rankings   fits the history and writes player_rankings.csv + team_rankings.csv
#   link_squads      reads BOTH of those and filters them to this season's squads
#   compute_strength reads player_rankings_2026_27.csv and refits on the LIVE model
#   build_dashboard  re-derives every player's current club (RULE 10) into the files above,
#                    so it must run LAST of the four or link_squads undoes the corrections
run build_player_stats.py     # player xG/xA/xGC/DC + per-gameweek + dated history snapshot
run team_stats.py             # per-match TEAM stat line + rolling form (needs the player file)
run fotmob.py                 # FotMob xG / xGOT / big chances + the xG strength index
run "$REPO/build_rankings.py"     # team + player rankings from history  -> outputs/*.csv
run "$REPO/link_squads.py"        # filter those to the 2026/27 squads
run "$REPO/compute_strength.py"   # strength index (live model) + top-50 players
run build_dashboard.py        # refits the model on finished results, relocks ledger + fantasy
run make_standalone.py        # regenerates the shareable HTML

# keep the log from growing without bound
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 4000 ]; then
  tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# A status file so a silent failure is not invisible. The desk reads it; you can too.
{
  echo "{"
  echo "  \"finished_at\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\","
  echo "  \"ok\": $([ "$fail" -eq 0 ] && echo true || echo false),"
  printf "  \"steps\": {"
  sep=""
  for k in ${(k)STEP_STATUS}; do printf "%s\n    \"%s\": \"%s\"" "$sep" "$k" "${STEP_STATUS[$k]}"; sep=","; done
  echo ""
  echo "  }"
  echo "}"
} > "$DIR/refresh_status.json"

if [ "$fail" -ne 0 ]; then
  echo "refresh FAILED $(date '+%Y-%m-%d %H:%M:%S %Z') - see the traceback above"
else
  echo "refresh done $(date '+%Y-%m-%d %H:%M:%S %Z')  exit=0"
fi
exit $fail
