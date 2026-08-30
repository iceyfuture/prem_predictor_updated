#!/bin/zsh
# One-time setup on a new machine. Creates the venv and installs dependencies.
set -e
DIR="${0:A:h}"
cd "$DIR"
echo "==> creating virtualenv"
python3 -m venv .venv
echo "==> installing dependencies"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo "==> setting up your FPL team id"
if [ ! -f dashboard/fpl_config.json ]; then
  cp dashboard/fpl_config.example.json dashboard/fpl_config.json
  echo "    edit dashboard/fpl_config.json and set your entry_id"
  echo "    (or export FPL_ENTRY_ID=<your id>, which takes priority)"
fi
echo
echo "Done. Build everything with:"
echo "    ./dashboard/refresh.sh"
echo "Then open dashboard/standalone.html in a browser."
