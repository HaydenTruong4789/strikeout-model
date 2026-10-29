#!/usr/bin/env bash
# Scheduled data update (run every 5 days by launchd; see scripts/install_schedule.sh):
# fetch games played since the last update, rebuild features, refit, and
# refresh the upcoming-start predictions. Output is appended to logs/update.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
exec >> logs/update.log 2>&1
echo "=== Update started $(date '+%Y-%m-%d %H:%M:%S') ==="
"$PY" src/fetch_data.py --incremental
"$PY" src/build_features.py
"$PY" src/train.py
"$PY" src/evaluate.py
"$PY" src/predict_upcoming.py --days 5
echo "=== Update finished $(date '+%Y-%m-%d %H:%M:%S') ==="
