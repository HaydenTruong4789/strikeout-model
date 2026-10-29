#!/usr/bin/env bash
# Run the full pipeline end to end: fetch -> features -> train -> evaluate -> patterns.
set -euo pipefail
cd "$(dirname "$0")"
python src/fetch_data.py "$@"
python src/build_features.py
python src/train.py
python src/evaluate.py
python src/analyze_patterns.py
