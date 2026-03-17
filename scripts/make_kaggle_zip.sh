#!/bin/bash
# Build kaggle-project.zip for uploading to Kaggle (Datasets or Notebooks).

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p outputs
ZIP="outputs/kaggle-project.zip"
[ -f "$ZIP" ] && rm -f "$ZIP"

zip -r "$ZIP" . \
  -x "*.parquet" \
  -x "*.pkl" \
  -x "*.zip" \
  -x "artifacts/*" \
  -x "outputs/*" \
  -x "reports/*" \
  -x ".git/*" \
  -x "__pycache__/*" \
  -x ".cursor_tmp_pkgs/*" \
  -x "*.log" \
  -x ".DS_Store"

echo "Created $ZIP"
ls -la "$ZIP"
