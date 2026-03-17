#!/bin/bash
# Build kaggle-project.zip for uploading to Kaggle (Datasets or Notebooks).
# Excludes large/data files and Chinese doc; includes code, config, README.

set -e
cd "$(dirname "$0")"
ZIP="kaggle-project.zip"
[ -f "$ZIP" ] && rm -f "$ZIP"

zip -r "$ZIP" . \
  -x "*.parquet" \
  -x "*.pkl" \
  -x "*.zip" \
  -x "submission.csv" \
  -x "项目介绍_中文.md" \
  -x ".git/*" \
  -x "__pycache__/*" \
  -x "*.log" \
  -x ".DS_Store"

echo "Created $ZIP"
ls -la "$ZIP"
