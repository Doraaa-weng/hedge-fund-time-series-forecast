#!/bin/bash
# Wrapper script to run train_quick.py with system Python that has packages installed

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
/usr/bin/python3 "$PROJECT_ROOT/train_quick.py" "$@"
