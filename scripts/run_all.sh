#!/usr/bin/env bash
# Full pipeline on Linux or macOS with a CUDA GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m lookahead data
python -m lookahead check
python -m lookahead extract
python -m lookahead analyze
