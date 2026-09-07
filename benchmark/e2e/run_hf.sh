#!/usr/bin/env bash
# Thin compatibility wrapper; .env is parsed by python-dotenv, never executed.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python3}" -m benchmark.e2e.hf_driver "$@"
