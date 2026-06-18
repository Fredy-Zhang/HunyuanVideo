#!/usr/bin/env bash
# Build contact sheets, the annotation CSV, and the pilot continuation summary.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PILOT_OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-${REPO_ROOT}/results/counter_prior_pilot}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" tools/summarize_counter_prior_pilot.py \
  --pilot-output-root "${PILOT_OUTPUT_ROOT}" \
  "$@"
