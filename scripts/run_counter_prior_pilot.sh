#!/usr/bin/env bash
# Run the 36-video, seed-matched counter-prior binding pilot.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_BASE="${MODEL_BASE:-${REPO_ROOT}/ckpts}"
DIT_WEIGHT="${DIT_WEIGHT:-${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}"
PILOT_OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-${REPO_ROOT}/results/counter_prior_pilot}"
PILOT_SEEDS="${PILOT_SEEDS:-42 123 456 789 1024 2026}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-544}"
VIDEO_WIDTH="${VIDEO_WIDTH:-960}"
VIDEO_LENGTH="${VIDEO_LENGTH:-129}"
INFER_STEPS="${INFER_STEPS:-50}"
EMBEDDED_CFG_SCALE="${EMBEDDED_CFG_SCALE:-6.0}"
FLOW_SHIFT="${FLOW_SHIFT:-7.0}"
USE_CPU_OFFLOAD="${USE_CPU_OFFLOAD:-1}"
USE_FP8="${USE_FP8:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEEDS <<< "${PILOT_SEEDS}"

COMMAND=(
  "${PYTHON_BIN}" tools/run_counter_prior_pilot.py
  --pilot-output-root "${PILOT_OUTPUT_ROOT}"
  --pilot-seeds "${SEEDS[@]}"
  --model-base "${MODEL_BASE}"
  --dit-weight "${DIT_WEIGHT}"
  --video-size "${VIDEO_HEIGHT}" "${VIDEO_WIDTH}"
  --video-length "${VIDEO_LENGTH}"
  --infer-steps "${INFER_STEPS}"
  --embedded-cfg-scale "${EMBEDDED_CFG_SCALE}"
  --flow-shift "${FLOW_SHIFT}"
  --flow-reverse
  --reproduce
)

if [[ "${USE_CPU_OFFLOAD}" == "1" ]]; then
  COMMAND+=(--use-cpu-offload)
fi
if [[ "${USE_FP8}" == "1" ]]; then
  COMMAND+=(--use-fp8)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  COMMAND+=(--dry-run)
fi

# Forward optional standard HunyuanVideo or pilot arguments supplied by caller.
COMMAND+=("$@")

cd "${REPO_ROOT}"
printf 'Running:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
