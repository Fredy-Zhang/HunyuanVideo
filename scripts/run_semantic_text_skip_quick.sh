#!/bin/bash
# Description: Fast smoke run for the Semantic Text Skip Connection experiment on
# 4 GPUs. Verifies the skip flags and token-drift output before a full run.
# Lower resolution and fewer steps; baseline + norm_match alpha 0.1 only.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 960x544 / length 129 supports 4x1, 2x2, 1x4 on 4 GPUs.
SPECS=${SPECS:-"none:0.0 norm_match:0.1"} \
VIDEO_H=${VIDEO_H:-544} \
VIDEO_W=${VIDEO_W:-960} \
INFER_STEPS=${INFER_STEPS:-8} \
NPROC_PER_NODE=${NPROC_PER_NODE:-4} \
ULYSSES_DEGREE=${ULYSSES_DEGREE:-4} \
RING_DEGREE=${RING_DEGREE:-1} \
SAVE_PATH=${SAVE_PATH:-"$(cd "${SCRIPT_DIR}/.." && pwd)/results/semantic_text_skip_quick"} \
    bash "${SCRIPT_DIR}/run_semantic_text_skip.sh" "${@}"
