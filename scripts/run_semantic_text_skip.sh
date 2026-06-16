#!/bin/bash
# Description: Semantic Text Skip Connection experiment for HunyuanVideo, launched
# across 4 GPUs with sequence parallelism (ulysses-degree x ring-degree).
#
# Re-injects clean pre-dual (prompt-semantic) text features into the
# video-conditioned text right before the long single-stream stage:
#
#     txt_single = txt_after_dual + alpha * anchor
#         raw:        anchor = txt_before_dual
#         norm_match: anchor = norm_match(txt_before_dual, txt_after_dual)
#         mid:        anchor = norm_match(txt_dual_block<mid>, txt_after_dual)
#
# Sweeps a list of "mode:alpha" specs (baseline = none:0.0). Each run produces a
# video and a token-drift JSONL so alignment through the single-stream blocks can
# be compared. Inference-only; does not change default generation behavior.
#
# Supported 4-GPU parallel configs for 1280x720 / 720x1280, length 129:
#   --ulysses-degree x --ring-degree = 4x1, 2x2, 1x4   (--nproc_per_node 4)

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_BASE=${MODEL_BASE:-"${EXP_ROOT}/ckpts"}
DIT_WEIGHT=${DIT_WEIGHT:-"${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"}
SAVE_PATH=${SAVE_PATH:-"${EXP_ROOT}/results/semantic_text_skip"}
PROMPT=${PROMPT:-"A red car drives past a white fence"}
PROMPT_TAG=${PROMPT_TAG:-"red_car"}
SEED=${SEED:-42}
VIDEO_H=${VIDEO_H:-720}
VIDEO_W=${VIDEO_W:-1280}
VIDEO_LENGTH=${VIDEO_LENGTH:-129}
INFER_STEPS=${INFER_STEPS:-50}
EMBEDDED_CFG_SCALE=${EMBEDDED_CFG_SCALE:-6.0}
FLOW_SHIFT=${FLOW_SHIFT:-7.0}
MID_BLOCK=${MID_BLOCK:-10}
DRIFT_BLOCKS=${DRIFT_BLOCKS:-"-1,0,10,20,30,39"}

# --- 4-GPU sequence parallelism ---
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
ULYSSES_DEGREE=${ULYSSES_DEGREE:-4}
RING_DEGREE=${RING_DEGREE:-1}

# Space-separated list of "mode:alpha" runs.
# Default: baseline + norm-matched skip at alpha 0.05 and 0.1 (recommended start).
SPECS=${SPECS:-"none:0.0 norm_match:0.05 norm_match:0.1"}

export MODEL_BASE

mkdir -p "${SAVE_PATH}"

for SPEC in ${SPECS}; do
    MODE="${SPEC%%:*}"
    ALPHA="${SPEC##*:}"
    RUN_TAG="${PROMPT_TAG}_${MODE}_a${ALPHA}"
    DRIFT_PATH="${SAVE_PATH}/drift_${RUN_TAG}.jsonl"
    rm -f "${DRIFT_PATH}"

    echo "=============================================================="
    echo "[semantic-skip] mode=${MODE} alpha=${ALPHA} | ${NPROC_PER_NODE} GPUs (ulysses=${ULYSSES_DEGREE} ring=${RING_DEGREE})"
    echo "[semantic-skip] prompt=\"${PROMPT}\" -> drift=${DRIFT_PATH}"
    echo "=============================================================="

    torchrun --nproc_per_node="${NPROC_PER_NODE}" "${EXP_ROOT}/sample_video.py" \
        --model-base "${MODEL_BASE}" \
        --dit-weight "${DIT_WEIGHT}" \
        --video-size "${VIDEO_H}" "${VIDEO_W}" \
        --video-length "${VIDEO_LENGTH}" \
        --infer-steps "${INFER_STEPS}" \
        --prompt "${PROMPT}" \
        --seed "${SEED}" \
        --embedded-cfg-scale "${EMBEDDED_CFG_SCALE}" \
        --flow-shift "${FLOW_SHIFT}" \
        --flow-reverse \
        --ulysses-degree "${ULYSSES_DEGREE}" \
        --ring-degree "${RING_DEGREE}" \
        --save-path "${SAVE_PATH}" \
        --name-suffix "${RUN_TAG}" \
        --single-stream-text-skip "${MODE}" \
        --single-stream-text-skip-alpha "${ALPHA}" \
        --single-stream-text-skip-mid-block "${MID_BLOCK}" \
        --token-drift-debug \
        --token-drift-debug-path "${DRIFT_PATH}" \
        --token-drift-debug-blocks="${DRIFT_BLOCKS}" \
        "${@}"

    echo "[semantic-skip] done: ${RUN_TAG}"
done

echo "Done. Videos and drift logs are under: ${SAVE_PATH}"
