#!/bin/bash
# ============================================================================
# ONE script to run the whole single-stream text-token experiment suite.
#
# It runs a labeled list of configurations, each as a 4-GPU sequence-parallel
# generation, and writes for every run:
#   - a video             -> results/all_text_experiments/<...>_<TAG>.mp4
#   - a token-drift JSONL  -> results/all_text_experiments/drift_<TAG>.jsonl
#
# Every run also has --token-drift-debug on, so you can compare how text->video
# alignment (img_txt_mean_cos across single-stream blocks) changes between the
# baseline and each intervention.
#
# Experiments covered (each is inference-only and flag-gated; baseline = no flags):
#   baseline                normal HunyuanVideo generation
#   ablation_zero           zero the text before single-stream
#   ablation_scale2         scale text x2 before single-stream
#   skip_raw_a0.05          raw semantic skip,        alpha 0.05
#   skip_normmatch_a0.05    norm-matched skip,        alpha 0.05   (recommended)
#   skip_normmatch_a0.10    norm-matched skip,        alpha 0.10   (recommended)
#   skip_mid_a0.10          mid (block 10) skip,      alpha 0.10
#
# NOTE: the `shuffle` ablation is intentionally NOT in the default list because
# under sequence parallelism each rank would shuffle text differently. To run it,
# use a single GPU:  NPROC_PER_NODE=1 ULYSSES_DEGREE=1 RING_DEGREE=1 \
#   EXPERIMENTS="shuffle|--single-stream-text-ablation shuffle" bash <this script>
# ============================================================================

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------- shared settings ------------------------------
MODEL_BASE=${MODEL_BASE:-"${EXP_ROOT}/ckpts"}
DIT_WEIGHT=${DIT_WEIGHT:-"${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"}
SAVE_PATH=${SAVE_PATH:-"${EXP_ROOT}/results/all_text_experiments"}
PROMPT=${PROMPT:-"A red car drives past a white fence"}
PROMPT_TAG=${PROMPT_TAG:-"red_car"}
SEED=${SEED:-42}
VIDEO_H=${VIDEO_H:-720}
VIDEO_W=${VIDEO_W:-1280}
VIDEO_LENGTH=${VIDEO_LENGTH:-129}
INFER_STEPS=${INFER_STEPS:-50}
EMBEDDED_CFG_SCALE=${EMBEDDED_CFG_SCALE:-6.0}
FLOW_SHIFT=${FLOW_SHIFT:-7.0}
DRIFT_BLOCKS=${DRIFT_BLOCKS:-"-1,0,10,20,30,39"}

# 4-GPU sequence parallelism (override for other layouts: 2x2, 1x4, or 1 GPU).
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
ULYSSES_DEGREE=${ULYSSES_DEGREE:-4}
RING_DEGREE=${RING_DEGREE:-1}

export MODEL_BASE
mkdir -p "${SAVE_PATH}"

# ------------------------- experiment table ---------------------------------
# Format per entry: "TAG|<extra sample_video.py flags>"
# Override the whole table by exporting EXPERIMENTS (newline-separated).
DEFAULT_EXPERIMENTS="baseline|
ablation_zero|--single-stream-text-ablation zero
ablation_scale2|--single-stream-text-ablation scale --single-stream-text-scale 2.0
skip_raw_a0.05|--single-stream-text-skip raw --single-stream-text-skip-alpha 0.05
skip_normmatch_a0.05|--single-stream-text-skip norm_match --single-stream-text-skip-alpha 0.05
skip_normmatch_a0.10|--single-stream-text-skip norm_match --single-stream-text-skip-alpha 0.10
skip_mid_a0.10|--single-stream-text-skip mid --single-stream-text-skip-alpha 0.10 --single-stream-text-skip-mid-block 10"

EXPERIMENTS=${EXPERIMENTS:-"${DEFAULT_EXPERIMENTS}"}

run_one() {
    local tag="$1"; shift
    local run_tag="${PROMPT_TAG}_${tag}"
    local drift_path="${SAVE_PATH}/drift_${run_tag}.jsonl"
    rm -f "${drift_path}"

    echo "=============================================================="
    echo "[all-exp] ${run_tag} | extra: $*"
    echo "[all-exp] ${NPROC_PER_NODE} GPUs (ulysses=${ULYSSES_DEGREE} ring=${RING_DEGREE}) -> ${drift_path}"
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
        --name-suffix "${run_tag}" \
        --token-drift-debug \
        --token-drift-debug-path "${drift_path}" \
        --token-drift-debug-blocks="${DRIFT_BLOCKS}" \
        "$@"
}

echo "##############################################################"
echo "# Text-token experiment suite"
echo "# prompt: \"${PROMPT}\"  (tag: ${PROMPT_TAG})"
echo "# output: ${SAVE_PATH}"
echo "##############################################################"

while IFS= read -r line; do
    [ -z "${line}" ] && continue
    tag="${line%%|*}"
    extra="${line#*|}"
    # shellcheck disable=SC2086  # intentional word-splitting of the flag string
    run_one "${tag}" ${extra}
done <<< "${EXPERIMENTS}"

echo "All experiments finished."
echo "Videos + per-run token-drift logs are under: ${SAVE_PATH}"
echo "Compare runs, e.g.:  grep '\"debug_stage\": \"single_block\"' ${SAVE_PATH}/drift_${PROMPT_TAG}_*.jsonl"
