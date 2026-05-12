#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
if [ -z "${HUNYUAN_DIR:-}" ]; then
    if [ -f "${ROOT_DIR}/sample_video.py" ]; then
        HUNYUAN_DIR="${ROOT_DIR}"
    else
        HUNYUAN_DIR="${ROOT_DIR}/HunyuanVideo"
    fi
fi
COMPBENCH_DIR="${COMPBENCH_DIR:-${ROOT_DIR}/../T2V-CompBench}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/../HunyuanVideo/ckpts}"
DIT_WEIGHT="${DIT_WEIGHT:-${CKPT_DIR}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}"

PROMPT_DIR="${PROMPT_DIR:-${COMPBENCH_DIR}/prompts}"
OUT_ROOT="${OUT_ROOT:-${COMPBENCH_DIR}/generated_videos_v3}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MODEL_BASE="${MODEL_BASE:-${CKPT_DIR}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
VIDEO_LENGTH="${VIDEO_LENGTH:-129}"
INFER_STEPS="${INFER_STEPS:-50}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29501}"

ULYSSES_DEGREE="${ULYSSES_DEGREE:-4}"
RING_DEGREE="${RING_DEGREE:-1}"

CFG_SCALE="${CFG_SCALE:-1.0}"
EMBEDDED_CFG_SCALE="${EMBEDDED_CFG_SCALE:-6.0}"
FLOW_SHIFT="${FLOW_SHIFT:-7.0}"
USE_FP8="${USE_FP8:-0}"
TEST_NUM="${TEST_NUM:-20}"

ENABLE_ST_ATTN_COHESION="${ENABLE_ST_ATTN_COHESION:-1}"
ST_ATTN_SPATIAL_TOPK_RATIO="${ST_ATTN_SPATIAL_TOPK_RATIO:-0.01}"
ST_ATTN_TEMPORAL_STRENGTH="${ST_ATTN_TEMPORAL_STRENGTH:-0.35}"
ST_ATTN_BACKGROUND_SUPPRESS="${ST_ATTN_BACKGROUND_SUPPRESS:-0.10}"
ST_ATTN_SCORE_MOMENTUM="${ST_ATTN_SCORE_MOMENTUM:-0.7}"
ST_ATTN_LOCALIZE_CHUNK_SIZE="${ST_ATTN_LOCALIZE_CHUNK_SIZE:-2048}"

ENABLE_GLYPH_GUIDANCE="${ENABLE_GLYPH_GUIDANCE:-0}"
GLYPH_OCR_BACKEND="${GLYPH_OCR_BACKEND:-clip}"
GLYPH_ETA="${GLYPH_ETA:-0.5}"
GLYPH_SIGMA_MIN="${GLYPH_SIGMA_MIN:-0.2}"
GLYPH_SIGMA_MAX="${GLYPH_SIGMA_MAX:-0.8}"
GLYPH_DECODE_RESIZE="${GLYPH_DECODE_RESIZE:-0.5}"
GLYPH_GRAD_CLIP="${GLYPH_GRAD_CLIP:-1.0}"
GLYPH_TARGET_TEXT="${GLYPH_TARGET_TEXT:-}"

declare -A FILE_TO_CATEGORY
FILE_TO_CATEGORY["1_consistent_attr.txt"]="consistent_attr"
FILE_TO_CATEGORY["2_dynamic_attr.txt"]="dynamic_attr"
FILE_TO_CATEGORY["3_spatial_relationship.txt"]="spatial_relationship"
FILE_TO_CATEGORY["4_motion_binding.txt"]="motion_binding"
FILE_TO_CATEGORY["5_action_binding.txt"]="action_binding"
FILE_TO_CATEGORY["6_interaction.txt"]="interaction"
FILE_TO_CATEGORY["7_numeracy.txt"]="numeracy"

PROMPT_FILES=(
    "1_consistent_attr.txt"
    "2_dynamic_attr.txt"
    "3_spatial_relationship.txt"
    "4_motion_binding.txt"
    "5_action_binding.txt"
    "6_interaction.txt"
    "7_numeracy.txt"
)

tmp_dir=""
cleanup_tmp() {
    if [ -n "${tmp_dir}" ] && [ -d "${tmp_dir}" ]; then
        rm -rf "${tmp_dir}"
    fi
}
trap cleanup_tmp EXIT

if [ ! -d "${HUNYUAN_DIR}" ]; then
    echo "HunyuanVideo dir not found: ${HUNYUAN_DIR}" >&2
    exit 1
fi

if [ ! -d "${PROMPT_DIR}" ]; then
    echo "Prompt dir not found: ${PROMPT_DIR}" >&2
    exit 1
fi

if [ ! -d "${CKPT_DIR}" ]; then
    echo "Checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi

if [ ! -f "${DIT_WEIGHT}" ]; then
    echo "DiT checkpoint not found: ${DIT_WEIGHT}" >&2
    exit 1
fi

if [ $((ULYSSES_DEGREE * RING_DEGREE)) -ne "${NUM_GPUS}" ]; then
    echo "NUM_GPUS (${NUM_GPUS}) must equal ULYSSES_DEGREE x RING_DEGREE (${ULYSSES_DEGREE} x ${RING_DEGREE})." >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}"
cd "${HUNYUAN_DIR}"

echo "Root dir: ${ROOT_DIR}"
echo "HunyuanVideo dir: ${HUNYUAN_DIR}"
echo "T2V-CompBench dir: ${COMPBENCH_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Output root: ${OUT_ROOT}"
echo "Resolution: ${HEIGHT}x${WIDTH}, frames=${VIDEO_LENGTH}, steps=${INFER_STEPS}"
echo "Parallel: gpus=${NUM_GPUS}, ulysses=${ULYSSES_DEGREE}, ring=${RING_DEGREE}, master_port=${MASTER_PORT}"
echo "Guidance: cfg=${CFG_SCALE}, embedded_cfg=${EMBEDDED_CFG_SCALE}, flow_shift=${FLOW_SHIFT}"
echo "ST cohesion: enabled=${ENABLE_ST_ATTN_COHESION}, topk=${ST_ATTN_SPATIAL_TOPK_RATIO}, temporal=${ST_ATTN_TEMPORAL_STRENGTH}, suppress=${ST_ATTN_BACKGROUND_SUPPRESS}"
echo "Glyph guidance: enabled=${ENABLE_GLYPH_GUIDANCE}, backend=${GLYPH_OCR_BACKEND}, eta=${GLYPH_ETA}, sigma=[${GLYPH_SIGMA_MIN},${GLYPH_SIGMA_MAX}], decode_resize=${GLYPH_DECODE_RESIZE}"
echo "Test num: ${TEST_NUM}"
echo ""

for prompt_file_name in "${PROMPT_FILES[@]}"; do
    category="${FILE_TO_CATEGORY[$prompt_file_name]}"
    prompt_file="${PROMPT_DIR}/${prompt_file_name}"
    out_dir="${OUT_ROOT}/${category}"

    mkdir -p "${out_dir}"

    echo "========================================"
    echo "Category: ${category}"
    echo "Prompt file: ${prompt_file}"
    echo "Output dir: ${out_dir}"
    echo "========================================"

    if [ ! -f "${prompt_file}" ]; then
        echo "Prompt file not found: ${prompt_file}"
        continue
    fi

    i=1
    while IFS= read -r prompt || [ -n "$prompt" ]; do
        if [[ -z "${prompt// }" ]]; then
            continue
        fi

        video_id=$(printf "%04d" "$i")
        final_video="${out_dir}/${video_id}.mp4"

        if [ -f "${final_video}" ]; then
            echo "[Skip] ${final_video} already exists."
            i=$((i + 1))
            continue
        fi

        tmp_dir="$(mktemp -d "${HUNYUAN_DIR}/tmp_t2v_${category}_${video_id}_XXXX")"

        echo ""
        echo "----------------------------------------"
        echo "Generating ${category}/${video_id}.mp4"
        echo "Prompt: ${prompt}"
        echo "Temp dir: ${tmp_dir}"
        echo "----------------------------------------"

        cmd=(
            torchrun
            --nproc_per_node="${NUM_GPUS}"
            --master_port="${MASTER_PORT}"
            sample_video.py
            --model-base "${CKPT_DIR}"
            --dit-weight "${DIT_WEIGHT}"
            --video-size "${HEIGHT}" "${WIDTH}"
            --video-length "${VIDEO_LENGTH}"
            --infer-steps "${INFER_STEPS}"
            --prompt "${prompt}"
            --seed "${SEED}"
            --cfg-scale "${CFG_SCALE}"
            --embedded-cfg-scale "${EMBEDDED_CFG_SCALE}"
            --flow-shift "${FLOW_SHIFT}"
            --flow-reverse
            --ulysses-degree "${ULYSSES_DEGREE}"
            --ring-degree "${RING_DEGREE}"
            --save-path "${tmp_dir}"
        )

        if [ "${USE_FP8}" = "1" ]; then
            cmd+=(--use-fp8)
        fi

        if [ "${ENABLE_ST_ATTN_COHESION}" = "1" ]; then
            cmd+=(
                --enable-st-attn-cohesion
                --st-attn-spatial-topk-ratio "${ST_ATTN_SPATIAL_TOPK_RATIO}"
                --st-attn-temporal-strength "${ST_ATTN_TEMPORAL_STRENGTH}"
                --st-attn-background-suppress "${ST_ATTN_BACKGROUND_SUPPRESS}"
                --st-attn-score-momentum "${ST_ATTN_SCORE_MOMENTUM}"
                --st-attn-localize-chunk-size "${ST_ATTN_LOCALIZE_CHUNK_SIZE}"
            )
        fi

        if [ "${ENABLE_GLYPH_GUIDANCE}" = "1" ]; then
            cmd+=(
                --glyph-guidance
                --glyph-ocr-backend "${GLYPH_OCR_BACKEND}"
                --glyph-eta "${GLYPH_ETA}"
                --glyph-sigma-min "${GLYPH_SIGMA_MIN}"
                --glyph-sigma-max "${GLYPH_SIGMA_MAX}"
                --glyph-decode-resize "${GLYPH_DECODE_RESIZE}"
                --glyph-grad-clip "${GLYPH_GRAD_CLIP}"
            )
            if [ -n "${GLYPH_TARGET_TEXT}" ]; then
                cmd+=(--glyph-target-text "${GLYPH_TARGET_TEXT}")
            fi
        fi

        "${cmd[@]}"

        shopt -s nullglob
        generated_videos=("${tmp_dir}"/*.mp4)
        shopt -u nullglob

        if [ "${#generated_videos[@]}" -gt 0 ]; then
            mv "${generated_videos[0]}" "${final_video}"
            echo "[Saved] ${final_video}"
        else
            echo "[Error] No generated mp4 found for ${category}/${video_id}" >&2
        fi

        cleanup_tmp
        tmp_dir=""

        i=$((i + 1))
        if [ "${TEST_NUM}" -gt 0 ] && [ "$i" -gt "${TEST_NUM}" ]; then
            echo "[Test mode] Finished first ${TEST_NUM} prompts for ${category}."
            break
        fi
    done < "${prompt_file}"
done

echo ""
echo "All done."
