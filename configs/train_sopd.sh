#!/usr/bin/env bash
# ============================================================
# CapField-OPD - single-teacher distillation for FLUX (minimal template)
# ============================================================
# One task axis: dim 0 = OCR (source 0, data: dataset/ocr, teacher: ocr_teacher).
# A coordinate is [w_ocr]:
#   [1.0] = OCR teacher | [0.7] / [0.4] = interpolated with the base model
#   [0.0] = raw base model (unconditioned)
#
# Configuration is two-layered:
#   STEP 1 - teacher pool: which teachers exist and their coordinate vectors.
#   STEP 2 - per-source prompts and the coordinates used for rollout / loss.
#
# Use this as a sanity check before running configs/train_mopd.sh.
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 1. Environment
# Create it once with (see README "Installation"):
#   conda create -n capfieldopd python=3.10 -y
#   conda activate capfieldopd
#   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
#   pip install -r requirements.txt
CONDA_ENV="${CONDA_ENV:-capfieldopd}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" \
        || echo "warning: conda env '${CONDA_ENV}' not found, using the current environment"
fi

export NNODES="${NODE_COUNT:-1}"
export NPROC_PER_NODE="${PROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29587}"

# 2. Model / teacher weights (replace the paths below with your own)
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/ckpts}"
FLUX_PATH="${FLUX_PATH:-/path/to/FLUX.1-dev}"
TEACHERS_DIR="${TEACHERS_DIR:-${CKPTS_DIR}/CapField-OPD/teachers_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/sopd}"

# ============================================================
# STEP 1 - Teacher pool
# ============================================================
# One entry per teacher: "LORA_PATH | w_ocr"
#   "1.0" = single-axis teacher; path "none" = raw base model (zero LoRA).
# ============================================================
EXPERTS=(
    "${TEACHERS_DIR}/ocr_teacher    | 1.0"
)

# ============================================================
# STEP 2 - Per-source config (source i <-> coordinate dim i)
# ============================================================
# SRCn_PROMPT_TXT    : training prompts of this source (one .txt or .jsonl).
# SRCn_ROLLOUT       : rollout coordinates, ';'-separated. One is drawn per
#                      prompt and reused for the whole trajectory.
# SRCn_ROLLOUT_PROBS : draw probability of each rollout coordinate.
# SRCn_LOSS_LEVELS   : loss-time coordinate pool, ';'-separated. One level is
#                      drawn per distillation timestep.
# ============================================================

# --- Source 0: OCR (axis 0) ---
SRC0_PROMPT_TXT="${REPO_ROOT}/dataset/ocr/train.txt"
SRC0_ROLLOUT="1.0"
SRC0_ROLLOUT_PROBS="1.0"
SRC0_LOSS_LEVELS="1.0; 0.7; 0.4; 0.0"

# 3. Aggregate into arrays (order = source/axis order)
SOURCE_PROMPT_PATHS=("${SRC0_PROMPT_TXT}")
SOURCE_LOSS_LEVELS=("${SRC0_LOSS_LEVELS}")
ROLLOUT_VECS=("${SRC0_ROLLOUT}")
ROLLOUT_PROBS=("${SRC0_ROLLOUT_PROBS}")

# 4. Launch training
mkdir -p "${OUTPUT_DIR}"
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/train.py" \
    --pretrained_model_name_or_path "${FLUX_PATH}" \
    --experts "${EXPERTS[@]}" \
    --source_prompt_txt_paths "${SOURCE_PROMPT_PATHS[@]}" \
    --source_cond_vectors "${SOURCE_LOSS_LEVELS[@]}" \
    --rollout_cond_vectors "${ROLLOUT_VECS[@]}" \
    --rollout_cond_probs "${ROLLOUT_PROBS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 142 \
    --batch_size 1 \
    --dataloader_num_workers 0 \
    --mixed_precision bf16 \
    --save_interval 200 \
    --height 512 \
    --width 512 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --task_cond_hidden_dim 256 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --max_train_steps 1000 \
    --denoising_steps 10 \
    --shift_mode sd3_shift \
    --train_stepidx 0 1 2 3 4 5 6 7 8 9 \
    --guidance_scale 4.5 \
    --loss_target velocity \
    --ema \
    --ema_decay 0.9 \
    --ema_update_step_interval 8 \
    2>&1 | tee -a "${OUTPUT_DIR}/train_stdout.log"
