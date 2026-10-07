#!/usr/bin/env bash
# ============================================================
# CapField-OPD - multi-teacher capability-field distillation for FLUX
# ============================================================
# One step: sample one prompt from EVERY source, rollout the student from noise
# under a coordinate, build the teacher target velocity from a coordinate, and
# minimize MSE(student velocity, teacher velocity).
#
# Coordinate axes (dim i of every vector):
#   dim 0 = geneval | dim 1 = ocr | dim 2 = aesthetics
#   e.g. [1,0,0] geneval | [0,1,0] ocr | [0,0,1] aesthetics | [0,1,1] ocr x aesthetics
#
# Two-layer config:
#   STEP 1 - teacher pool: which teachers exist, as "LORA_PATH | v0 v1 v2".
#   STEP 2 - per-source prompts and the coordinates used for rollout / loss.
#
# No reward model is used (evaluation only, see reward_server.py).
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
export NPROC_PER_NODE="${PROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29586}"

# 2. Model / teacher weights (replace the paths below with your own)
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/ckpts}"
FLUX_PATH="${FLUX_PATH:-/path/to/FLUX.1-dev}"
TEACHERS_DIR="${TEACHERS_DIR:-${CKPTS_DIR}/CapField-OPD/teachers_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/mopd}"

# ============================================================
# STEP 1 - Teacher pool
# ============================================================
# One entry per teacher: "LORA_PATH | v0 v1 v2" (v = v_geneval v_ocr v_aes)
#   one non-zero dim  = single-axis teacher ("1.0 0.0 0.0" = geneval teacher)
#   two non-zero dims = joint teacher       ("1.0 0.0 1.0" = geneval x aesthetics)
#   path "none"       = raw base model (zero LoRA)
# ============================================================
EXPERTS=(
    "${TEACHERS_DIR}/geneval_teacher             | 1.0 0.0 0.0"
    "${TEACHERS_DIR}/ocr_teacher                 | 0.0 1.0 0.0"
    "${TEACHERS_DIR}/aesthetic_teacher           | 0.0 0.0 1.0"
    "${TEACHERS_DIR}/geneval_aesthetic_teacher   | 1.0 0.0 1.0"
    "${TEACHERS_DIR}/ocr_aesthetic_teacher       | 0.0 1.0 1.0"
)

# ============================================================
# STEP 2 - Per-source config
# ============================================================
# SRCn_PROMPT_TXT    : this source's prompts (one .txt or .jsonl file).
# SRCn_ROLLOUT       : rollout coordinates, ';'-separated; one is drawn per
#                      prompt and reused for the whole trajectory.
# SRCn_ROLLOUT_PROBS : draw probability of each rollout coordinate.
# SRCn_LOSS_LEVELS   : loss-time coordinate pool, ';'-separated; one level is
#                      drawn per timestep, among levels whose active dims match
#                      the rollout coordinate.
# ============================================================

# --- Source 0: OCR (teacher = ocr_teacher, axis 1) ---
SRC0_PROMPT_TXT="${REPO_ROOT}/dataset/ocr/train.txt"
SRC0_ROLLOUT="0.0 1.0 0.0; 0.0 1.0 1.0"
SRC0_ROLLOUT_PROBS="0.5 0.5"
SRC0_LOSS_LEVELS="0.0 1.0 0.0; 0.0 0.7 0.0; 0.0 0.4 0.0; 0.0 0.0 0.0; 0.0 0.7 0.4; 0.0 0.4 0.7; 0.0 0.5 0.5; 0.0 1.0 1.0"

# --- Source 1: aesthetics (teacher = aesthetic_teacher, axis 2) ---
SRC1_PROMPT_TXT="${REPO_ROOT}/dataset/aesthetic/train.txt"
SRC1_ROLLOUT="0.0 0.0 1.0"
SRC1_ROLLOUT_PROBS="1.0"
SRC1_LOSS_LEVELS="0.0 0.0 1.0; 0.0 0.0 0.7; 0.0 0.0 0.4; 0.0 0.0 0.1; 0.0 0.0 0.0"

# --- Source 2: geneval (teacher = geneval_teacher, axis 0) ---
SRC2_PROMPT_TXT="${REPO_ROOT}/dataset/geneval/train_metadata.jsonl"
SRC2_ROLLOUT="1.0 0.0 0.0; 1.0 0.0 1.0"
SRC2_ROLLOUT_PROBS="0.5 0.5"
SRC2_LOSS_LEVELS="1.0 0.0 0.0; 0.7 0.0 0.0; 0.4 0.0 0.0; 0.0 0.0 0.0; 0.7 0.0 0.4; 0.4 0.0 0.7; 0.5 0.0 0.5; 1.0 0.0 1.0"

# 3. Aggregate into arrays (order = source order)
SOURCE_PROMPT_PATHS=("${SRC0_PROMPT_TXT}" "${SRC1_PROMPT_TXT}" "${SRC2_PROMPT_TXT}")
SOURCE_LOSS_LEVELS=("${SRC0_LOSS_LEVELS}" "${SRC1_LOSS_LEVELS}" "${SRC2_LOSS_LEVELS}")
ROLLOUT_VECS=("${SRC0_ROLLOUT}" "${SRC1_ROLLOUT}" "${SRC2_ROLLOUT}")
ROLLOUT_PROBS=("${SRC0_ROLLOUT_PROBS}" "${SRC1_ROLLOUT_PROBS}" "${SRC2_ROLLOUT_PROBS}")

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
    --seed 42 \
    --batch_size 1 \
    --dataloader_num_workers 0 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 1 \
    --save_interval 200 \
    --height 512 \
    --width 512 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --task_cond_hidden_dim 256 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --max_train_steps 2000 \
    --denoising_steps 10 \
    --shift_mode sd3_shift \
    --train_stepidx 0 1 2 3 4 5 6 7 8 9 \
    --guidance_scale 4.5 \
    --loss_target velocity \
    --ema \
    --ema_decay 0.9 \
    --ema_update_step_interval 8 \
    2>&1 | tee -a "${OUTPUT_DIR}/train_stdout.log"
