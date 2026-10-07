#!/usr/bin/env bash
# CapField-OPD - inference: one prompt, several coordinates.
#
# Produces one image per coordinate (shared initial noise) so you can inspect
# how the student moves through the capability field.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-capfieldopd}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" \
        || echo "warning: conda env '${CONDA_ENV}' not found, using the current environment"
fi

# Student weights: a local folder, or download straight from the Hugging Face Hub.
#   local       : CKPT=/path/to/student_ckpt   (CKPT_SUBFOLDER ignored if the
#                                               subfolder does not exist there)
#   from the Hub: CKPT=Yang18/CapField-OPD  CKPT_SUBFOLDER=student_ckpt
FLUX_PATH="${FLUX_PATH:-black-forest-labs/FLUX.1-dev}"
CKPT="${CKPT:-Yang18/CapField-OPD}"
CKPT_SUBFOLDER="${CKPT_SUBFOLDER:-student_ckpt}"
PROMPT="${PROMPT:-A close-up of a sign that reads \"HELLO WORLD\"}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/inference}"

# Coordinates are 3-axis [geneval, ocr, aesthetics]:
#   (0,0,0) base | (1,0,0) geneval | (0,1,0) ocr | (0,0,1) aesthetics
#   (1,0,1) geneval+aesthetics | (0,1,1) ocr+aesthetics
python "${REPO_ROOT}/inference.py" \
    --ckpt "${CKPT}" \
    --ckpt_subfolder "${CKPT_SUBFOLDER}" \
    --base_model "${FLUX_PATH}" \
    --prompt "${PROMPT}" \
    --coords "0.0 0.0 0.0" "1.0 0.0 0.0" "0.0 1.0 0.0" "0.0 0.0 1.0" "1.0 0.0 1.0" "0.0 1.0 1.0" \
    --denoising_steps 28 \
    --guidance_scale 4.5 \
    --shift_mode flux1_shift \
    --output_dir "${OUTPUT_DIR}"
