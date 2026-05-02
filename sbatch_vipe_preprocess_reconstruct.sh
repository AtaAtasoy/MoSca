#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=mosca-vipe-recon
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ata.atasoy@tum.de

set -euo pipefail
export LC_ALL=C

source ~/.bashrc
conda activate mosca

MOSCA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MOSCA_DIR"

mkdir -p logs

WS="${MOSCA_DIR}/runs/cowboy_cat_original_scale"
PREP_CFG="${MOSCA_DIR}/profile/vipe/cowboy_cat_prep_vipe_depth.yaml"
FIT_CFG="${MOSCA_DIR}/profile/vipe/cowboy_cat_fit_fixed_scene.yaml"

echo "MoSca dir : $MOSCA_DIR"
echo "Workspace : $WS"
echo "Prep cfg  : $PREP_CFG"
echo "Fit cfg   : $FIT_CFG"

if [[ ! -d "$WS" ]]; then
  echo "Workspace directory not found: $WS" >&2
  exit 1
fi

python mosca_precompute.py \
  --ws "$WS" \
  --cfg "$PREP_CFG"

python mosca_reconstruct.py \
  --ws "$WS" \
  --cfg "$FIT_CFG" \
  --no_viz
