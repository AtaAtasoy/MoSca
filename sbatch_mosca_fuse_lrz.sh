#!/bin/bash
#SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --job-name=mosca-fuse-gen3c-spiral-useful
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ata.atasoy@tum.de
source ~/.bashrc
conda activate mosca

MOSCA_DIR="${PROJECT_DIR}/MoSca"

cd "$MOSCA_DIR"

mkdir -p logs

PATCH_CFG="${MOSCA_DIR}/profile/vipe/cowboy_cat_gen3c_fuse_spiral_useful.yaml"

echo "MoSca dir  : $MOSCA_DIR"
echo "Patch cfg  : $PATCH_CFG"

if [[ ! -f "$PATCH_CFG" ]]; then
  echo "Patch config not found: $PATCH_CFG" >&2
  exit 1
fi

python mosca_fuse.py \
  --cfg "$PATCH_CFG"
