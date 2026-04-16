#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=mosca-fuse-gen3c-8k-8frame-very-aggressive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ata.atasoy@tum.de
source ~/.bashrc
conda activate mosca

MOSCA_DIR="${PROJECT_DIR}/MoSca"

cd "$MOSCA_DIR"

mkdir -p logs

PATCH_CFG="${MOSCA_DIR}/profile/vipe/cowboy_cat_fuse_gen3c_8k_8frame_very_aggressive.yaml"

echo "MoSca dir  : $MOSCA_DIR"
echo "Patch cfg  : $PATCH_CFG"

if [[ ! -f "$PATCH_CFG" ]]; then
  echo "Patch config not found: $PATCH_CFG" >&2
  exit 1
fi

python mosca_fuse.py \
  --cfg "$PATCH_CFG" \
 