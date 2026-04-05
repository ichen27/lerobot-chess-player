#!/usr/bin/env bash
# =============================================================================
# download_model.sh
#
# Download the fine-tuned Pi0 chess model from HuggingFace Hub
# to a local directory for inference on the robot.
#
# Usage:
#   chmod +x download_model.sh
#   ./download_model.sh
# =============================================================================
set -euo pipefail

HF_REPO="izchen/pi0_chess"
LOCAL_DIR="$HOME/chess/weights/pi0_chess"

echo "==> Downloading $HF_REPO to $LOCAL_DIR ..."

# Create the target directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the full model repository
huggingface-cli download "$HF_REPO" --local-dir "$LOCAL_DIR"

echo "==> Model downloaded to: $LOCAL_DIR"
echo ""
echo "Contents:"
ls -lh "$LOCAL_DIR"
