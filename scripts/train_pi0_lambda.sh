#!/usr/bin/env bash
# =============================================================================
# train_pi0_lambda.sh
#
# Fine-tune Pi0 on a chess pick-and-place dataset using LeRobot.
# Intended for a Lambda Labs A100 instance (80 GB VRAM).
#
# Dataset : izchen/chess_pick_place (51 episodes, 1280x720, 6-DOF joints)
# Base model: lerobot/pi0_base
# Output   : izchen/pi0_chess on HuggingFace Hub
#
# Usage:
#   chmod +x train_pi0_lambda.sh
#   ./train_pi0_lambda.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "==> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs build-essential

# ---------------------------------------------------------------------------
# 2. Clone LeRobot and install with Pi0 dependencies
# ---------------------------------------------------------------------------
LEROBOT_DIR="$HOME/lerobot"

if [ ! -d "$LEROBOT_DIR" ]; then
    echo "==> Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
else
    echo "==> LeRobot already cloned, pulling latest..."
    git -C "$LEROBOT_DIR" pull --ff-only
fi

echo "==> Installing LeRobot with Pi0 extras..."
cd "$LEROBOT_DIR"
pip install --upgrade pip
pip install -e ".[pi0]"

# ---------------------------------------------------------------------------
# 3. Log in to HuggingFace (required for dataset download and model push)
#    Set HF_TOKEN as an environment variable before running this script,
#    or the CLI will prompt you interactively.
# ---------------------------------------------------------------------------
echo "==> Logging in to HuggingFace Hub..."
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not set. Attempting interactive login..."
    huggingface-cli login
fi

# ---------------------------------------------------------------------------
# 4. Run training
#
#    Key settings:
#      - pi0 policy, pretrained from lerobot/pi0_base
#      - Vision encoder frozen to save memory and speed up training
#      - Batch size 8 fits comfortably on a single A100-80GB
#      - 20 000 gradient steps (~enough to converge on 51 episodes)
#      - Checkpoints saved every 5 000 steps
#      - Output directory: outputs/train/pi0_chess
# ---------------------------------------------------------------------------
OUTPUT_DIR="outputs/train/pi0_chess"

echo "==> Starting Pi0 fine-tuning (20k steps)..."
python -m lerobot.scripts.train \
    --policy.type=pi0 \
    --policy.path=lerobot/pi0_base \
    --policy.freeze_vision_encoder=true \
    --dataset.repo_id=izchen/chess_pick_place \
    --batch_size=8 \
    --steps=20000 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --log_freq=100 \
    --output_dir="$OUTPUT_DIR"

echo "==> Training complete. Checkpoints saved to $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 5. Push the final checkpoint to HuggingFace Hub
#
#    After training, the best/final checkpoint lives under the output dir.
#    We use huggingface-cli to upload it as a model repo.
# ---------------------------------------------------------------------------
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints/last/pretrained_model"
HF_REPO="izchen/pi0_chess"

echo "==> Pushing trained model to HuggingFace Hub ($HF_REPO)..."
huggingface-cli upload "$HF_REPO" "$CHECKPOINT_DIR" --repo-type model

echo "============================================="
echo "  Done! Model uploaded to: $HF_REPO"
echo "============================================="
