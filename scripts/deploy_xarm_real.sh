#!/usr/bin/env bash
#
# Deploy a trained ACT policy autonomously on the REAL xArm6 (no teleop, no Gello).
#
# Uses `lerobot-rollout` with the default SYNC inference engine, which drives the policy through
# `policy.select_action` once per control tick. This is the path that maintains ACT's observation-
# history queue, so it works transparently for both single-frame (n_obs_steps=1) and history
# (n_obs_steps>1) checkpoints — point --policy.path at the right checkpoint and nothing else changes.
#
# Runs in the project's uv environment (lerobot 0.5.2 + cuda torch). No conda env needed.
#
# !!!  SAFETY  !!!
#   - A learned policy can output large or unexpected motions, especially early in an episode.
#   - Keep a hand on the xArm e-stop / power. Clear the workspace. Start with the arm near the
#     training start pose.
#   - MAX_RELATIVE_TARGET caps how far each joint target may jump per control tick (in the policy's
#     degree space) — keep it conservative for the first runs and raise it once the motion looks safe.
#
# Usage:
#   bash scripts/deploy_xarm_real.sh                                   # autonomous, no recording
#   CHECKPOINT=outputs/train/act_cs283_history2/checkpoints/last/pretrained_model \
#       bash scripts/deploy_xarm_real.sh                               # deploy the history model
#   RECORD=1 USE_HUB=0 bash scripts/deploy_xarm_real.sh               # also record an eval dataset
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Policy ------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-outputs/train/act_cs283/checkpoints/last/pretrained_model}"

# --- Robot -------------------------------------------------------------------
ROBOT_IP="${ROBOT_IP:-192.168.2.202}"
ROBOT_ID="${ROBOT_ID:-main}"
# Per-tick safety cap on the joint target jump (policy degree space). Lower = safer/slower.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
CAM_ARM_SN="${CAM_ARM_SN:-317622075882}"
CAM_FRONT_SN="${CAM_FRONT_SN:-231522072820}"

# --- Runtime -----------------------------------------------------------------
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"          # seconds; 0 = run until Ctrl-C
TASK="${TASK:-put kettle on stove}" # ACT ignores this, but keep it descriptive
DEVICE="${DEVICE:-cuda}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

# --- Optional: record the rollout as an eval dataset -------------------------
# RECORD=1 switches from the 'base' strategy (no recording) to 'sentry' (always-on recording).
RECORD="${RECORD:-0}"
EVAL_REPO_ID="${EVAL_REPO_ID:-local/eval_cs283}"
EVAL_ROOT="${EVAL_ROOT:-out_eval}"
USE_HUB="${USE_HUB:-0}"

CAMERAS="{
  \"cam_arm\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"${CAM_ARM_SN}\", \"fps\": ${FPS}, \"width\": 640, \"height\": 480, \"use_depth\": false, \"color_mode\": \"rgb\"},
  \"cam_front\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"${CAM_FRONT_SN}\", \"fps\": ${FPS}, \"width\": 640, \"height\": 480, \"use_depth\": false, \"color_mode\": \"rgb\"}
}"

STRATEGY_ARGS=(--strategy.type=base)
DATASET_ARGS=()
if [[ "${RECORD}" == "1" ]]; then
  STRATEGY_ARGS=(--strategy.type=sentry)
  DATASET_ARGS=(
    --dataset.repo_id="${EVAL_REPO_ID}"
    --dataset.root="${EVAL_ROOT}"
    --dataset.single_task="${TASK}"
    --dataset.fps="${FPS}"
    --dataset.push_to_hub=$([[ "${USE_HUB}" == "1" ]] && echo true || echo false)
  )
fi

echo "Checkpoint     : ${CHECKPOINT}"
echo "Robot IP       : ${ROBOT_IP}"
echo "max_rel_target : ${MAX_RELATIVE_TARGET}"
echo "fps / duration : ${FPS} / ${DURATION}s"
echo "record dataset : ${RECORD}"
echo
echo ">>> SAFETY: keep a hand on the e-stop. Press Ctrl-C to abort. <<<"
echo

uv run lerobot-rollout \
  --policy.path="${CHECKPOINT}" \
  --robot.type=xarm_follower \
  --robot.ip="${ROBOT_IP}" \
  --robot.id="${ROBOT_ID}" \
  --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.cameras="${CAMERAS}" \
  --inference.type=sync \
  "${STRATEGY_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  --fps="${FPS}" \
  --duration="${DURATION}" \
  --task="${TASK}" \
  --device="${DEVICE}" \
  --display_data="${DISPLAY_DATA}"
