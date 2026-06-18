#!/usr/bin/env bash
#
# Train an ACT policy WITH 2-step observation history (n_obs_steps=2) on the cs283 dataset.
#
# Identical to scripts/train_act_cs283.sh in every other setting (batch, steps, chunk, lr, etc.);
# the only change is --policy.n_obs_steps=2, plus a separate output dir / job name so it does not
# collide with the single-frame run. History feeds the last 2 frames (current + previous) with
# spatio-temporal position encodings.
#
# Runs through the project's own uv environment (lerobot 0.5.2 + cuda torch). No conda env needed.
#
# Usage:
#   bash scripts/train_act_cs283_history2.sh                 # train, no wandb
#   USE_WANDB=1 WANDB_ENTITY=<you> bash scripts/train_act_cs283_history2.sh   # log loss to wandb
#   RESUME=1 bash scripts/train_act_cs283_history2.sh        # resume from last checkpoint
#
# Override any default inline, e.g.:
#   STEPS=60000 BATCH_SIZE=4 bash scripts/train_act_cs283_history2.sh
#
set -euo pipefail

# --- Resolve repo root (this script lives in <repo>/scripts) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Tunable parameters (override via environment) ---------------------------
DATASET_ROOT="${DATASET_ROOT:-./dataset/cs283}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/cs283}"   # label only; data comes from DATASET_ROOT
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_cs283_history2}"
JOB_NAME="${JOB_NAME:-act_cs283_history2}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-120000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

N_OBS_STEPS="${N_OBS_STEPS:-2}"   # <-- the only functional difference from train_act_cs283.sh
CHUNK_SIZE="${CHUNK_SIZE:-100}"
DEVICE="${DEVICE:-cuda}"
# NOTE: history doubles the number of image tokens (2 cameras x 2 frames), so peak GPU memory is
# higher than the single-frame run. At batch 8 on the RTX 3090 (24 GB) this is fine; if you raise
# the batch and hit OOM, lower BATCH_SIZE.

# --- Weights & Biases (opt-in) -----------------------------------------------
# Enable with USE_WANDB=1. Set WANDB_ENTITY to your wandb user/team (optional).
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-cs283_act}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"   # online | offline | disabled

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--wandb.enable=true --wandb.project="${WANDB_PROJECT}" --wandb.mode="${WANDB_MODE}")
  [[ -n "${WANDB_ENTITY}" ]] && WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}")
fi

# --- Resume from last checkpoint (opt-in) ------------------------------------
# Enable with RESUME=1. Restores step / optimizer / scheduler and continues the
# same wandb run. All other settings come from the saved train_config.json, so we
# only pass --config_path and --resume; the full arg list below is skipped.
RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  RESUME_CFG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "${RESUME_CFG}" ]]; then
    echo "ERROR: no checkpoint to resume from at ${RESUME_CFG}" >&2
    echo "       (need at least one saved checkpoint under ${OUTPUT_DIR}/checkpoints/)" >&2
    exit 1
  fi
  echo "Resuming from  : ${RESUME_CFG}"
  exec uv run lerobot-train --config_path="${RESUME_CFG}" --resume=true
fi

# --- Launch (fresh run) ------------------------------------------------------
echo "Repo root      : ${REPO_ROOT}"
echo "Dataset root   : ${DATASET_ROOT}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Batch / steps  : ${BATCH_SIZE} / ${STEPS}"
echo "n_obs_steps    : ${N_OBS_STEPS}"
echo "wandb enabled  : ${USE_WANDB}"
echo

uv run lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device="${DEVICE}" \
  --policy.n_obs_steps="${N_OBS_STEPS}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  "${WANDB_ARGS[@]}"
