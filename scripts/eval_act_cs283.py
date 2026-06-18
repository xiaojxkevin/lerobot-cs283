#!/usr/bin/env python
"""Open-loop evaluation of a trained ACT policy on the cs283 dataset.

For each selected episode this replays the dataset observations through the policy exactly as on the
robot (`predict_action`: preprocess -> policy.select_action -> postprocess) and compares the predicted
action to the dataset's ground-truth action. It reports per-joint and overall MAE / MSE, per episode
and aggregated.

This is "open-loop" (a.k.a. teacher-forced) evaluation: observations always come from the dataset, so
errors do not compound the way they would in a closed-loop real-robot rollout. It is the quickest way
to sanity-check a checkpoint and to compare two checkpoints (e.g. single-frame vs history) on equal
footing. It works for any n_obs_steps: the policy rebuilds its temporal window internally from the
sequential per-frame observations via its observation queue (call order matters, so we reset() per
episode and feed frames in order).

Example:
    uv run python scripts/eval_act_cs283.py \
        --checkpoint outputs/train/act_cs283/checkpoints/last/pretrained_model --episodes all

    uv run python scripts/eval_act_cs283.py \
        --checkpoint outputs/train/act_cs283_history2/checkpoints/last/pretrained_model \
        --episodes 0,1,2
"""

import argparse

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop eval of an ACT policy on the cs283 dataset.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/act_cs283/checkpoints/last/pretrained_model",
        help="Path to a pretrained_model directory.",
    )
    p.add_argument("--dataset-root", type=str, default="./dataset/cs283")
    p.add_argument("--repo-id", type=str, default="local/cs283")
    p.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="'all' or a comma-separated list of episode indices, e.g. '0,1,2'.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames per episode (0 = no cap). For quick checks.")
    return p.parse_args()


def to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).numpy()


def eval_episode(policy, preprocessor, postprocessor, dataset, device, task, robot_type, max_frames):
    """Return (pred, gt) arrays of shape (N, action_dim) for one already-loaded single-episode dataset."""
    policy.reset()  # clear action + observation-history queues at the episode boundary
    preds, gts = [], []
    n = len(dataset)
    if max_frames > 0:
        n = min(n, max_frames)
    for i in range(n):
        frame = dataset[i]
        observation = {
            "observation.state": frame["observation.state"].numpy().astype(np.float32),
            "observation.images.cam_arm": to_hwc_uint8(frame["observation.images.cam_arm"]),
            "observation.images.cam_front": to_hwc_uint8(frame["observation.images.cam_front"]),
        }
        action = predict_action(
            observation, policy, device, preprocessor, postprocessor,
            use_amp=False, task=task, robot_type=robot_type,
        )
        preds.append(action.numpy().reshape(-1).astype(np.float32))
        gts.append(frame["action"].numpy().reshape(-1).astype(np.float32))
    return np.stack(preds), np.stack(gts)


def main():
    args = parse_args()
    device = torch.device(args.device)

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    motor_names = meta.info["features"]["action"]["names"]
    if args.episodes.strip().lower() == "all":
        episodes = list(range(meta.total_episodes))
    else:
        episodes = [int(x) for x in args.episodes.split(",") if x.strip() != ""]

    print(f"Loading policy from {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    policy.config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"n_obs_steps={policy.config.n_obs_steps}  chunk_size={policy.config.chunk_size}  "
          f"n_action_steps={policy.config.n_action_steps}")
    print(f"Evaluating {len(episodes)} episode(s): {episodes}\n")

    all_pred, all_gt = [], []
    header = f"{'episode':>8} | {'frames':>6} | {'MAE':>7} | {'MSE':>8}"
    print(header)
    print("-" * len(header))
    for ep in episodes:
        ds = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[ep])
        pred, gt = eval_episode(
            policy, preprocessor, postprocessor, ds, device, meta.tasks.index[0], meta.robot_type, args.max_frames
        )
        all_pred.append(pred)
        all_gt.append(gt)
        mae = np.mean(np.abs(pred - gt))
        mse = np.mean((pred - gt) ** 2)
        print(f"{ep:>8} | {pred.shape[0]:>6} | {mae:>7.3f} | {mse:>8.3f}")

    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    per_joint_mae = np.mean(np.abs(pred - gt), axis=0)
    per_joint_mse = np.mean((pred - gt) ** 2, axis=0)

    print("\n=== Aggregate over all evaluated frames ===")
    print(f"frames: {pred.shape[0]}   overall MAE: {np.mean(np.abs(pred - gt)):.3f}   "
          f"overall MSE: {np.mean((pred - gt) ** 2):.3f}")
    print(f"\n{'joint':<12} | {'MAE':>8} | {'MSE':>9}")
    print("-" * 34)
    for name, a, s in zip(motor_names, per_joint_mae, per_joint_mse):
        print(f"{name:<12} | {a:>8.3f} | {s:>9.3f}")


if __name__ == "__main__":
    main()
