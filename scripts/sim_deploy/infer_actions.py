#!/usr/bin/env python
"""Stage 1 of SAPIEN deployment: run the trained policy over a dataset episode and dump actions.

This runs in the lerobot environment (py>=3.12). For every frame of the chosen episode it feeds the
observation (state + camera images) through the policy exactly as it would be done on the real robot
(`predict_action`: preprocess -> policy.select_action -> postprocess), and collects the predicted 7-D
action. The model's internal observation queue handles temporal history (n_obs_steps), so we feed one
frame at a time.

The predicted actions (and the dataset ground-truth actions, for comparison) are saved to an .npz that
stage 2 (`replay_xarm_sapien.py`, run in a SAPIEN env) reads back to drive the xArm6 URDF.

Example:
    uv run python scripts/sim_deploy/infer_actions.py \
        --checkpoint outputs/train/act_cs283/checkpoints/last/pretrained_model \
        --dataset-root ./dataset/cs283 --episode 0 \
        --out scripts/sim_deploy/actions_ep0.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    p = argparse.ArgumentParser(description="Run ACT over a dataset episode and dump actions for SAPIEN replay.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/act_cs283/checkpoints/last/pretrained_model",
        help="Path to a pretrained_model directory (contains config.json + model weights + processors).",
    )
    p.add_argument("--dataset-root", type=str, default="./dataset/cs283", help="LeRobot dataset root.")
    p.add_argument(
        "--repo-id",
        type=str,
        default="local/cs283",
        help="Dataset repo id label (ignored for loading when --dataset-root is a local dataset).",
    )
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    p.add_argument("--out", type=str, default="scripts/sim_deploy/actions_ep0.npz", help="Output .npz path.")
    p.add_argument("--device", type=str, default="cuda", help="Inference device.")
    return p.parse_args()


def to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """LeRobotDataset image (C, H, W) float in [0, 1] -> (H, W, C) uint8 expected by predict_action."""
    return (img.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)

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

    print(f"Loading episode {args.episode} from {args.dataset_root}")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[args.episode])
    task = dataset.meta.tasks.index[0] if hasattr(dataset.meta, "tasks") else ""
    motor_names = dataset.meta.info["features"]["action"]["names"]

    # Fresh episode: clear the policy's action / observation-history queues.
    policy.reset()

    preds, gts = [], []
    n = len(dataset)
    for i in range(n):
        frame = dataset[i]
        observation = {
            "observation.state": frame["observation.state"].numpy().astype(np.float32),
            "observation.images.cam_arm": to_hwc_uint8(frame["observation.images.cam_arm"]),
            "observation.images.cam_front": to_hwc_uint8(frame["observation.images.cam_front"]),
        }
        action = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=False,
            task=task,
            robot_type=dataset.meta.robot_type,
        )
        preds.append(action.numpy().reshape(-1).astype(np.float32))
        gts.append(frame["action"].numpy().reshape(-1).astype(np.float32))
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  frame {i + 1}/{n}")

    preds = np.stack(preds)  # (N, 7)
    gts = np.stack(gts)  # (N, 7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pred=preds,
        gt=gts,
        fps=dataset.fps,
        motor_names=np.array(motor_names),
        episode=args.episode,
    )
    mae = np.mean(np.abs(preds - gts), axis=0)
    print(f"\nSaved {preds.shape[0]} actions to {out}")
    print(f"Per-joint MAE (pred vs dataset gt): {np.round(mae, 3).tolist()}")
    print(f"Joints: {motor_names}")


if __name__ == "__main__":
    main()
