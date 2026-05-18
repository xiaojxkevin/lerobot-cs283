<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

# LeRobot — Gello + xArm6 Teleoperation

This is a fork of [LeRobot](https://github.com/huggingface/lerobot) with added support for **Gello leader** (Dynamixel XL330 servos) teleoperating a **UFACTORY xArm6** follower.

## Hardware Setup

| Component | Hardware | Connection |
|---|---|---|
| **Leader** | Gello arm (7x Dynamixel XL330 servos) | USB-FTDI serial |
| **Follower** | UFACTORY xArm6 + gripper | Ethernet (default: 192.168.1.212) |
| **Cameras** | 2x Intel RealSense D415/D435 | USB 3.0 |

## Installation

```bash
uv pip install -e . -i http://mirrors.aliyun.com/pypi/simple/

# Install xArm-Python-SDK manually (from UFACTORY source)
uv pip install xarm-python-sdk

# Install Intel RealSense SDK
uv pip install pyrealsense2

# For data collection
uv pip install 'lerobot[dataset]' -i http://mirrors.aliyun.com/pypi/simple/
uv pip install 'lerobot[dynamixel]'
uv pip install rerun-sdk
```

## Calibration

Calibrate both devices before first use. Run each command separately:

```bash
# Calibrate the Gello leader arm
# Saved to ~/.cache/huggingface/lerobot/calibration/teleoperators/gello_leader/test.json
lerobot-calibrate \
  --teleop.type=gello_leader \
  --teleop.port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U1QU-if00-port0 \
  --teleop.id=test

# Calibrate the xArm6 follower
# ~/.cache/huggingface/lerobot/calibration/robots/xarm_follower/test.json
lerobot-calibrate \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=test
```

Follow the on-screen instructions to move each arm through its range of motion.

## Teleoperate

Basic teleoperation (no cameras):

```bash
lerobot-teleoperate \
  --teleop.type=gello_leader \
  --teleop.port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U1QU-if00-port0 \
  --teleop.id=main \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main
```

With cameras enabled:

```bash
lerobot-teleoperate \
  --teleop.type=gello_leader \
  --teleop.port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U1QU-if00-port0 \
  --teleop.id=main \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main \
  --robot.cameras="{
    \"cam_arm\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"317622075882\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false},
    \"cam_front\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"231522072820\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false}
  }" \
  --fps=30
```

## Record Data

Record a dataset by teleoperating the robot:

```bash
export DISPLAY=:1 &&
lerobot-record \
  --teleop.type=gello_leader \
  --teleop.port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U1QU-if00-port0 \
  --teleop.id=main \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main \
  --robot.cameras="{
    \"cam_arm\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"317622075882\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false, \"color_mode\": \"rgb\"},
    \"cam_front\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"231522072820\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false, \"color_mode\": \"rgb\"}
  }" \
  --dataset.repo_id=local/pick_and_place \
  --dataset.root=out \
  --dataset.single_task="put kettle on stove" \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=5 \
  --dataset.fps=30 \
  --dataset.video=True \
  --dataset.push_to_hub=false
```

### Recording Controls

| Key | Action |
|---|---|
| **Space** / **S** | Start the current episode |
| **Right Arrow** | End current episode early / reset |
| **Left Arrow** | Re-record current episode |
| **Esc** | Stop recording entirely |

## Replay

Replay a recorded episode on the xArm6:

```bash
lerobot-replay \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main \
  --dataset.repo_id=local/pick_and_place \
  --dataset.root=out \
  --dataset.episode=0
```

## Viz dataset

```bash
export DISPLAY=:1 &&
lerobot-dataset-viz \
  --repo-id pick_and_place \
  --episode-index 0 \
  --root ./out
```

Change `--dataset.episode` to replay a different episode (e.g., `num_episodes - 1` for the last one).

## Training

The merged dataset is at `./data/out_merged` (40 episodes, 14,320 frames, pick and place task).

`--policy.n_obs_steps` controls whether the model sees a single frame or a temporal history of frames. For local training, `--dataset.repo_id` is just a label — it doesn't affect data loading when `--dataset.root` is set.

### Standard ACT (single frame, `n_obs_steps=1`)

```bash
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=100 \
  --policy.image_resize_size="[256, 320]" \
  --dataset.repo_id=local/pick_place \
  --dataset.root=./data/0430 \
  --batch_size=64 \
  --steps=20000 \
  --num_workers=16 \
  --save_freq=200
```

### ACT with history frames (`n_obs_steps > 1`)

When `n_obs_steps > 1`, the model receives the last N frames with spatio-temporal position encodings, giving it temporal context for better action prediction:

```bash
lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.chunk_size=100 \
  --policy.image_resize_size="[256, 320]" \
  --dataset.repo_id=local/pick_place \
  --dataset.root=./data/0430 \
  --batch_size=16 \
  --steps=20000 \
  --num_workers=16 \
  --save_freq=2000
```

Set `--policy.n_obs_steps` to the desired number of history frames (e.g. 2, 3, 5). Higher values use more GPU memory since image token count scales as `n_obs_steps × H × W`.

### Key ACT parameters

| Parameter | Default | Description |
|---|---|---|
| `--policy.n_obs_steps` | `1` | History-frame toggle. `1` = single frame, `>1` = temporal history |
| `--policy.image_resize_size` | `None` | (H, W) to resize images before the backbone, e.g. `[256, 320]`. `None` = native 480×640 |
| `--policy.chunk_size` | `100` | Number of actions predicted per forward pass |
| `--policy.n_action_steps` | `100` | How many predicted actions to execute before re-querying |
| `--policy.dim_model` | `512` | Transformer hidden dimension |
| `--policy.n_encoder_layers` | `4` | Transformer encoder depth |
| `--policy.n_decoder_layers` | `1` | Transformer decoder depth |
| `--policy.use_vae` | `true` | Enable VAE for temporal smoothness; disable with `false` for speed |
| `--policy.kl_weight` | `10.0` | Weight for KL-divergence loss term |
| `--policy.optimizer_lr` | `1e-5` | Learning rate (transformer + action head) |
| `--policy.optimizer_lr_backbone` | `1e-5` | Learning rate for the vision backbone |

## Upstream

Built on [LeRobot](https://github.com/huggingface/lerobot) — an open-source library for end-to-end robot learning. See the [upstream documentation](https://huggingface.co/docs/lerobot/index) for policies, environments, and more.
