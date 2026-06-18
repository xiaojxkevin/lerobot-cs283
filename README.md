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

```
Gello raw (12-bit encoder + homing_offset)
  → Gello DEGREES normalization: (raw - range_mid) * 360 / 4095  → output degrees
    → Identity processor (1:1)
      → xArm revert: raw_xarm = degrees - xarm_homing_offset  → send to xArm
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
rm -rf out &&
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
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=16 \
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

## Deployment

Policy deployment uses `lerobot-rollout`. The policy type (`act`) is auto-detected from the checkpoint —
no need to specify `--policy.type` on the CLI.

### How it works

| Component | Detail |
|---|---|
| Inference | **Synchronous** (`--inference.type=sync`, default). Each control tick runs preprocessor → policy → postprocessor inline, then sends the action to the robot. |
| Temporal ensembling | **Inference-time only** — no retraining needed. Every step queries the policy for a new `chunk_size=100` action chunk, then fuses it with previous chunks via exponentially-weighted averaging (coeff = `0.01`, per the original ACT paper). This produces fully closed-loop control at ~9ms latency, well within the ~33ms budget at 30 FPS. |
| `n_action_steps=1` | Required by the temporal ensembler: `update()` pops 1 fused action per call, so the control loop must query every step. The model still predicts 100-step chunks every time — the ensemble fuses overlapping predictions for the same future timestep. |
| Action space | **Absolute joint positions** (7 DoF: j1–j6 degrees + gripper 0–100%). |
| Start position | Robot interpolates from its current pose to `--start_position` before the control loop begins. On shutdown, `--return_to_initial_position=true` (default) smoothly returns to this same pose. **Values are in raw hardware space** (as shown in xArm Studio) — the robot's calibration is applied internally to convert to the normalized joint space that the policy expects. |
| Multi-episode | `--num_rollouts=N` runs N episodes back-to-back. Between episodes the robot returns to `--start_position` and the policy is reset so every episode starts from a consistent state. |

### Rollout Controls

| Key | Action |
|---|---|
| **Space** | Start the next episode (after homing to start position) |
| **Right Arrow** | End current episode early → return to start → wait for Space |
| **Esc** | Stop rollout entirely |

### Run

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

## Deployment

Policy deployment uses `lerobot-rollout`. The policy type (`act`) is auto-detected from the checkpoint —
no need to specify `--policy.type` on the CLI.

### How it works

| Component | Detail |
|---|---|
| Inference | **Synchronous** (`--inference.type=sync`, default). Each control tick runs preprocessor → policy → postprocessor inline, then sends the action to the robot. |
| Temporal ensembling | **Inference-time only** — no retraining needed. Every step queries the policy for a new `chunk_size=100` action chunk, then fuses it with previous chunks via exponentially-weighted averaging (coeff = `0.01`, per the original ACT paper). This produces fully closed-loop control at ~9ms latency, well within the ~33ms budget at 30 FPS. |
| `n_action_steps=1` | Required by the temporal ensembler: `update()` pops 1 fused action per call, so the control loop must query every step. The model still predicts 100-step chunks every time — the ensemble fuses overlapping predictions for the same future timestep. |
| Action space | **Absolute joint positions** (7 DoF: j1–j6 degrees + gripper 0–100%). |
| Start position | Robot interpolates from its current pose to `--start_position` before the control loop begins. On shutdown, `--return_to_initial_position=true` (default) smoothly returns to this same pose. **Values are in raw hardware space** (as shown in xArm Studio) — the robot's calibration is applied internally to convert to the normalized joint space that the policy expects. |
| Multi-episode | `--num_rollouts=N` runs N episodes back-to-back. Between episodes the robot returns to `--start_position` and the policy is reset so every episode starts from a consistent state. |

### Rollout Controls

| Key | Action |
|---|---|
| **Space** | Start the next episode (after homing to start position) |
| **Right Arrow** | End current episode early → return to start → wait for Space |
| **Esc** | Stop rollout entirely |

### Run

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=checkpoints/history/120000/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main \
  --robot.cameras="{
    \"cam_arm\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"317622075882\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false},
    \"cam_front\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"231522072820\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false}
  }" \
  --start_position='{"j1.pos": -1.4, "j2.pos": 15.4, "j3.pos": -84.1, "j4.pos": -2.1, "j5.pos": 75.7, "j6.pos": 19.0, "gripper.pos": 400.0}' \
  --start_position_duration=3.0 \
  --fps=30 \
  --duration=30 \
  --num_rollouts=3
```

### Run with recording (sentry strategy)

Switch `--strategy.type` from `base` to `sentry` to record both camera streams
alongside robot state and policy actions:

```bash
rm -rf output &&
export DISPLAY=:1 &&
uv run lerobot-rollout \
  --strategy.type=sentry \
  --strategy.upload_every_n_episodes=5 \
  --policy.path=checkpoints/history/120000/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  --robot.type=xarm_follower \
  --robot.ip=192.168.2.202 \
  --robot.id=main \
  --robot.cameras="{
    \"cam_arm\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"317622075882\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false},
    \"cam_front\": {\"type\": \"intelrealsense\", \"serial_number_or_name\": \"231522072820\", \"fps\": 30, \"width\": 640, \"height\": 480, \"use_depth\": false}
  }" \
  --start_position='{"j1.pos": -1.4, "j2.pos": 15.4, "j3.pos": -84.1, "j4.pos": -2.1, "j5.pos": 75.7, "j6.pos": 19.0, "gripper.pos": 400.0}' \
  --start_position_duration=3.0 \
  --dataset.repo_id=local/rollout_history \
  --dataset.root=output \
  --dataset.single_task="put kettle on stove" \
  --dataset.fps=30 \
  --dataset.video=True \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=8 \
  --dataset.push_to_hub=false \
  --fps=30 \
  --duration=30 \
  --num_rollouts=40
```

Recorded datasets can be visualized with `lerobot-dataset-viz` and replayed with `lerobot-replay`.

### Key parameters

| Parameter | Description |
|---|---|
| `--policy.path` | Path to `pretrained_model/` directory |
| `--policy.temporal_ensemble_coeff` | `0.01` = standard ACT temporal ensembling (override from checkpoint `null`) |
| `--policy.n_action_steps` | Must be `1` with temporal ensembling (override from checkpoint `100`) |
| `--strategy.type` | `base` = inference only; `sentry` = inference + recording; `dagger` = human-in-the-loop |
| `--robot.ip` | xArm6 controller IP |
| `--robot.cameras` | Camera config dict (type, serial, resolution) |
| `--start_position` | Raw hardware values (copy from xArm Studio). Dict `{"j1.pos": raw_angle, ...}` or JSON file path |
| `--start_position_duration` | Seconds for the interpolation (default 3.0) |
| `--start_position_in_raw` | `true` (default) = values are raw hardware readings; auto-converted via calibration |
| `--fps` | Control loop frequency (match training data: 30) |
| `--duration` | Seconds per episode (`0` = infinite, until Ctrl+C or Right Arrow) |
| `--num_rollouts` | Number of episodes to run sequentially (default 1). Robot returns to start_position between episodes |
| `--display_data` | `true` to enable Rerun visualization |
| `--dataset.repo_id` | Dataset label (sentry/dagger modes; local identifier when `--dataset.push_to_hub=false`) |
| `--dataset.root` | Output directory for recorded data (sentry/dagger modes) |
| `--dataset.video` | `true` = store camera frames as video, more efficient than per-frame images |
| `--dataset.streaming_encoding` | `true` = encode video in background threads, avoids disk I/O blocking the control loop |
| `--dataset.push_to_hub` | `true` to auto-upload to Hugging Face Hub after each episode/session |

## Upstream

Built on [LeRobot](https://github.com/huggingface/lerobot) — an open-source library for end-to-end robot learning. See the [upstream documentation](https://huggingface.co/docs/lerobot/index) for policies, environments, and more.
