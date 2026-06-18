#!/usr/bin/env python
"""Stage 2 of SAPIEN deployment: replay policy-predicted actions on the xArm6 URDF.

This runs in a SAPIEN environment (e.g. the `grx-sim` conda env: py3.11, sapien 3.0.3). It loads the
xArm6 URDF and, for every action produced by stage 1 (`infer_actions.py`), drives the 6 arm joints
(joint1..joint6) kinematically via `set_qpos` and renders the scene in an interactive viewer.

Only the 6 arm joints are driven (the gripper is ignored, per the deployment spec). Dataset actions are
in degrees (the xArm calibration's normalized space) and are converted to radians for the URDF.

Run with the SAPIEN env's python, e.g.:
    /home/lxb/miniconda3/envs/grx-sim/bin/python scripts/sim_deploy/replay_xarm_sapien.py \
        --urdf scripts/sim_deploy/assets/xarm6/xarm6_with_gripper.urdf \
        --actions scripts/sim_deploy/actions_ep0.npz --source pred --loop
"""

import argparse
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import sapien

# xArm6 arm joints in URDF order; dataset action columns 0..5 (j1..j6) map to these in order.
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def parse_args():
    p = argparse.ArgumentParser(description="Replay predicted xArm6 joint actions in SAPIEN.")
    p.add_argument(
        "--urdf",
        type=str,
        default="scripts/sim_deploy/assets/xarm6/xarm6_with_gripper.urdf",
        help="Path to the xArm6 URDF.",
    )
    p.add_argument(
        "--actions",
        type=str,
        default="scripts/sim_deploy/actions_ep0.npz",
        help=".npz produced by infer_actions.py (keys: pred, gt, fps, motor_names).",
    )
    p.add_argument(
        "--source",
        choices=["pred", "gt"],
        default="pred",
        help="Replay the model predictions ('pred') or the dataset ground truth ('gt').",
    )
    p.add_argument("--fps", type=float, default=0.0, help="Playback fps. 0 = use the fps stored in the npz.")
    p.add_argument("--loop", action="store_true", help="Loop the replay until the viewer is closed.")
    return p.parse_args()


def preprocess_urdf(urdf_path: Path) -> str:
    """Make the pybullet xArm URDF loadable by SAPIEN, writing the result to a temp URDF.

    Two fixes are needed:
    1. Mesh URIs use ROS `package://<pkg>/...`, with the `xarm_description` / `xarm_gripper` folders
       sitting next to the URDF. SAPIEN does not resolve `package://`, so we rewrite the prefix to the
       URDF's directory.
    2. The `<transmission>` and `<gazebo>` blocks are ROS/Gazebo-only and irrelevant to SAPIEN. Worse,
       this particular URDF has a literal `<mechanicalReduction>reduction</mechanicalReduction>` typo
       that crashes SAPIEN's URDF parser. We strip both block types entirely.
    """
    text = urdf_path.read_text()
    urdf_dir = str(urdf_path.resolve().parent)
    text = text.replace("package://", urdf_dir + "/")
    text = re.sub(r"<transmission\b.*?</transmission>", "", text, flags=re.DOTALL)
    text = re.sub(r"<gazebo\b.*?</gazebo>", "", text, flags=re.DOTALL)
    # Mesh paths are now absolute, so the temp URDF can live anywhere (no repo pollution).
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def main():
    args = parse_args()

    data = np.load(args.actions, allow_pickle=True)
    actions = data[args.source]  # (N, 7)
    fps = args.fps if args.fps > 0 else float(data["fps"])
    dt = 1.0 / fps
    n = actions.shape[0]
    print(f"Loaded {n} '{args.source}' actions @ {fps} fps from {args.actions}")

    # --- Scene setup ---------------------------------------------------------
    scene = sapien.Scene()
    scene.set_timestep(dt)
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0, 0.5, -1], color=[3.0, 3.0, 3.0])
    scene.add_ground(0.0)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=1.2, y=0.0, z=0.8)
    viewer.set_camera_rpy(r=0, p=-0.4, y=3.14)

    # --- Load the xArm6 URDF (root fixed, kinematic visualization) -----------
    urdf_to_load = preprocess_urdf(Path(args.urdf))
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_to_load)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    # Map the 6 arm joints to their index in the full qpos vector.
    active_joints = robot.get_active_joints()
    name_to_idx = {j.get_name(): i for i, j in enumerate(active_joints)}
    missing = [j for j in ARM_JOINTS if j not in name_to_idx]
    if missing:
        raise RuntimeError(f"URDF is missing expected arm joints {missing}. Found: {list(name_to_idx)}")
    arm_idx = [name_to_idx[j] for j in ARM_JOINTS]

    # Per-joint limits, to clamp converted targets defensively.
    limits = robot.get_qlimits()  # (dof, 2)

    qpos = np.zeros(robot.dof, dtype=np.float32)

    def apply_frame(i: int):
        joints_deg = actions[i, :6]
        joints_rad = np.deg2rad(joints_deg)
        for k, idx in enumerate(arm_idx):
            lo, hi = limits[idx]
            qpos[idx] = float(np.clip(joints_rad[k], lo, hi))
        robot.set_qpos(qpos)

    print("Replaying. Close the viewer window to exit.")
    i = 0
    while not viewer.closed:
        t0 = time.perf_counter()
        apply_frame(i)
        scene.update_render()
        viewer.render()

        i += 1
        if i >= n:
            if not args.loop:
                # Hold the last frame and keep the window responsive.
                while not viewer.closed:
                    scene.update_render()
                    viewer.render()
                break
            i = 0

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
