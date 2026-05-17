#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import time
from enum import Enum
from functools import cached_property
from pathlib import Path

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xarm_follower import XarmFollowerRobotConfig

logger = logging.getLogger(__name__)

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270
# For joints in percentage (i.e. linear joints like gripper),
# their nominal range is [0, 100] %.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

MODEL_RESOLUTION = {
    "xarm": 360,
    "xarm_gripper": 800,
}

GRIPPER_OPEN = 800
GRIPPER_CLOSE = 0
XARM_GRIPPER_WRITE_EPS = float(os.getenv("LEROBOT_XARM_GRIPPER_WRITE_EPS", "1.0"))


class CalibrationMode(Enum):
    DEGREE = 0
    LINEAR = 1


class JointOutOfRangeError(Exception):
    pass


MOTOR_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


class XarmFollower(Robot):
    """xArm6 follower robot controlled via Ethernet using the UFACTORY xArm SDK.

    The xArm6 has 6 servo joints plus a gripper. Communication is over TCP/IP
    using the xArm Python SDK (XArmAPI), not through the MotorsBus abstraction.
    """

    config_class = XarmFollowerRobotConfig
    name = "xarm_follower"

    def __init__(self, config: XarmFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot = None
        self._last_gripper_goal = None
        self._gripper_enabled = True
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.robot is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        require_package("xarm-python-sdk", extra="xarm", import_name="xarm")
        from xarm.wrapper import XArmAPI

        self.robot = XArmAPI(self.config.ip)
        self.robot.set_mode(6)
        time.sleep(1)
        self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        self.robot.set_state(state=0)
        time.sleep(1)
        if self._gripper_enabled:
            self.robot.set_gripper_enable(True)
            time.sleep(1)
            self.robot.set_gripper_mode(0)
            time.sleep(1)
            self.robot.set_gripper_speed(3000)
            time.sleep(1)
            self.robot.set_gripper_position(GRIPPER_OPEN, wait=False)

        if not self.is_calibrated and calibrate:
            logger.info("No calibration file found for xarm follower.")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration file for {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        print("Move the xArm6 to its zero position (all joints at 0 degrees, gripper closed) and press ENTER.")
        input()
        code, zero_angles = self.robot.get_servo_angle()
        code, zero_gripper = self.robot.get_gripper_position()
        zero_positions = list(zero_angles)
        zero_positions[-1] = zero_gripper

        print("Now move each joint through its full range of motion to record limits.")
        print("Press ENTER when done.")
        input()

        code, end_angles = self.robot.get_servo_angle()
        code, end_gripper = self.robot.get_gripper_position()

        # Build calibration: for DEGREE joints, homing_offset = -zero_position
        # For LINEAR gripper, store start/end range
        self.calibration = {
            "motor_names": list(MOTOR_NAMES),
            "calib_mode": ["DEGREE"] * 6 + ["LINEAR"],
            "homing_offset": [
                float(-zero_positions[0]),
                float(-zero_positions[1]),
                float(-zero_positions[2]),
                float(-zero_positions[3]),
                float(-zero_positions[4]),
                float(-zero_positions[5]),
                0.0,
            ],
            "drive_mode": [0] * 7,
            "start_pos": [0.0] * 6 + [float(zero_gripper)],
            "end_pos": [0.0] * 6 + [float(end_gripper)],
        }
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f:
            self.calibration = json.load(f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f:
            json.dump(self.calibration, f, indent=4)

    def configure(self) -> None:
        pass

    def _apply_calibration(self, values: np.ndarray) -> np.ndarray:
        """Convert from raw xArm values to normalized range.
        For DEGREE joints: normalized = raw_angle + homing_offset (both in degrees, range [-180, 180]).
        For LINEAR gripper: normalized = (raw_pos - start_pos) / (end_pos - start_pos) * 100 (range [0, 100]).
        """
        values = values.astype(np.float64)
        for i, name in enumerate(MOTOR_NAMES):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                resolution = MODEL_RESOLUTION["xarm"]

                if drive_mode:
                    values[i] *= -1

                values[i] = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE

                if values[i] < LOWER_BOUND_DEGREE or values[i] > UPPER_BOUND_DEGREE:
                    raise JointOutOfRangeError(
                        f"Joint {name} out of range: {values[i]:.1f} degrees "
                        f"(expected [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}]). "
                        "Try recalibrating with `lerobot-calibrate`."
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if values[i] < LOWER_BOUND_LINEAR or values[i] > UPPER_BOUND_LINEAR:
                    raise JointOutOfRangeError(
                        f"Gripper out of range: {values[i]:.1f} % "
                        f"(expected [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}]). "
                        "Try recalibrating with `lerobot-calibrate`."
                    )

        return values

    def _revert_calibration(self, values: np.ndarray) -> np.ndarray:
        """Inverse of `_apply_calibration` — convert normalized values back to raw xArm values."""
        values = values.astype(np.float64)
        for i, name in enumerate(MOTOR_NAMES):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                resolution = MODEL_RESOLUTION["xarm"]

                values[i] = values[i] / HALF_TURN_DEGREE * (resolution // 2)
                values[i] -= homing_offset

                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        return values

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        code, servo_angle = self.robot.get_servo_angle()
        code, gripper_pos = self.robot.get_gripper_position()

        # Build raw values array: [j1, j2, j3, j4, j5, j6, gripper]
        raw_values = np.array(list(servo_angle) + [gripper_pos], dtype=np.float64)

        if self.calibration:
            raw_values = self._apply_calibration(raw_values)

        obs_dict = {f"{name}.pos": float(raw_values[i]) for i, name in enumerate(MOTOR_NAMES)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): key for key in action if key.endswith(".pos")}
        # Build array in motor name order
        raw_values = np.array([action.get(f"{name}.pos", 0.0) for name in MOTOR_NAMES], dtype=np.float64)

        if self.calibration:
            raw_values = self._revert_calibration(raw_values)

        # Cap goal position when too far away from present position (safety)
        if self.config.max_relative_target is not None:
            code, servo_angle = self.robot.get_servo_angle()
            code, gripper_pos = self.robot.get_gripper_position()
            present_raw = np.array(list(servo_angle) + [gripper_pos], dtype=np.float64)
            if self.calibration:
                present_norm = self._apply_calibration(present_raw.copy())
            else:
                present_norm = present_raw

            goal_present_pos = {
                f"{name}.pos": (float(raw_values[i]) if not self.calibration
                                else float(action.get(f"{name}.pos", 0.0)),
                                float(present_norm[i]))
                for i, name in enumerate(MOTOR_NAMES)
            }
            safe_positions = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            raw_values = np.array([safe_positions[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float64)
            if self.calibration:
                raw_values = self._revert_calibration(raw_values)

        # Send to xArm
        joint_values = raw_values[:6].tolist()
        gripper_value = float(raw_values[6])

        self.robot.set_servo_angle(angle=joint_values, wait=False)

        if (
            self._last_gripper_goal is None
            or abs(gripper_value - self._last_gripper_goal) >= XARM_GRIPPER_WRITE_EPS
        ):
            self.robot.set_gripper_position(gripper_value, wait=False)
            self._last_gripper_goal = gripper_value

        # Return the action actually sent (normalized)
        if self.calibration:
            normalized = self._apply_calibration(raw_values.copy())
            return {f"{name}.pos": float(normalized[i]) for i, name in enumerate(MOTOR_NAMES)}
        return {f"{name}.pos": float(raw_values[i]) for i, name in enumerate(MOTOR_NAMES)}

    @check_if_not_connected
    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
