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

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_gello_leader import GelloLeaderTeleopConfig

logger = logging.getLogger(__name__)


class GelloLeader(Teleoperator):
    """Gello leader teleoperator using Dynamixel XL330 servos.

    The Gello leader is a haptic input device built from 7 Dynamixel XL330 servos
    (6 joints + 1 gripper) daisy-chained on a USB-FTDI serial bus. The operator
    physically moves the leader arm and joint positions are read to control a
    follower robot (typically an xArm6).
    """

    config_class = GelloLeaderTeleopConfig
    name = "gello_leader"

    def __init__(self, config: GelloLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "j1": Motor(1, "xl330-m288", norm_mode_body),
                "j2": Motor(2, "xl330-m288", norm_mode_body),
                "j3": Motor(3, "xl330-m288", norm_mode_body),
                "j4": Motor(4, "xl330-m288", norm_mode_body),
                "j5": Motor(5, "xl330-m288", norm_mode_body),
                "j6": Motor(6, "xl330-m288", norm_mode_body),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        print(
            "\n⚠️  IMPORTANT: For each joint, move it to its MECHANICAL MID-POINT "
            "(center of its physical range of motion), not to an extreme.\n"
            "The homing offset will set this position as 0 degrees."
        )
        input(f"Move {self} so each joint is at its mechanical mid-point, then press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        # Validate homing offsets — warn if any joint appears to be at an extreme
        max_res = 4095
        half_turn = max_res // 2  # 2047
        extreme_threshold = half_turn // 2  # ~1023 — offset larger than this is suspicious
        for motor, offset in homing_offsets.items():
            if abs(offset) > extreme_threshold:
                logger.warning(
                    f"Motor '{motor}' has homing_offset={offset}. "
                    f"This means the encoder was at ~{half_turn - offset} during homing, "
                    f"which is near an extreme (not mid-range). "
                    f"Consider re-running calibration with the joint at its mechanical center."
                )

        print(
            "\nNow move ALL joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        drive_modes = self.bus.sync_read("Drive_Mode", normalize=False)

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=int(drive_modes[motor]),
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()

        # Print calibration summary for verification
        print(f"\nCalibration saved to {self.calibration_fpath}")
        print(f"{'Motor':<10} | {'homing_offset':>14} | {'range_min':>10} | {'range_max':>10}")
        print("-" * 52)
        for motor in self.bus.motors:
            c = self.calibration[motor]
            print(f"{motor:<10} | {c.homing_offset:>14} | {c.range_min:>10} | {c.range_max:>10}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def get_raw_action(self) -> dict[str, float]:
        """Read raw Present_Position values without normalization (for debugging)."""
        raw = self.bus.sync_read("Present_Position", normalize=False)
        return {f"{motor}.raw": float(val) for motor, val in raw.items()}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
