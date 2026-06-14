"""Inspect Gello leader raw encoder values at different positions.

Usage:
    python scripts/inspect_gello.py --port /dev/ttyUSB0
    python scripts/inspect_gello.py --reset   # zero homing to see true encoder
    python scripts/inspect_gello.py --calibration /path/to/main.json
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus


def load_calibration(calib_path: str | None = None) -> dict | None:
    paths = []
    if calib_path:
        paths.append(Path(calib_path))
    paths.append(
        Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/gello_leader/main.json"
    )
    for p in paths:
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            return {
                name: MotorCalibration(
                    id=data["id"],
                    drive_mode=data["drive_mode"],
                    homing_offset=data["homing_offset"],
                    range_min=data["range_min"],
                    range_max=data["range_max"],
                )
                for name, data in raw.items()
            }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--calibration", default=None, help="Path to calibration JSON")
    parser.add_argument(
        "--reset", action="store_true",
        help="Temporarily zero all homing offsets to expose true encoder values"
    )
    args = parser.parse_args()

    calibration = load_calibration(args.calibration)

    bus = DynamixelMotorsBus(
        port=args.port,
        motors={
            "j1": Motor(1, "xl330-m288", MotorNormMode.DEGREES),
            "j2": Motor(2, "xl330-m288", MotorNormMode.DEGREES),
            "j3": Motor(3, "xl330-m288", MotorNormMode.DEGREES),
            "j4": Motor(4, "xl330-m288", MotorNormMode.DEGREES),
            "j5": Motor(5, "xl330-m288", MotorNormMode.DEGREES),
            "j6": Motor(6, "xl330-m288", MotorNormMode.DEGREES),
            "gripper": Motor(7, "xl330-m077", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )
    bus.connect()

    if args.reset:
        print("Zeroing all homing offsets to expose true encoder values...")
        bus.reset_calibration()
        calibration = None
    else:
        print("Connected. Press Ctrl+C to stop.")
        print("Tip: use --reset to zero homing offsets for true encoder values.\n")

    try:
        while True:
            raw = bus.sync_read("Present_Position", normalize=False)
            ho = bus.sync_read("Homing_Offset", normalize=False)
            norm = {}
            if calibration:
                norm = bus.sync_read("Present_Position", normalize=True)

            print("\033[2J\033[H")
            header = f"{'Motor':<10} | {'PP (raw)':>10} | {'Homing_Off':>12} | {'Encoder':>10}"
            if calibration:
                header += f" | {'mid':>8} | {'range':>16} | {'norm':>10}"
            print(header)
            print("-" * len(header))

            for name in ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]:
                pp = raw.get(name, float("nan"))
                home_off = ho.get(name, float("nan"))
                encoder = pp - home_off

                line = f"{name:<10} | {pp:>10} | {home_off:>12} | {encoder:>10}"
                if calibration:
                    c = calibration[name]
                    mid = (c.range_min + c.range_max) / 2
                    rng = f"[{c.range_min}, {c.range_max}]"
                    n = norm.get(name, float("nan"))
                    line += f" | {mid:>8.1f} | {rng:>16} | {n:>10.2f}"
                print(line)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
