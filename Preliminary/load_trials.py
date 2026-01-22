#!/usr/bin/env python3
"""Load RGB-D trials and torque arrays for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


TORQUE_SCALE = 0.072


def _load_trial(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        return data.item()
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected npy payload format in {path}")


def _describe_array(arr: np.ndarray) -> str:
    return f"shape={arr.shape} dtype={arr.dtype}"


def _load_required_arrays(
    payload: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = payload.get("rgb")
    depth = payload.get("depth")
    timestamps = payload.get("timestamps")
    robot_state = payload.get("robot_state")

    if not isinstance(rgb, np.ndarray):
        raise ValueError("missing or invalid 'rgb' array")
    if not isinstance(depth, np.ndarray):
        raise ValueError("missing or invalid 'depth' array")
    if not isinstance(timestamps, np.ndarray):
        raise ValueError("missing or invalid 'timestamps' array")
    if not isinstance(robot_state, dict):
        raise ValueError("missing or invalid 'robot_state' dict")

    robot_times = robot_state.get("time")
    right_adduction = robot_state.get("rightadduction_curr")
    right_sweeping = robot_state.get("rightsweeping_curr")
    if not isinstance(robot_times, np.ndarray):
        raise ValueError("robot_state missing 'time' array")
    if not isinstance(right_adduction, np.ndarray):
        raise ValueError("robot_state missing 'rightadduction_curr' array")
    if not isinstance(right_sweeping, np.ndarray):
        raise ValueError("robot_state missing 'rightsweeping_curr' array")

    return rgb, depth, timestamps, robot_times, right_adduction, right_sweeping


def load_trial_arrays(path: Path) -> Dict[str, np.ndarray]:
    try:
        payload = _load_trial(path)
        rgb, depth, timestamps, robot_times, right_adduction, right_sweeping = _load_required_arrays(payload)
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc

    right_adduction_torque = right_adduction * TORQUE_SCALE
    right_sweeping_torque = right_sweeping * TORQUE_SCALE

    return {
        "rgb": rgb,
        "depth": depth,
        "timestamps": timestamps,
        "robot_time": robot_times,
        "rightadduction_torque": right_adduction_torque,
        "rightsweeping_torque": right_sweeping_torque,
    }


def _gather_trials(session_dir: Path) -> List[Path]:
    if session_dir.is_file():
        return [session_dir]
    return sorted(session_dir.glob("*.npy"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Load RGB-D trials and return torque arrays.")
    ap.add_argument(
        "session_dir",
        nargs="?",
        default="/home/qianlab/Desktop/session_preliminary/preliminary_data",
        type=Path,
        help="Session folder containing trial npy files (default: data/session_preliminary)",
    )
    args = ap.parse_args()

    session_dir = Path(str(args.session_dir).strip())
    trials = _gather_trials(session_dir)
    if not trials:
        print(f"No npy trials found under {session_dir}")
        return 1

    exit_code = 0
    for trial in trials:
        try:
            arrays = load_trial_arrays(trial)
        except ValueError as exc:
            print(f"{trial.name}: ERROR loading trial - {exc}")
            exit_code = 1
            continue
        print(f"{trial.name}: loaded")
        for key, value in arrays.items():
            print(f"  {key}: {_describe_array(value)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())