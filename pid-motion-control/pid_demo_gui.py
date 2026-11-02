"""GUI demonstration of the tuned PID control PID controller.

Screen-record this script for the PID control GIF evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_pipette_position
from controller import build_pid

DT = 1.0 / 240.0
MAX_STEPS = 600
SETTLE_M = 0.001
DWELL = 10


def run_target(sim, pid, target: np.ndarray, label: str) -> None:
    pid.reset()
    consecutive = 0
    settle_step = None

    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        action = pid.step(target, pos, dt=DT)
        sim.run([[float(action[0]), float(action[1]), float(action[2]), 0.0]], num_steps=1)

        post = precise_pipette_position(sim)
        error_m = float(np.linalg.norm(target - post))
        if error_m < SETTLE_M:
            consecutive += 1
            if consecutive >= DWELL:
                settle_step = step + 1
                break
        else:
            consecutive = 0

    reached = precise_pipette_position(sim)
    error_mm = float(np.linalg.norm(target - reached) * 1000.0)
    print(
        f"{label}: target={np.round(target, 5)} | reached={np.round(reached, 5)} | "
        f"error={error_mm:.4f} mm | settle_step={settle_step}"
    )


def main() -> None:
    pid = build_pid()
    e = PRACTICAL_ENVELOPE
    m = 0.03
    targets = [
        e.center,
        np.array([e.x_min + m, e.y_min + m, e.z_min + 0.02]),
        np.array([e.x_max - m, e.y_min + m, e.z_max - 0.02]),
        np.array([e.x_max - m, e.y_max - m, e.z_min + 0.02]),
        np.array([e.x_min + m, e.y_max - m, e.z_max - 0.02]),
        e.center,
    ]

    sim = make_sim(num_agents=1, render=True, rgb_array=False)
    try:
        print("Task 10 PID GUI demo")
        for i, target in enumerate(targets, start=1):
            run_target(sim, pid, target, f"target {i}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
