"""Task 9: move the OT-2 pipette to all 8 operational-workspace corners.

This is deliberately NOT the Task 10 PID controller. It uses a minimal clipped
proportional command only to demonstrate Task 9 environment interaction.

Outputs:
- results/eight_corners.csv
- results/eight_corners.gif (when the simulator exposes RGB frames)
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_joint_velocity, precise_pipette_position
from ot2.teleport import teleport_pipette

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
TOLERANCE_M = 0.001
DWELL_STEPS = 5
MAX_STEPS = 3000
KP_DEMO = 25.0
MAX_ACTION = 0.55
FRAME_EVERY = 12


def operational_corners() -> list[np.ndarray]:
    env = PRACTICAL_ENVELOPE
    return [
        np.array([x, y, z], dtype=np.float64)
        for x, y, z in product(
            [env.x_min, env.x_max],
            [env.y_min, env.y_max],
            [env.z_min, env.z_max],
        )
    ]


def _frame_with_text(frame, corner_idx: int, pos: np.ndarray,
                     vel: np.ndarray, target: np.ndarray, error_m: float):
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    text = (
        f"Task 9 - corner {corner_idx}/8\n"
        f"target: {target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f} m\n"
        f"position: {pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f} m\n"
        f"velocity: {vel[0]:+.4f}, {vel[1]:+.4f}, {vel[2]:+.4f}\n"
        f"distance: {error_m * 1000:.3f} mm"
    )
    draw.rectangle((8, 8, 510, 92), fill=(0, 0, 0))
    draw.multiline_text((14, 13), text, fill=(255, 255, 255), spacing=2)
    return np.asarray(im)


def drive_to(sim, target: np.ndarray, corner_idx: int, frames: list[np.ndarray]) -> dict:
    consecutive = 0
    closest = float("inf")

    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        vel = precise_joint_velocity(sim)
        error_vec = target - pos
        error = float(np.linalg.norm(error_vec))
        closest = min(closest, error)

        if error <= TOLERANCE_M:
            consecutive += 1
        else:
            consecutive = 0

        if step % FRAME_EVERY == 0:
            annotated = _frame_with_text(
                getattr(sim, "current_frame", None),
                corner_idx,
                pos,
                vel,
                target,
                error,
            )
            if annotated is not None:
                frames.append(annotated)

        if consecutive >= DWELL_STEPS:
            break

        # Simple Task-9 demonstration command, not the Task-10 PID.
        action_xyz = np.clip(KP_DEMO * error_vec, -MAX_ACTION, MAX_ACTION)
        sim.run(
            [[float(action_xyz[0]), float(action_xyz[1]), float(action_xyz[2]), 0]],
            num_steps=1,
        )
    else:
        step = MAX_STEPS - 1

    final_pos = precise_pipette_position(sim)
    final_vel = precise_joint_velocity(sim)
    final_error = float(np.linalg.norm(target - final_pos))

    annotated = _frame_with_text(
        getattr(sim, "current_frame", None),
        corner_idx,
        final_pos,
        final_vel,
        target,
        final_error,
    )
    if annotated is not None:
        frames.extend([annotated] * 10)

    print(
        f"corner {corner_idx}: target={np.round(target, 4)} | "
        f"reached={np.round(final_pos, 4)} | "
        f"error={final_error * 1000:.3f} mm | steps={step + 1}"
    )

    return {
        "corner": corner_idx,
        "target_x_m": float(target[0]),
        "target_y_m": float(target[1]),
        "target_z_m": float(target[2]),
        "reached_x_m": float(final_pos[0]),
        "reached_y_m": float(final_pos[1]),
        "reached_z_m": float(final_pos[2]),
        "final_error_mm": final_error * 1000.0,
        "steps": step + 1,
        "success_lt_1mm": bool(final_error < 0.001),
        "closest_error_mm": closest * 1000.0,
    }


def main() -> None:
    corners = operational_corners()
    frames: list[np.ndarray] = []
    records: list[dict] = []

    sim = make_sim(num_agents=1, render=True, rgb_array=False)
    try:
        sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)

        # Put the robot in a collision-safe central start. This reset is only
        # the starting condition; every corner itself is reached via commands.
        teleport_pipette(sim, PRACTICAL_ENVELOPE.center)
        sim.run([[0.0, 0.0, 0.0, 0]], num_steps=2)

        print("Operational work envelope (m):")
        print("X:", PRACTICAL_ENVELOPE.x_min, "to", PRACTICAL_ENVELOPE.x_max)
        print("Y:", PRACTICAL_ENVELOPE.y_min, "to", PRACTICAL_ENVELOPE.y_max)
        print("Z:", PRACTICAL_ENVELOPE.z_min, "to", PRACTICAL_ENVELOPE.z_max)
        print()

        for idx, target in enumerate(corners, start=1):
            records.append(drive_to(sim, target, idx, frames))

    finally:
        sim.close()

    csv_path = RESULTS / "eight_corners.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("\n✅ Saved corner coordinates:", csv_path)

    if frames:
        gif_path = RESULTS / "eight_corners.gif"
        imageio.mimsave(gif_path, frames, duration=0.08, loop=0)
        print("✅ Saved GIF:", gif_path)
    else:
        print(
            "⚠️ Simulator did not expose RGB frames. Corner measurements were "
            "saved, but the GIF needs to be captured with the simulator's GUI."
        )


if __name__ == "__main__":
    main()
