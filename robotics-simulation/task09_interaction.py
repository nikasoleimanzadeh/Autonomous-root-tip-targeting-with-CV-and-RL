"""Task 9: basic OT-2 interaction and observation processing.

Demonstrates:
- sending velocity/action commands to the simulator;
- reading pipette position and velocity;
- computing Euclidean distance to a target;
- observing how the robot state changes after commands.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_joint_velocity, precise_pipette_position

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def snapshot(sim, target: np.ndarray, label: str) -> dict:
    pos = precise_pipette_position(sim)
    vel = precise_joint_velocity(sim)
    distance = float(np.linalg.norm(target - pos))
    row = {
        "label": label,
        "position_m": pos.tolist(),
        "velocity_m_s": vel.tolist(),
        "target_m": target.tolist(),
        "distance_to_target_m": distance,
    }
    print(
        f"{label:>14} | "
        f"pos={np.round(pos, 6)} m | "
        f"vel={np.round(vel, 6)} | "
        f"distance={distance * 1000:.3f} mm"
    )
    return row


def main() -> None:
    sim = make_sim(num_agents=1, render=False, rgb_array=False)
    rows: list[dict] = []

    try:
        # Let the simulator settle.
        sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)

        target = PRACTICAL_ENVELOPE.center
        print("Practical-envelope centre target:", np.round(target, 6), "m")
        print()

        rows.append(snapshot(sim, target, "initial"))

        commands = [
            ("+X command", [0.35, 0.00, 0.00, 0], 80),
            ("-X command", [-0.35, 0.00, 0.00, 0], 80),
            ("+Y command", [0.00, 0.35, 0.00, 0], 80),
            ("-Y command", [0.00, -0.35, 0.00, 0], 80),
            ("+Z command", [0.00, 0.00, 0.35, 0], 80),
            ("-Z command", [0.00, 0.00, -0.35, 0], 80),
        ]

        for label, action, steps in commands:
            sim.run([action], num_steps=steps)
            rows.append(snapshot(sim, target, label))

        # Show the upstream observation dictionary too, because Task 9 asks us
        # to process observations returned by the environment.
        raw_state = sim.get_states()
        print("\nRaw simulator state keys:")
        print(raw_state)

        out = RESULTS / "interaction_observations.json"
        out.write_text(json.dumps(rows, indent=2))
        print("\n✅ Saved observations to", out)

    finally:
        sim.close()


if __name__ == "__main__":
    main()
