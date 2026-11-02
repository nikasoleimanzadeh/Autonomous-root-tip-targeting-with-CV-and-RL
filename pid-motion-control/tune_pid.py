"""Tune the OT-2 per-axis PID controller on the local simulator.

Strategy
--------
Each axis is tuned independently around a strong PI seed. For every candidate,
the pipette is reset to the same centre pose and driven both below and above the
centre on that axis. Candidates are ranked using final positioning error first,
then settling speed. The chosen gains are saved to ``results/gains_final.json``.

Run from repository root:
    python task10_pid_controller/tune_pid.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.control.pid import AxisPID
from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_safe

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
MAX_STEPS = 450
SETTLE_M = 0.001       # rubric threshold: 1 mm
DWELL = 10

# Strong seed recovered from the supplied robotics scaffold. The search below
# validates and retunes it on this machine rather than blindly reusing results.
SEED = {
    "x": (72.73408306256486, 1323.153703458222, 0.0),
    "y": (77.05194653406970, 1485.2571151046516, 0.0),
    "z": (86.29272019286083, 1860.0441691493995, 0.0),
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_targets(axis: str) -> list[float]:
    i = AXIS_INDEX[axis]
    low = PRACTICAL_ENVELOPE.low[i]
    high = PRACTICAL_ENVELOPE.high[i]
    centre = PRACTICAL_ENVELOPE.center[i]
    # Two long moves plus two moderate moves. All remain safely inside envelope.
    span = high - low
    return [
        low + 0.08 * span,
        centre - 0.20 * span,
        centre + 0.20 * span,
        high - 0.08 * span,
    ]


def run_axis_episode(sim, axis: str, pid: AxisPID, target_value: float) -> dict:
    idx = AXIS_INDEX[axis]
    start = PRACTICAL_ENVELOPE.center.copy()
    teleport_safe(sim, start)
    pid.reset()

    consecutive = 0
    settle_step = None
    final_error = float("inf")
    peak_error = 0.0
    effort = 0.0

    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        error = float(target_value - pos[idx])
        abs_error = abs(error)
        peak_error = max(peak_error, abs_error)

        action = np.zeros(4, dtype=np.float64)
        action[idx] = pid.step(target_value, float(pos[idx]), DT)
        effort += abs(float(action[idx]))
        sim.run([action.tolist()], num_steps=1)

        post = precise_pipette_position(sim)
        final_error = abs(float(target_value - post[idx]))
        if final_error < SETTLE_M:
            consecutive += 1
            if consecutive >= DWELL and settle_step is None:
                settle_step = step + 1
        else:
            consecutive = 0

    return {
        "final_error_mm": final_error * 1000.0,
        "settle_step": settle_step,
        "settled": settle_step is not None,
        "effort_per_step": effort / MAX_STEPS,
        "peak_error_mm": peak_error * 1000.0,
    }


def score(rows: list[dict]) -> float:
    errors = np.array([r["final_error_mm"] for r in rows], dtype=float)
    settles = np.array([
        r["settle_step"] if r["settle_step"] is not None else MAX_STEPS * 2
        for r in rows
    ], dtype=float)
    failures = sum(not r["settled"] for r in rows)
    # Accuracy dominates. Settling speed breaks ties. Non-settling is expensive.
    return float(np.mean(errors) + 0.002 * np.mean(settles) + 10.0 * failures)


def main() -> None:
    print("Task 10 PID tuning")
    print("Practical envelope:", PRACTICAL_ENVELOPE)

    sim = make_sim(num_agents=1, render=False, rgb_array=False)
    all_rows: list[dict] = []
    chosen: dict[str, dict] = {}

    try:
        for axis in ("x", "y", "z"):
            print(f"\nTuning {axis.upper()} axis...")
            kp0, ki0, kd0 = SEED[axis]
            candidates = []

            for kp_factor in (0.80, 1.00, 1.20):
                for ki_factor in (0.80, 1.00, 1.20):
                    candidates.append((kp0 * kp_factor, ki0 * ki_factor, kd0))

            best = None
            for candidate_id, (kp, ki, kd) in enumerate(candidates, start=1):
                pid = AxisPID(
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    settle_threshold=0.0,
                    min_output=0.0,
                )
                rows = []
                for target in axis_targets(axis):
                    r = run_axis_episode(sim, axis, pid, target)
                    rows.append(r)
                    all_rows.append({
                        "axis": axis,
                        "candidate": candidate_id,
                        "kp": kp,
                        "ki": ki,
                        "kd": kd,
                        "target": target,
                        **r,
                    })

                candidate_score = score(rows)
                max_err = max(r["final_error_mm"] for r in rows)
                settled = sum(r["settled"] for r in rows)
                print(
                    f"  {candidate_id:02d}: kp={kp:.4f} ki={ki:.4f} kd={kd:.4f} | "
                    f"score={candidate_score:.6f} | max={max_err:.6f} mm | "
                    f"settled={settled}/{len(rows)}"
                )

                if best is None or candidate_score < best[0]:
                    best = (candidate_score, kp, ki, kd, rows)

            assert best is not None
            _, kp, ki, kd, rows = best
            chosen[axis] = {
                "kp": float(kp),
                "ki": float(ki),
                "kd": float(kd),
                "settle_threshold": 0.0,
                "min_output": 0.0,
            }
            print(
                f"BEST {axis.upper()}: kp={kp:.6f}, ki={ki:.6f}, kd={kd:.6f}, "
                f"max error={max(r['final_error_mm'] for r in rows):.6f} mm"
            )
    finally:
        sim.close()

    csv_path = RESULTS / "tuning_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    gains_path = RESULTS / "gains_final.json"
    gains_path.write_text(json.dumps(chosen, indent=2))

    print("\n✅ Tuning complete")
    print("Saved:", csv_path)
    print("Saved:", gains_path)
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
