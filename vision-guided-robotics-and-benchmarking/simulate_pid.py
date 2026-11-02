"""Run the original PID control PI controller on already converted CV targets.

Uses robotics simulation make_sim/teleport_safe/precise_pipette_position. Simulates a deposit
marker only *after* 10 consecutive observations within 1 mm. Does NOT model
liquid dispensing or biological placement error.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np

DT = 1.0 / 240.0
THRESHOLD_M = 0.001
DWELL_STEPS = 10
MAX_STEPS_PER_PHASE = 600
SAFE_Z_M = 0.245


def move_pid(sim, pid, target: np.ndarray, on_step: Callable | None = None,
             max_steps: int = MAX_STEPS_PER_PHASE) -> dict:
    """Reset PID per waypoint and reach target with a 10-step accuracy dwell."""
    from ot2.sim import precise_pipette_position
    target = np.asarray(target, dtype=np.float64)
    pid.reset()
    dwell = 0
    first_settle = None
    final_error = float('inf')
    for step in range(1, max_steps + 1):
        pos = precise_pipette_position(sim)
        a = pid.step(target, pos, dt=DT)
        sim.run([[float(a[0]), float(a[1]), float(a[2]), 0.0]], num_steps=1)
        new_pos = precise_pipette_position(sim)
        final_error = float(np.linalg.norm(target - new_pos))
        if on_step is not None:
            on_step(new_pos, target, step, final_error * 1000.0)
        if final_error < THRESHOLD_M:
            dwell += 1
            if dwell >= DWELL_STEPS:
                first_settle = step
                break
        else:
            dwell = 0
    return {"settled": first_settle is not None,
            "steps": step, "time_s": step * DT,
            "final_error_mm": final_error * 1000.0,
            "reached_xyz_m": precise_pipette_position(sim).tolist()}


def run_plate(sim, pid, targets: list[dict], renderer=None) -> list[dict]:
    """Hover over each root tip, mark a simulated deposit only after success."""
    from ot2.envelope import PRACTICAL_ENVELOPE
    from ot2.sim import precise_pipette_position
    if not PRACTICAL_ENVELOPE.z_min <= SAFE_Z_M <= PRACTICAL_ENVELOPE.z_max:
        raise ValueError("Safe clearance Z outside Task 9 workspace")
    records = []
    for row in targets:
        goal = np.array([row["target_x_m"], row["target_y_m"], row["target_z_m"]])
        if goal[2] > SAFE_Z_M:
            raise ValueError("Target hover Z is above transit clearance Z")
        transit = np.array([goal[0], goal[1], SAFE_Z_M])
        cb = renderer.on_step if renderer is not None else None
        hover = move_pid(sim, pid, transit, on_step=cb)
        reach = {"settled": False, "final_error_mm": float('nan'),
                 "steps": 0, "time_s": 0.0,
                 "reached_xyz_m": precise_pipette_position(sim).tolist()}
        if hover["settled"]:
            reach = move_pid(sim, pid, goal, on_step=cb)
        deposited = bool(hover["settled"] and reach["settled"])
        if deposited and renderer is not None:
            renderer.mark_deposit(goal, row)
        retreat = move_pid(sim, pid, transit, on_step=cb)
        records.append({**row, "approach_settled": hover["settled"],
                        "hover_settled": reach["settled"],
                        "simulated_deposit": deposited,
                        "position_error_mm": reach["final_error_mm"],
                        "reached_xyz_m": reach["reached_xyz_m"],
                        "approach_steps": hover["steps"],
                        "hover_steps": reach["steps"],
                        "retreat_steps": retreat["steps"],
                        "simulated_motion_time_s": hover["time_s"] + reach["time_s"] + retreat["time_s"],
                        "retreat_settled": retreat["settled"]})
        if not retreat["settled"]:
            raise RuntimeError(f"Failed to retreat to safe Z after {row['plate']}, plant {row['plant']}")
    return records
