"""Evaluate final Task 10 gains on random 3-D targets.

Outputs:
- results/pid_test_results.csv
- results/pid_summary.json
- results/pid_error_plot.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envelope import PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_pipette_position
from ot2.teleport import teleport_safe
from controller import build_pid

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 240.0
MAX_STEPS = 600
N_TARGETS = 20
SEED = 244258
SETTLE_M = 0.001
DWELL = 10


def run_episode(sim, pid, target: np.ndarray) -> dict:
    teleport_safe(sim, PRACTICAL_ENVELOPE.center)
    pid.reset()
    settle_step = None
    consecutive = 0
    effort = 0.0
    trajectory_error_mm = []

    for step in range(MAX_STEPS):
        pos = precise_pipette_position(sim)
        action = pid.step(target, pos, dt=DT)
        effort += float(np.sum(np.abs(action)))
        sim.run([[float(action[0]), float(action[1]), float(action[2]), 0.0]], num_steps=1)

        post = precise_pipette_position(sim)
        err_m = float(np.linalg.norm(target - post))
        trajectory_error_mm.append(err_m * 1000.0)

        if err_m < SETTLE_M:
            consecutive += 1
            if consecutive >= DWELL and settle_step is None:
                settle_step = step + 1
        else:
            consecutive = 0

    return {
        "final_error_mm": trajectory_error_mm[-1],
        "min_error_mm": float(np.min(trajectory_error_mm)),
        "settle_step": settle_step,
        "settle_time_s": None if settle_step is None else settle_step * DT,
        "settled_under_1mm": settle_step is not None,
        "effort_per_step": effort / MAX_STEPS,
    }


def main() -> None:
    pid = build_pid()
    rng = np.random.default_rng(SEED)
    targets = [sample_target(rng, margin=0.01) for _ in range(N_TARGETS)]

    rows = []
    sim = make_sim(num_agents=1, render=False, rgb_array=False)
    try:
        for i, target in enumerate(targets, start=1):
            r = run_episode(sim, pid, target)
            row = {
                "target_id": i,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_z": float(target[2]),
                **r,
            }
            rows.append(row)
            print(
                f"target {i:02d}: final={r['final_error_mm']:.6f} mm | "
                f"settle={r['settle_step']} steps | success={r['settled_under_1mm']}"
            )
    finally:
        sim.close()

    errors = np.array([r["final_error_mm"] for r in rows], dtype=float)
    settle_times = np.array([
        r["settle_time_s"] for r in rows if r["settle_time_s"] is not None
    ], dtype=float)
    success_rate = 100.0 * np.mean([r["settled_under_1mm"] for r in rows])

    summary = {
        "n_targets": N_TARGETS,
        "mean_final_error_mm": float(np.mean(errors)),
        "median_final_error_mm": float(np.median(errors)),
        "max_final_error_mm": float(np.max(errors)),
        "success_rate_under_1mm_percent": float(success_rate),
        "mean_settle_time_s": None if len(settle_times) == 0 else float(np.mean(settle_times)),
        "max_settle_time_s": None if len(settle_times) == 0 else float(np.max(settle_times)),
        "rubric_top_tier": bool(np.max(errors) < 1.0 and success_rate == 100.0),
    }

    csv_path = RESULTS / "pid_test_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS / "pid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_TARGETS + 1), errors, marker="o")
    plt.axhline(1.0, linestyle="--", label="1 mm rubric threshold")
    plt.xlabel("Target")
    plt.ylabel("Final positioning error (mm)")
    plt.title("Task 10 PID positioning error across random targets")
    plt.legend()
    plt.tight_layout()
    plot_path = RESULTS / "pid_error_plot.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("\n✅ PID evaluation complete")
    print(json.dumps(summary, indent=2))
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Saved:", plot_path)


if __name__ == "__main__":
    main()
