"""Task 11: locked SAC versus P and PID on the SAME original 100 holdout cases.

Run at the repository root:
    python -B .\task11_rl_controller\compare_final_controllers.py

READS the final model and preserved test files; never trains, saves or edits model.
WRITES only a new timestamped task11_rl_controller/evidence/comparison_* folder.
The original holdout remains untouched and is never used for tuning gains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PATH_Y = HERE / "Path_Y"
for directory in (REPO / "src", PATH_Y):
    if not directory.is_dir():
        raise FileNotFoundError(f"Required dependency folder missing: {directory}")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ot2.envelope import PRACTICAL_ENVELOPE
from path_y_env import PathYEnv
from evaluate_sac import evaluate, summarize, save_csv

RUN = PATH_Y / "runs" / "final_push_20261012_155636_757034"
MODEL = RUN / "best_learned_sac_model.zip"
CASES = RUN / "holdout_cases_DO_NOT_TUNE_ON.json"
ORIGINAL_SUMMARY = RUN / "final_summary.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ProportionalController:
    """Fixed Kp=80, same observed physical-coordinate error as expert teacher."""
    def predict(self, observation, deterministic=True):
        error_m = np.asarray(observation[3:6], dtype=np.float64) * PRACTICAL_ENVELOPE.extent
        action = np.clip(80.0 * error_m, -1.0, 1.0).astype(np.float32)
        return action, None


class PIDController:
    """Explicit fixed-gain PID. NOT optimized against the holdout cases.

    Kp=80 (same as the known P baseline); Ki=0.25, Kd=0.01 are declared
    illustrative nonzero gains. Integral is anti-windup bounded. Velocity
    is decoded from the existing 12D observation; there is no extra sensor.
    Assumes the environment advances one physics step at 240 Hz.
    """
    def __init__(self):
        self.integral_error = np.zeros(3, dtype=np.float64)

    def reset(self):
        self.integral_error[:] = 0.0

    def predict(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float64)
        error_m = obs[3:6] * PRACTICAL_ENVELOPE.extent
        physical_velocity_mps = np.arctanh(np.clip(obs[9:12], -0.999999, 0.999999)) * 0.25
        self.integral_error = np.clip(self.integral_error + error_m / 240.0, -0.05, 0.05)
        action = 80.0 * error_m + 0.25 * self.integral_error - 0.01 * physical_velocity_mps
        return np.clip(action, -1.0, 1.0).astype(np.float32), None


def run_cases(controller, cases, label):
    """Separate resets for every case; same Path Y simulator and termination."""
    env = PathYEnv(start_strategy="uniform", max_steps=600)
    rows = []
    try:
        for i, case in enumerate(cases, start=1):
            if hasattr(controller, "reset"):
                controller.reset()
            rows.extend(evaluate(controller, env, [case]))
            if i % 10 == 0:
                print(f"{label}: {i}/{len(cases)}", flush=True)
    finally:
        env.close()
    return rows


def validate_rows(rows, cases, name):
    if len(rows) != len(cases):
        raise RuntimeError(f"{name}: wrong episode count")
    for row, case in zip(rows, cases):
        if row["case"] != case["case"] or row["group"] != case["group"]:
            raise RuntimeError(f"{name}: target alignment failed")
        if not np.isfinite(row["final_error_mm"]):
            raise RuntimeError(f"{name}: nonfinite position error")


def plot_results(all_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["SAC + imitation", "P (Kp=80)", "PID (fixed gains)"]
    keys = ["rl", "p", "pid"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    arrays = [np.asarray([r["final_error_mm"] for r in all_rows[k]], dtype=float) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(arrays, tick_labels=names, showfliers=True)
    ax.axhline(1.0, color="tab:red", linestyle="--", label="1 mm rubric threshold")
    ax.set_ylabel("Final Euclidean error (mm)")
    ax.set_title("Same 100 held-out targets: final positioning error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_final_error_distribution.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(keys))
    settle = [sum(r["settled"] for r in all_rows[k]) for k in keys]
    under = [sum(r["final_error_mm"] < 1.0 for r in all_rows[k]) for k in keys]
    ax.bar(x - .19, settle, width=.38, label="Settled (<1 mm for 10 steps)")
    ax.bar(x + .19, under, width=.38, label="Final error <1 mm")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Targets out of 100")
    ax.set_title("Success on identical holdout targets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_success_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    steps = [[r["steps"] for r in all_rows[k]] for k in keys]
    ax.boxplot(steps, tick_labels=names, showfliers=True)
    ax.set_ylabel("Simulation steps (maximum 600)")
    ax.set_title("Steps to successful settling, unsafe termination, or timeout")
    fig.tight_layout()
    fig.savefig(out_dir / "03_episode_steps.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(arrays[1], arrays[0], s=24, alpha=.7, c=colors[0])
    xymax = float(max(a.max() for a in arrays[:2]) * 1.05)
    ax.plot([0, xymax], [0, xymax], "--", color="gray", label="Equal final error")
    ax.axvline(1.0, color="tab:red", linestyle=":", alpha=.6)
    ax.axhline(1.0, color="tab:red", linestyle=":", alpha=.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("P controller final error (mm)")
    ax.set_ylabel("Learned SAC policy final error (mm)")
    ax.set_title("Paired errors: same start/target for each point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_paired_rl_vs_p.png", dpi=170)
    plt.close(fig)


def main():
    from stable_baselines3 import SAC

    parser = argparse.ArgumentParser(description="Final Task 11 controller comparison")
    parser.add_argument("--output", type=Path, default=None,
                        help="New, nonexistent directory for plots and CSVs")
    args = parser.parse_args()
    for required in (MODEL, CASES, ORIGINAL_SUMMARY):
        if not required.is_file():
            raise FileNotFoundError(f"STOP - protected evidence missing: {required}")

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_SUMMARY.read_text(encoding="utf-8"))
    if len(cases) != 100 or len({c["case"] for c in cases}) != 100:
        raise RuntimeError("STOP: expected 100 unique original holdout cases")
    locked_hash = sha256(MODEL)
    print("MODEL SHA256 BEFORE:", locked_hash)
    print("Reading original holdout. No hyperparameter tuning or training.")

    # A new separate output folder: never touch protected run directories.
    out = args.output or HERE / "evidence" / ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {out}")
    out.mkdir(parents=True, exist_ok=False)

    learned = SAC.load(str(MODEL), device="cpu")
    all_rows = {}
    try:
        all_rows["rl"] = run_cases(learned, cases, "SAC")
        validate_rows(all_rows["rl"], cases, "SAC")
        old = original["holdout_stats"]["all"]
        new = summarize(all_rows["rl"])["all"]
        if abs(new["mean_final_error_mm"] - old["mean_final_error_mm"]) > 0.001 or new["settled_count"] != old["settled_count"]:
            raise RuntimeError("STOP: SAC benchmark differs from original. Existing model untouched.")
        print("SAC REPRODUCED:", new["mean_final_error_mm"], "mm;", new["settled_count"], "settled")

        all_rows["p"] = run_cases(ProportionalController(), cases, "P")
        all_rows["pid"] = run_cases(PIDController(), cases, "PID")
        for key in ("p", "pid"):
            validate_rows(all_rows[key], cases, key)

        stats = {key: summarize(rows)["all"] for key, rows in all_rows.items()}
        for key, rows in all_rows.items():
            save_csv(out / f"{key}_holdout_100.csv", rows)
        with (out / "matched_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case", "group", "rl_final_error_mm", "p_final_error_mm", "pid_final_error_mm",
                             "rl_steps", "p_steps", "pid_steps", "rl_settled", "p_settled", "pid_settled"])
            for i in range(len(cases)):
                r, p, pid = (all_rows[k][i] for k in ("rl", "p", "pid"))
                writer.writerow([r["case"], r["group"], r["final_error_mm"], p["final_error_mm"], pid["final_error_mm"],
                                 r["steps"], p["steps"], pid["steps"], r["settled"], p["settled"], pid["settled"]])
        (out / "comparison_summary.json").write_text(json.dumps({
            "model_path": str(MODEL), "model_sha256": locked_hash,
            "holdout_file": str(CASES), "holdout_count": 100,
            "note": "SAC actor initialized by RL then improved using expert imitation; no teacher at inference."
                    " P: Kp=80 only. PID: Kp=80, Ki=0.25, Kd=0.01, assumed 240 Hz."
                    " PID gains fixed before running the holdout, not independently optimized.",
            "metrics": stats,
        }, indent=2), encoding="utf-8")
        plot_results(all_rows, out)
        print("\nCOMPARISON RESULTS")
        for key, s in stats.items():
            print(f"{key.upper():3s} | mean {s['mean_final_error_mm']:.4f} mm | "
                  f"settled {s['settled_count']}/100 | under 1mm {s['final_under_1mm_count']}/100 | "
                  f"unsafe {s['unsafe_count']} | mean steps {s['mean_steps']:.1f}")
        print("Evidence saved:", out.resolve())
    finally:
        hash_after = sha256(MODEL)
        print("MODEL SHA256 AFTER:", hash_after)
        if locked_hash != hash_after:
            raise RuntimeError("CRITICAL: model changed unexpectedly")
        print("MODEL UNCHANGED: PASS")


if __name__ == "__main__":
    main()
