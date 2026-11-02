"""Evaluate a trained PPO model on unseen OT-2 targets without rendering.

This is a fast numerical evaluation used after each training run. It records
per-target error and settling time and writes both CSV and JSON summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envs.ot2_env import OT2Env
from ot2.envelope import PRACTICAL_ENVELOPE, sample_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Task 11 PPO model.")
    parser.add_argument("--model", type=str, default=str(HERE / "weights" / "latest.zip"))
    parser.add_argument("--n_targets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=340258)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--success_threshold_mm", type=float, default=1.0)
    parser.add_argument("--dwell_steps", type=int, default=10)
    parser.add_argument("--tag", type=str, default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    env = OT2Env(
        seed=args.seed,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold_mm / 1000.0,
        dwell_steps=args.dwell_steps,
        randomise_start=False,
    )

    rng = np.random.default_rng(args.seed)
    targets = [sample_target(rng, envelope=PRACTICAL_ENVELOPE, margin=0.01) for _ in range(args.n_targets)]

    rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        obs, info = env.reset(options={"target": target})
        settled = False
        steps_used = 0

        for step in range(1, args.max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_used = step
            if terminated or truncated:
                settled = bool(terminated and info["dwell"] >= args.dwell_steps)
                break

        err_mm = float(info["distance_m"] * 1000.0)
        rows.append({
            "target_id": i,
            "target_x_m": float(target[0]),
            "target_y_m": float(target[1]),
            "target_z_m": float(target[2]),
            "final_x_m": float(info["pipette"][0]),
            "final_y_m": float(info["pipette"][1]),
            "final_z_m": float(info["pipette"][2]),
            "final_error_mm": err_mm,
            "steps": steps_used,
            "settled": settled,
            "success_under_1mm": err_mm < 1.0,
        })
        print(
            f"target {i:02d}: final={err_mm:8.3f} mm | "
            f"steps={steps_used:3d} | settled={settled}"
        )

    env.close()

    errors = np.asarray([r["final_error_mm"] for r in rows], dtype=float)
    successes = np.asarray([r["success_under_1mm"] for r in rows], dtype=bool)
    settled = np.asarray([r["settled"] for r in rows], dtype=bool)

    summary = {
        "model": str(model_path),
        "tag": args.tag,
        "seed": args.seed,
        "n_targets": args.n_targets,
        "mean_final_error_mm": float(errors.mean()),
        "median_final_error_mm": float(np.median(errors)),
        "max_final_error_mm": float(errors.max()),
        "min_final_error_mm": float(errors.min()),
        "success_rate_under_1mm_percent": float(successes.mean() * 100.0),
        "settled_rate_percent": float(settled.mean() * 100.0),
        "rubric_top_tier": bool(np.all(errors < 1.0)),
    }

    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"rl_eval_{args.tag}.csv"
    json_path = results_dir / f"rl_eval_{args.tag}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== RL evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
