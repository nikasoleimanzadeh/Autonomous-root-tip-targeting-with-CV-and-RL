"""Independent, repeatable Path Y SAC evaluation on unseen start/target pairs.

This file does not train or modify a model. All coordinates are metres; reported
errors are millimetres. Near and wide-workspace samples use fixed independent
seeds; validation sets must not be used as training targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for loc in (REPO / "src", HERE):
    if str(loc) not in sys.path:
        sys.path.insert(0, str(loc))

from ot2.envelope import PRACTICAL_ENVELOPE
from path_y_env import PathYEnv


def fixed_cases(seed: int = 7341, per_group: int = 4) -> list[dict]:
    """Deterministic, independent near (20-60 mm) and wide (>=100 mm) cases."""
    rng = np.random.default_rng(seed)
    env_box = PRACTICAL_ENVELOPE
    starts_lo, starts_hi = env_box.low + .010, env_box.high - .010
    goals_lo, goals_hi = env_box.low + .005, env_box.high - .005
    rows = []
    for group in ("near", "wide"):
        for idx in range(per_group):
            for attempt in range(2000):
                target = rng.uniform(goals_lo, goals_hi)
                if group == "near":
                    direction = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    start = target + direction * rng.uniform(.020, .060)
                    valid = bool(np.all(start >= starts_lo) and np.all(start <= starts_hi))
                else:
                    start = rng.uniform(starts_lo, starts_hi)
                    valid = bool(np.linalg.norm(start - target) >= .100)
                if valid:
                    rows.append({"case": f"{group}_{idx+1:02d}", "group": group,
                                 "start": start.tolist(), "target": target.tolist()})
                    break
            else:
                raise RuntimeError(f"Could not sample {group} case {idx+1}")
    return rows


def evaluate(model, env: PathYEnv, cases: list[dict], max_steps: int = 600):
    """Predict deterministic actions only. No teacher actions or replay updates."""
    rows = []
    for case in cases:
        obs, info = env.reset(options={"start": case["start"], "target": case["target"]})
        start_mm = float(info["distance_mm"])
        best_mm = start_mm
        total_reward = 0.0
        reason = "time_limit"
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(action)):
                raise FloatingPointError("Non-finite SAC evaluation action")
            obs, reward, term, trunc, info = env.step(action)
            if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
                raise FloatingPointError("Non-finite evaluation reward/observation")
            best_mm = min(best_mm, float(info["distance_mm"]))
            total_reward += float(reward)
            if term or trunc:
                reason = str(info["termination_reason"])
                break
        rows.append({
            "case": case["case"], "group": case["group"],
            "start_distance_mm": round(start_mm, 6),
            "min_error_mm": round(best_mm, 6),
            "final_error_mm": round(float(info["distance_mm"]), 6),
            "steps": step, "settled": bool(info["settled"]),
            "unsafe": bool(info["unsafe"]), "reason": reason,
            "total_reward": round(total_reward, 6),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    def stats(items):
        arr = np.asarray([v["final_error_mm"] for v in items], dtype=np.float64)
        return {
            "count": len(items),
            "settled_count": sum(v["settled"] for v in items),
            "settled_rate": sum(v["settled"] for v in items) / len(items),
            "final_under_1mm_count": sum(v["final_error_mm"] < 1.0 for v in items),
            "unsafe_count": sum(v["unsafe"] for v in items),
            "mean_final_error_mm": float(arr.mean()),
            "median_final_error_mm": float(np.median(arr)),
            "max_final_error_mm": float(arr.max()),
            "mean_steps": float(np.mean([v["steps"] for v in items])),
        }
    return {"all": stats(rows), "near": stats([v for v in rows if v["group"] == "near"]),
            "wide": stats([v for v in rows if v["group"] == "wide"])}


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from stable_baselines3 import SAC
    p = argparse.ArgumentParser(description="Evaluate a saved Path Y SAC model, without training")
    p.add_argument("model", type=Path, help="Path to the trained SAC .zip")
    p.add_argument("--cases", type=int, default=10, help="Number of near AND wide cases")
    p.add_argument("--seed", type=int, default=7341)
    p.add_argument("--output", type=Path, default=HERE / "results" / "manual_eval")
    args = p.parse_args()
    if not args.model.is_file():
        p.error(f"Model missing: {args.model}")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = fixed_cases(seed=args.seed, per_group=args.cases)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    env = PathYEnv(max_steps=600, start_strategy="uniform")
    try:
        model = SAC.load(args.model, device="cpu")
        rows = evaluate(model, env, cases)
    finally:
        env.close()
    summary = summarize(rows)
    save_csv(args.output / "episodes.csv", rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
