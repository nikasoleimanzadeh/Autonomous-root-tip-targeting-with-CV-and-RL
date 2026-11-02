"""Path Y: short, independent SAC smoke test (no HER, no PPO, no teacher).

IMPORTANT: One PathYEnv / PyBullet instance only. Existing src/ot2/sim.py
reads some PyBullet state without physicsClientId, so two concurrent sim
instances could silently read from the wrong connection. Baseline evaluation,
training, and final evaluation therefore run SEQUENTIALLY on the same env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for loc in (REPO / "src", HERE):
    if str(loc) not in sys.path:
        sys.path.insert(0, str(loc))

from path_y_env import PathYEnv
from evaluate_sac import fixed_cases, evaluate, save_csv, summarize


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.logger import configure
    import stable_baselines3
    import gymnasium
    import torch

    parser = argparse.ArgumentParser(description="Independent Path Y SAC smoke test")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=244258)
    parser.add_argument("--start-strategy", choices=("uniform", "near_target"), default="uniform")
    args = parser.parse_args()
    if args.steps < 1000:
        parser.error("--steps must be >= 1000 for this smoke run")

    expected_r = -(0.75*.05/(.05+.05) + 0.25*.05/(.05+.002) + .005)
    if abs(PathYEnv.distance_reward(.05) - expected_r) > 1e-10:
        raise RuntimeError("Wrong Path Y reward version: expected bounded V2")

    run_dir = HERE / "runs" / ("sac_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    cases = fixed_cases(seed=7341, per_group=4)
    (run_dir / "eval_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    config = {
        "run_type": "smoke_only_NOT_final_model", "algorithm": "SAC", "policy": "MlpPolicy",
        "steps": args.steps, "seed": args.seed, "training_start_strategy": args.start_strategy,
        "learning_rate": 0.0003, "gamma": 0.99, "batch_size": 128,
        "buffer_size": 100000, "learning_starts": 500, "tau": 0.005,
        "train_freq": 1, "gradient_steps": 1, "ent_coef": "auto",
        "network": [128, 128], "max_episode_steps": 600,
        "reward": "V2 dense bounded, +5 settled, -120 unsafe",
        "single_simulator_instance": True,
        "versions": {"sb3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__,
                     "torch": torch.__version__, "numpy": np.__version__},
        "environment_sha256": hashlib.sha256((HERE / "path_y_env.py").read_bytes()).hexdigest(),
        "evaluation_seed": 7341, "near_cases": 4, "wide_cases": 4,
        "note": "Independent fixed evaluations; smoke run cannot establish 1-mm generalization.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    class EpisodeLogger(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.completed = []

        def _on_step(self):
            rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
            new_obs = np.asarray(self.locals["new_obs"], dtype=np.float64)
            if not np.isfinite(rewards).all() or not np.isfinite(new_obs).all():
                raise FloatingPointError("SAC produced NaN/Inf reward or observation")
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if bool(done):
                    self.completed.append({
                        "timestep": int(self.num_timesteps),
                        "reason": str(info.get("termination_reason")),
                        "final_error_mm": float(info["distance_mm"]),
                        "settled": bool(info["settled"]),
                        "unsafe": bool(info["unsafe"]),
                    })
            if self.num_timesteps % 500 == 0:
                recent = self.completed[-20:]
                print(f"Train {self.num_timesteps}/{args.steps} | finished={len(self.completed)} "
                      f"| last20 settled={sum(x['settled'] for x in recent)} "
                      f"unsafe={sum(x['unsafe'] for x in recent)}", flush=True)
            return True

    set_random_seed(args.seed)
    env = None
    model = None
    callback = EpisodeLogger()
    status = "not_started"
    try:
        env = PathYEnv(start_strategy=args.start_strategy, max_steps=600)
        if env.unsafe_penalty != 120.0 or env.settled_bonus != 5.0:
            raise RuntimeError("Expected Path Y V2 defaults: unsafe=120, bonus=5")
        env.reset(seed=args.seed)
        model = SAC(
            "MlpPolicy", env, device="cpu", seed=args.seed, verbose=0,
            learning_rate=0.0003, gamma=0.99, buffer_size=100000,
            learning_starts=500, batch_size=128, tau=0.005,
            train_freq=1, gradient_steps=1, ent_coef="auto",
            policy_kwargs={"net_arch": [128, 128]},
        )
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["csv"]))
        print("PATH Y SAC — SHORT SMOKE TEST", flush=True)
        print("Run folder:", run_dir, flush=True)
        print("Training starts:", args.start_strategy, flush=True)
        print("BASELINE deterministic policy (untrained) / SAME simulator:", flush=True)
        before_rows = evaluate(model, env, cases)
        save_csv(run_dir / "eval_before.csv", before_rows)
        before = summarize(before_rows)
        (run_dir / "summary_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
        print(f"BEFORE: settled={before['all']['settled_count']}/8, "
              f"mean error={before['all']['mean_final_error_mm']:.3f} mm", flush=True)
        status = "running"
        model.learn(total_timesteps=args.steps, callback=callback,
                    log_interval=1, progress_bar=False)
        status = "trained"
        print("FINAL deterministic evaluation / SAME simulator:", flush=True)
        after_rows = evaluate(model, env, cases)
        save_csv(run_dir / "eval_after.csv", after_rows)
        after = summarize(after_rows)
        (run_dir / "summary_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
        print(f"AFTER: settled={after['all']['settled_count']}/8, "
              f"mean error={after['all']['mean_final_error_mm']:.3f} mm, "
              f"unsafe={after['all']['unsafe_count']}", flush=True)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        print("CTRL+C: saving partial model/replay; skipping further evaluation.", flush=True)
    except Exception:
        status = "error"
        raise
    finally:
        if model is not None:
            model_name = "sac_smoke_final.zip" if status == "completed" else "sac_smoke_partial.zip"
            model.save(run_dir / model_name)
            model.save_replay_buffer(run_dir / "replay_buffer.pkl")
        if callback.completed:
            save_csv(run_dir / "training_episodes.csv", callback.completed)
        (run_dir / "run_status.json").write_text(json.dumps({
            "status": status,
            "timesteps": int(model.num_timesteps) if model is not None else 0,
            "finished_training_episodes": len(callback.completed),
            "train_settled": sum(r["settled"] for r in callback.completed),
            "train_unsafe": sum(r["unsafe"] for r in callback.completed),
        }, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
        print("Saved:", run_dir, flush=True)
    if status == "completed":
        print("SMOKE RUN COMPLETE — check summary_before/after.json and run_status.json.", flush=True)


if __name__ == "__main__":
    main()
