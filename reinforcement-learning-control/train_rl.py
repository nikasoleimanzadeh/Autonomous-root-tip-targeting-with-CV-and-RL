"""Train an individual PPO controller for reinforcement learning control.

This script intentionally keeps experiment tracking local. It does not contain
ClearML/W&B credentials and does not depend on another student's saved model.

Run a short baseline first from the repository root:

    python task11_rl_controller/train_rl.py --total_timesteps 100000 \
        --experiment_name nika_ppo_baseline_100k

The best evaluation checkpoint and final model are written to
``task11_rl_controller/weights``. TensorBoard logs go to ``runs/tensorboard``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.envs.ot2_env import OT2Env

WEIGHTS_DIR = HERE / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = HERE / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the OT-2 Gymnasium environment.")

    # PPO hyperparameters.
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--net_arch", type=str, default="64,64")

    # Training setup.
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=244258)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--experiment_name", type=str, default="nika_ppo_baseline_100k")

    # Environment settings.
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--success_threshold_mm", type=float, default=1.0)
    parser.add_argument("--dwell_steps", type=int, default=10)

    # Evaluation/checkpointing.
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--checkpoint_freq", type=int, default=20_000)
    return parser.parse_args()


def make_env(seed: int, rank: int, args: argparse.Namespace, randomise_start: bool = True):
    """Return a thunk that creates one monitored OT-2 environment."""
    def _make():
        env = OT2Env(
            seed=seed + rank,
            max_steps=args.max_episode_steps,
            success_threshold=args.success_threshold_mm / 1000.0,
            dwell_steps=args.dwell_steps,
            randomise_start=randomise_start,
        )
        return Monitor(env)

    return _make


def main() -> None:
    args = parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tb_dir = RUNS_DIR / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = RUNS_DIR / "eval" / args.experiment_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=== Task 11 PPO training ===")
    print(f"experiment:          {args.experiment_name}")
    print(f"seed:                {args.seed}")
    print(f"timesteps:           {args.total_timesteps:,}")
    print(f"parallel envs:        {args.n_envs}")
    print(f"success threshold:   {args.success_threshold_mm:.3f} mm")
    print(f"dwell steps:          {args.dwell_steps}")
    print(f"network:              [{args.net_arch}]")
    print()

    env_fns = [make_env(args.seed, i, args, randomise_start=True) for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = vec_cls(env_fns)

    # Evaluation targets use a separate seed and a deterministic centre start.
    eval_env = DummyVecEnv([
        make_env(args.seed + 100_000, 0, args, randomise_start=False)
    ])

    net_arch = [int(v.strip()) for v in args.net_arch.split(",") if v.strip()]

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(WEIGHTS_DIR),
            name_prefix=args.experiment_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(WEIGHTS_DIR / f"{args.experiment_name}_best"),
            log_path=str(eval_dir),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.experiment_name,
    )

    final_path = WEIGHTS_DIR / f"{args.experiment_name}_final.zip"
    model.save(str(final_path))

    # Convenience alias for later evaluation scripts.
    latest_path = WEIGHTS_DIR / "latest.zip"
    shutil.copyfile(final_path, latest_path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "net_arch": net_arch,
        },
        "environment": {
            "max_episode_steps": args.max_episode_steps,
            "success_threshold_mm": args.success_threshold_mm,
            "dwell_steps": args.dwell_steps,
            "randomise_start_training": True,
            "randomise_start_evaluation": False,
        },
        "final_model": str(final_path.relative_to(PROJECT_ROOT)),
        "latest_model": str(latest_path.relative_to(PROJECT_ROOT)),
        "best_model_directory": str((WEIGHTS_DIR / f"{args.experiment_name}_best").relative_to(PROJECT_ROOT)),
    }
    manifest_path = RESULTS_DIR / f"{args.experiment_name}_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Training complete ===")
    print(f"final model:  {final_path}")
    print(f"latest alias: {latest_path}")
    print(f"manifest:     {manifest_path}")
    print(f"best eval:    {WEIGHTS_DIR / f'{args.experiment_name}_best' / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
