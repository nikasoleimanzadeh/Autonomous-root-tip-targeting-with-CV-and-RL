"""Fresh Path-B PPO replication for reinforcement learning control.

This deliberately reproduces the effective setup of the successful reference run
as closely as the submitted artifacts allow:

- fresh PPO policy (no resume from Nika checkpoints)
- PPO seed 42
- training env seeds 43, 44, 45, 46
- evaluation env seed 5043
- 4 randomized-start environments
- 600 max episode steps
- 1 mm success threshold
- 10 dwell steps
- [128, 128] policy/value MLP
- learning rate 3e-4
- n_steps 2048
- batch size 64
- 10 PPO epochs
- gamma 0.99
- GAE lambda 0.95
- clip range 0.2
- no target-KL early stopping

The existing OT2Env is used unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = HERE / "weights"
RESULTS_DIR = HERE / "results"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--total_timesteps", type=int, default=3_300_000)
    p.add_argument("--experiment_name", type=str, default="nika_pathB_exact_replication")

    p.add_argument("--checkpoint_freq", type=int, default=100_000)
    p.add_argument("--eval_freq", type=int, default=100_000)
    p.add_argument("--eval_episodes", type=int, default=10)

    return p.parse_args()


def make_env(
    seed: int,
    rank: int = 0,
    *,
    max_episode_steps: int = 600,
    success_threshold: float = 1.0e-3,
    dwell_steps: int = 10,
):
    def _make():
        env = OT2Env(
            seed=seed + rank,
            max_steps=max_episode_steps,
            success_threshold=success_threshold,
            dwell_steps=dwell_steps,
            randomise_start=True,
        )
        return Monitor(env)

    return _make


def main():
    args = parse_args()

    # These are intentionally fixed for this replication.
    PPO_SEED = 42
    TRAIN_ENV_BASE_SEED = 43
    EVAL_ENV_SEED = 5043
    N_ENVS = 4

    MAX_EPISODE_STEPS = 600
    SUCCESS_THRESHOLD = 1.0e-3
    DWELL_STEPS = 10

    LEARNING_RATE = 3.0e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.0
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NET_ARCH = [128, 128]

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tb_dir = RUNS_DIR / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = RUNS_DIR / "eval" / args.experiment_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv(
        [
            make_env(
                TRAIN_ENV_BASE_SEED,
                rank=i,
                max_episode_steps=MAX_EPISODE_STEPS,
                success_threshold=SUCCESS_THRESHOLD,
                dwell_steps=DWELL_STEPS,
            )
            for i in range(N_ENVS)
        ]
    )

    # Match the reference resume-training setup:
    # eval also uses a randomized start and a large seed offset.
    eval_env = DummyVecEnv(
        [
            make_env(
                EVAL_ENV_SEED,
                rank=0,
                max_episode_steps=MAX_EPISODE_STEPS,
                success_threshold=SUCCESS_THRESHOLD,
                dwell_steps=DWELL_STEPS,
            )
        ]
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs={"net_arch": NET_ARCH},
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=PPO_SEED,
        device="auto",
    )

    print("=== Task 11: Path-B exact-replication run ===")
    print(f"experiment:            {args.experiment_name}")
    print(f"fresh policy:          YES")
    print(f"PPO seed:              {PPO_SEED}")
    print(f"training env seeds:    43, 44, 45, 46")
    print(f"evaluation env seed:   {EVAL_ENV_SEED}")
    print(f"randomised eval start: YES")
    print(f"network:               {NET_ARCH}")
    print(f"learning rate:         {LEARNING_RATE}")
    print(f"n_steps:               {N_STEPS}")
    print(f"batch size:            {BATCH_SIZE}")
    print(f"n_epochs:              {N_EPOCHS}")
    print(f"gamma:                 {GAMMA}")
    print(f"GAE lambda:            {GAE_LAMBDA}")
    print(f"clip range:            {CLIP_RANGE}")
    print(f"target KL:             NONE")
    print(f"max episode steps:     {MAX_EPISODE_STEPS}")
    print(f"success threshold:     {SUCCESS_THRESHOLD * 1000:.1f} mm")
    print(f"dwell steps:           {DWELL_STEPS}")
    print(f"requested timesteps:   {args.total_timesteps:,}")
    print()

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // N_ENVS, 1),
            save_path=str(WEIGHTS_DIR),
            name_prefix=args.experiment_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(
                WEIGHTS_DIR / f"{args.experiment_name}_best"
            ),
            log_path=str(eval_dir),
            eval_freq=max(args.eval_freq // N_ENVS, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    manifest = {
        "experiment_name": args.experiment_name,
        "fresh_policy": True,
        "ppo_seed": PPO_SEED,
        "train_env_base_seed": TRAIN_ENV_BASE_SEED,
        "train_env_seeds": [43, 44, 45, 46],
        "eval_env_seed": EVAL_ENV_SEED,
        "randomise_start_training": True,
        "randomise_start_eval": True,
        "total_timesteps_requested": args.total_timesteps,
        "n_envs": N_ENVS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "success_threshold_mm": SUCCESS_THRESHOLD * 1000.0,
        "dwell_steps": DWELL_STEPS,
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "net_arch": NET_ARCH,
        "target_kl": None,
    }

    manifest_path = RESULTS_DIR / f"{args.experiment_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.experiment_name,
        reset_num_timesteps=True,
    )

    final_path = WEIGHTS_DIR / f"{args.experiment_name}_final.zip"
    model.save(str(final_path))

    shutil.copyfile(final_path, WEIGHTS_DIR / "latest.zip")

    manifest["actual_final_timesteps"] = int(model.num_timesteps)
    manifest["final_model"] = str(final_path)
    manifest["best_model_dir"] = str(
        WEIGHTS_DIR / f"{args.experiment_name}_best"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("=== TRAINING COMPLETE ===")
    print(f"actual final timesteps: {model.num_timesteps:,}")
    print(f"final model:            {final_path}")
    print(
        "best eval model:        "
        f"{WEIGHTS_DIR / f'{args.experiment_name}_best' / 'best_model.zip'}"
    )
    print(f"manifest:               {manifest_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
