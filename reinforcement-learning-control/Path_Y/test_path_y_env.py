"""Pre-training validation for Path Y's standalone Gymnasium environment.

Runs the SAME simulator/action pipeline as Phase 1, but NO RL training.
Expected environment (Windows PowerShell):
    $env:OT2_SIM_DIR='C:\\Users\\ASUS\\Y2B-2023-OT2_Twin'
    python task11_rl_controller\\Path_Y\\test_path_y_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from path_y_env import PathYEnv
from ot2.envelope import PRACTICAL_ENVELOPE


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def check_reward() -> None:
    distances = np.array([0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1])
    rewards = [PathYEnv.distance_reward(float(x)) for x in distances]
    require(all(a > b for a, b in zip(rewards[:-1], rewards[1:])), "Reward is not monotonic")
    print("1) Dense reward at distances (mm / reward):")
    for d, r in zip(distances, rewards):
        print(f"   {d * 1000:7.3f} mm -> {r:8.4f}")
    print("   PASS: no positive reward wells or sparse plateaus")


def check_sim_env() -> None:
    env = PathYEnv(start_strategy="uniform", max_steps=600)
    try:
        print("2) SB3 Gymnasium check_env...")
        check_env(env, warn=True)
        print("   PASS")

        print("3) Deterministic reset and physical velocity representation...")
        o1, i1 = env.reset(seed=244258)
        o2, i2 = env.reset(seed=244258)
        require(np.allclose(i1["target"], i2["target"], atol=1e-10), "Target sampling not reproducible")
        require(np.allclose(i1["start"], i2["start"], atol=1e-10), "Start sampling not reproducible")
        require(env.observation_space.contains(o2), "Invalid reset observation")
        require(np.isfinite(o2).all(), "Nonfinite reset observation")
        require(np.allclose(o2[9:12], np.tanh(np.array(i2["physical_velocity_mps"]) / 0.25), atol=1e-5), "Velocity channels mis-signed/scaled")
        print("   PASS")

        print("4) Near-target precision feature check (1 mm physical error)...")
        center = PRACTICAL_ENVELOPE.center
        target = center + np.array([0.001, 0, 0])
        obs, info = env.reset(options={"target": target, "start": center})
        require(abs(obs[6] - 0.1) < 0.01, f"Expected fine X feature near 0.1, got {obs[6]}")
        require(abs(info["distance_mm"] - 1.0) < 0.05, "Reset position not precise enough")
        print(f"   PASS: fine_x={obs[6]:.5f}, true_error={info['distance_mm']:.4f} mm")

        print("5) Step/termination test: zero actions while at exact target...")
        obs, info = env.reset(options={"target": center, "start": center})
        reason = None
        for t in range(1, env.dwell_steps + 3):
            obs, r, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            require(env.observation_space.contains(obs), "Invalid observation")
            require(np.isfinite(r), "Nonfinite reward")
            if terminated or truncated:
                reason = info["termination_reason"]
                break
        require(reason == "settled", f"Expected settled termination, got {reason}; info={info}")
        require(info["dwell"] >= env.dwell_steps, "Successful dwell not reached")
        print(f"   PASS: settled after {t} steps at {info['distance_mm']:.6f} mm")

        print("6) 1,000 random steps, resetting on termination/truncation...")
        rng = np.random.default_rng(45451)
        obs, info = env.reset(seed=45451)
        resets = 0
        unsafe = 0
        for _ in range(1000):
            action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            require(env.observation_space.contains(obs), "Observation outside Box")
            require(np.isfinite(reward), "Reward was nonfinite")
            require(np.isfinite(info["distance_m"]), "Distance was nonfinite")
            if terminated or truncated:
                unsafe += int(info["unsafe"])
                resets += 1
                obs, info = env.reset()
        print(f"   PASS: 1000 actions; resets={resets}, unsafe_endings={unsafe}")
    finally:
        env.close()

    print("7) Short P-controller compatibility test through Path Y (NOT RL)...")
    env = PathYEnv(max_steps=240, start_strategy="uniform")
    try:
        center = PRACTICAL_ENVELOPE.center
        tests = {
            "x_plus_40mm": [0.040, 0.0, 0.0],
            "mixed_xyz": [0.030, -0.030, 0.020],
        }
        for name, delta in tests.items():
            target = center + np.array(delta, dtype=np.float64)
            obs, info = env.reset(options={"target": target, "start": center})
            for t in range(1, 241):
                # Diagnostic ONLY: world error -> simulator action has measured +XYZ signs.
                err = np.asarray(info["target"]) - np.asarray(info["pipette"])
                action = np.clip(80.0 * err, -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            require(info["settled"] and not info["unsafe"], f"Failed {name}: {info}")
            print(f"   PASS: {name} settled in {t} steps at {info['distance_mm']:.4f} mm")
    finally:
        env.close()

    print("\n=== PATH Y ENVIRONMENT PASSED ALL PRE-TRAINING TESTS ===")
    print("Note: This verifies mechanics, not SAC learning or generalization.")


def main() -> None:
    print("PATH Y / PHASE 2 — NO RL TRAINING")
    check_reward()
    check_sim_env()


if __name__ == "__main__":
    main()
