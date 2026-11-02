"""Required 1,000-step random-action validation of the final reinforcement learning control wrapper.

Run from the repository root:
    python -B task11_rl_controller/test_wrapper.py

Tests the actual Path Y environment used by the winning SAC policy.
No training, checkpoint loading, or model modification occurs.
"""
from __future__ import annotations

import time

import numpy as np

from ot2_gym_wrapper import OT2Env

TOTAL_STEPS = 1000
SEED = 244258


def main() -> None:
    # Defaults match the 100-case evaluation environment, including 600 steps.
    env = OT2Env(render_mode=None, start_strategy="uniform", max_steps=600)
    env.action_space.seed(SEED)
    episodes_completed = 0
    settled = 0
    unsafe = 0
    truncated_count = 0
    last_info = None
    start_time = time.monotonic()

    try:
        observation, info = env.reset(seed=SEED)
        assert env.observation_space.contains(observation), "Invalid initial observation"
        assert observation.shape == (12,), "Unexpected observation dimensions"
        assert env.action_space.shape == (3,), "Unexpected action dimensions"

        print("Final Path Y Gymnasium wrapper random-action test", flush=True)
        print("Observation space:", env.observation_space, flush=True)
        print("Action space:", env.action_space, flush=True)

        for step_number in range(1, TOTAL_STEPS + 1):
            action = env.action_space.sample()
            assert env.action_space.contains(action), "Invalid sampled action"

            observation, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(observation), (
                f"Observation violates declared space at step {step_number}"
            )
            assert np.all(np.isfinite(observation)), "Nonfinite observation"
            assert np.isfinite(reward), "Nonfinite reward"
            assert isinstance(terminated, (bool, np.bool_)), "Bad terminated flag"
            assert isinstance(truncated, (bool, np.bool_)), "Bad truncated flag"
            assert np.isfinite(info["distance_m"]), "Nonfinite target distance"
            last_info = info

            if terminated or truncated:
                episodes_completed += 1
                settled += int(bool(info["settled"]))
                unsafe += int(bool(info["unsafe"]))
                truncated_count += int(bool(truncated))
                observation, info = env.reset()
                assert env.observation_space.contains(observation)

            if step_number % 200 == 0:
                print(f"Completed {step_number}/{TOTAL_STEPS} actions", flush=True)
    finally:
        env.close()

    print("\nWRAPPER TEST: PASS")
    print(f"Random actions executed: {TOTAL_STEPS}")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Successful settled episodes: {settled}")
    print(f"Unsafe terminations: {unsafe}")
    print(f"Time-limit truncations: {truncated_count}")
    if last_info is not None:
        print(f"Last observed target error: {last_info['distance_mm']:.3f} mm")
    print(f"Elapsed: {time.monotonic()-start_time:.1f} seconds")
    print("NOTE: Random-action success is not trained-model performance.")


if __name__ == "__main__":
    main()
