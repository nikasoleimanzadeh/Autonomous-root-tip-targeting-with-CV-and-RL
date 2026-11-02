"""Path Y Phase 4: reward audit & honest simulated rollouts. NO RL training.

After installing the revised path_y_env.py in Path_Y, run from repository root:
    python task11_rl_controller/Path_Y/test_path_y_reward_v2.py

Audits 0.99-discounted incentives, actually simulates stationary/P trajectories,
tests successful dwell bonus and the unsafe branch, writes fresh CSV/JSON.
For the unsafe code-path test only, _safe is forced to return False; this is
explicitly a *mocked safety flag*, NOT a physical collision trajectory.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for item in (REPO / "src", HERE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from path_y_env import PathYEnv
from ot2.envelope import PRACTICAL_ENVELOPE

GAMMA = 0.99
MAX_STEPS = 600
REWARD_BOUND = 1.005  # exact upper bound on ABS per-step dense reward


def check(ok: bool, message: str):
    if not ok:
        raise AssertionError(message)


def discounted(terms: list[float], gamma: float = GAMMA) -> float:
    return float(sum((gamma ** k) * r for k, r in enumerate(terms)))


def check_analytical_rewards() -> list[dict]:
    print("\n1. MATHEMATICAL REWARD & DISCOUNT AUDIT")
    ds = np.array([0.0, .0001, .0005, .00099, .00101, .002, .005, .01, .05, .1, .2, .4, .6])
    rs = [PathYEnv.distance_reward(float(d)) for d in ds]
    check(all(rs[i] > rs[i + 1] for i in range(len(rs) - 1)), "Reward is not strictly improving with proximity")
    check(all(-REWARD_BOUND < r <= -0.005 for r in rs), "Dense reward out of expected range")

    max_discounted_cost = REWARD_BOUND * (1.0 - GAMMA ** MAX_STEPS) / (1.0 - GAMMA)
    env = PathYEnv()
    try:
        safety = env.unsafe_penalty
        bonus = env.settled_bonus
    finally:
        env.close()
    check(abs(safety - 120.0) < 1e-8 and abs(bonus - 5.0) < 1e-8,
          "Expected Path Y V2 defaults (unsafe=120, settled_bonus=5). Wrong version installed?")
    check(safety > REWARD_BOUND / (1.0 - GAMMA),
          "Unsafe penalty fails infinite-horizon worst-cost bound at gamma=0.99")
    print(f"Bounded dense reward: (-{REWARD_BOUND:.3f}, -0.005]")
    print(f"Worst 600-step discounted safe cost: -{max_discounted_cost:.3f}")
    print(f"Worst infinite-horizon discounted safe cost: -{REWARD_BOUND/(1-GAMMA):.3f}")
    print(f"Unsafe terminal penalty: -{safety:.1f}; settled bonus: +{bonus:.1f}")

    rows = []
    for d in [0.00101, .002, .005, .01, .02, .05, .1, .2, .4, .6]:
        r = PathYEnv.distance_reward(d)
        stationary = r * (1-GAMMA**MAX_STEPS)/(1-GAMMA)
        unsafe = r - safety  # hypothetical immediate unsafe terminal, same distance
        check(stationary > unsafe, f"Unsafe return outscored stationary at d={d}")
        rows.append(dict(distance_mm=round(d * 1000, 5), dense_step_reward=r,
                         hypothetical_stationary_return=stationary,
                         hypothetical_immediate_unsafe_return=unsafe,
                         unsafe_preferred=False))
        print(f"  {d*1000:7.2f} mm | safe stationary {stationary:9.3f} | unsafe {unsafe:9.3f} | PASS")
    print("PASS: Immediate unsafe termination cannot beat a stationary safe 600-step policy in this audit.")
    return rows


def cases():
    c = PRACTICAL_ENVELOPE.center
    low, high = PRACTICAL_ENVELOPE.low, PRACTICAL_ENVELOPE.high
    return [
        ("near_2mm", c, c + [0.002, 0, 0]),
        ("medium_50mm", c, c + [0.05, 0, 0]),
        ("long_200mm", c, c + [0.20, 0, 0]),
        ("cross_410mm", np.array([low[0]+.015, c[1], c[2]]),
         np.array([high[0]-.015, c[1], c[2]])),
        ("mixed_xyz", c + [-.10, -.10, -.04], c + [.11, .12, .04]),
        ("near_z_floor", c, np.array([c[0], c[1], low[2]+.008])),
    ]


def rollout(env: PathYEnv, start: np.ndarray, target: np.ndarray, mode: str) -> dict:
    obs, info = env.reset(options={"start": start, "target": target})
    init_mm = float(info["distance_mm"])
    rewards = []
    reason = "unknown"
    for n in range(1, MAX_STEPS + 1):
        if mode == "P":
            err = np.asarray(info["error_xyz_m"], dtype=np.float64)
            action = np.clip(80.0 * err, -1., 1.).astype(np.float32)
        else:
            action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            reason = info["termination_reason"] or "unknown"
            break
    return dict(mode=mode, start_distance_mm=init_mm,
                final_error_mm=float(info["distance_mm"]), steps=n,
                discounted_return=discounted(rewards),
                undiscounted_return=float(sum(rewards)),
                reason=reason, settled=bool(info["settled"]),
                unsafe=bool(info["unsafe"]))


def test_rollouts() -> list[dict]:
    print("\n2. ACTUAL PYBULLET ROLLOUTS: SAME STARTS/TARGETS / P vs ZERO")
    rows = []
    env = PathYEnv(max_steps=MAX_STEPS)
    try:
        for name, start, target in cases():
            pair = {}
            for mode in ("zero", "P"):
                r = rollout(env, np.asarray(start), np.asarray(target), mode)
                r["case"] = name
                rows.append(r)
                pair[mode] = r
                print(f"  {name:16} {mode:4} start={r['start_distance_mm']:7.1f} mm "
                      f"final={r['final_error_mm']:8.4f} mm steps={r['steps']:3} "
                      f"return={r['discounted_return']:9.3f} reason={r['reason']}")
            check(pair["P"]["settled"] and not pair["P"]["unsafe"],
                  f"P did not settle safely: {name}: {pair['P']}")
            check(pair["P"]["discounted_return"] > pair["zero"]["discounted_return"],
                  f"Reward prefers stationary zero action over successful P: {name}: {pair}")
            check(pair["zero"]["reason"] == "time_limit",
                  f"Stationary scenario unexpectedly ended early: {name}: {pair['zero']}")
    finally:
        env.close()
    print("PASS: on six physical start/target pairs, P reaches/settles and earns better discounted return than waiting.")
    return rows


def test_terminal_branches() -> dict:
    print("\n3. TERMINAL REWARD CODE PATHS")
    env = PathYEnv()
    center = PRACTICAL_ENVELOPE.center
    try:
        _, _ = env.reset(options={"start": center, "target": center})
        bonuses = []
        for _ in range(env.dwell_steps):
            _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            bonuses.append(info["reward_success_bonus"])
        check(term and not trunc and info["settled"] and info["termination_reason"] == "settled",
              "Success should terminate only after 10 consecutive steps")
        check(bonuses.count(env.settled_bonus) == 1 and bonuses[-1] == env.settled_bonus,
              f"Success bonus should occur exactly once, got {bonuses}")
        print(f"  Actual simulation: 10-step dwell -> settled, bonus +{env.settled_bonus:.1f} exactly once: PASS")

        # We explicitly mock only the safety predicate, not the simulator.
        # This exercises reward logic without claiming a real physical crash.
        _, _ = env.reset(options={"start": center, "target": center + [0.05, 0, 0]})
        original_safe = env._safe
        try:
            env._safe = lambda p: False
            _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        finally:
            env._safe = original_safe
        check(term and not trunc and info["unsafe"] and info["termination_reason"] == "unsafe",
              "Mocked unsafe flag should terminate")
        check(abs(info["reward_safety_cost"] - env.unsafe_penalty) < 1e-8,
              "Unsafe terminal cost absent")
        check(info["reward_success_bonus"] == 0,
              "Unsafe termination must not award successful settling bonus")
        check(abs(reward - (info["reward_dense"] - env.unsafe_penalty)) < 1e-6,
              "Unsafe reward decomposition mismatch")
        print(f"  Mocked unsafe flag ONLY (not real crash): safety cost -{env.unsafe_penalty:.1f}: PASS")
        return {"actual_settled_bonus_once": True,
                "mocked_unsafe_cost_applied_once": True,
                "unsafe_was_physically_reached": False}
    finally:
        env.close()


def main():
    out = HERE / "results" / "reward_v2" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print("PATH Y PHASE 4: BOUNDED-REWARD PREFLIGHT — NO RL TRAINING")
    print("Report directory:", out, flush=True)
    audit = check_analytical_rewards()
    simulations = test_rollouts()
    branches = test_terminal_branches()
    with (out / "discount_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    with (out / "rollouts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(simulations[0]))
        writer.writeheader()
        writer.writerows(simulations)
    summary = {
        "phase": "Path Y Phase 4 reward audit / NO RL TRAINING",
        "gamma": GAMMA, "max_steps": MAX_STEPS,
        "reward_version": "bounded dense distance v2",
        "audit_case_count": len(audit),
        "unsafe_preferred_count": sum(r["unsafe_preferred"] for r in audit),
        "physical_P_cases": sum(r["mode"] == "P" for r in simulations),
        "physical_P_settled": sum(r["mode"] == "P" and r["settled"] for r in simulations),
        "physical_P_better_return_than_zero": all(
            next(r for r in simulations if r["case"] == name and r["mode"] == "P")["discounted_return"] >
            next(r for r in simulations if r["case"] == name and r["mode"] == "zero")["discounted_return"]
            for name, *_ in cases()),
        "terminal_branch_checks": branches,
        "limitations": [
            "Reward audit uses gamma=0.99; re-run/rework if gamma changes.",
            "Proportional control is diagnostic, NOT trained reinforcement learning.",
            "Mocked unsafe path tests code only; not a real crash or collision.",
            "Six deterministic start-target cases cannot prove full-workspace success.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PHASE 4 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nAll tests passed. Save/upload summary.json and rollouts.csv.")


if __name__ == "__main__":
    main()
