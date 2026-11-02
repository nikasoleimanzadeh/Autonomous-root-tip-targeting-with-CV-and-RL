"""Path Y: independent, precision-oriented Gymnasium OT-2 environment.

reinforcement learning control requires the policy to learn continuous 3-axis pipette positioning.
This environment deliberately does NOT import Path B, Experiment Z, HER or PID.
It uses only the validated low-level OT-2 simulator/teleport/envelope utilities.

Observation (12 float32 values):
    [0:3]   world position normalized by practical envelope to around [-1, 1]
    [3:6]   target - position in units of envelope axis span (coarse error)
    [6:9]   target - position in units of 10 mm, clipped to [-1, 1] (fine error)
    [9:12]  tanh(physical pipette velocity in m/s / 0.25 m/s)

Actions: 3 continuous velocity commands [-1, 1]. The 4th simulator drop
command is always 0. Physical action sign was independently measured positive
on all three axes by path_y_phase1_diagnostics.py. No hidden P/PID controller.

Reward (V2): -(0.75*d/(d+0.05) + 0.25*d/(d+0.002) + 0.005),
    where d is Euclidean target distance in METRES. This is bounded strictly
    between -1.005 and -0.005, monotone, and sensitive to fine positioning.
    Settling awards +5 ONCE (and ends the episode). Unsafe termination costs
    -120 ONCE (and ends the episode). With gamma=0.99, the maximum magnitude
    of an infinite stationary cost is 1.005/(1-0.99)=100.5 < 120, making
    deliberate immediate failure worse than staying safe for all distances.
    IMPORTANT: re-audit if gamma, reward scale, or penalties change.

Success: Euclidean distance < 0.001 m for 10 consecutive simulation steps.
Terminated on settled success or unsafe position; truncated at max_steps.
The 10-step dwell is a deliberate stability standard beyond the scoring cutoff.

The practical envelope excludes a measured lower-z chassis-interference region.
Any claim of whole-workspace coverage must refer to the documented collision-
free practical envelope, not the raw joint-limit AABB.

These are PRE-TRAINING candidate choices. No hyperparameter or accuracy claim
is implied by this file.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ot2.envelope import Envelope, PRACTICAL_ENVELOPE, sample_target
from ot2.sim import make_sim, precise_joint_velocity, precise_pipette_position
from ot2.teleport import teleport_pipette


class PathYEnv(gym.Env):
    """One robot, one target, physical coordinates in metres (Gymnasium API)."""

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 240}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        max_steps: int = 600,
        success_threshold_m: float = 0.001,
        dwell_steps: int = 10,
        envelope: Envelope = PRACTICAL_ENVELOPE,
        target_margin_m: float = 0.005,
        start_margin_m: float = 0.010,
        boundary_tolerance_m: float = 0.010,
        start_strategy: str = "uniform",  # or 'near_target' for curriculum diagnostics
        near_min_distance_m: float = 0.020,
        near_max_distance_m: float = 0.060,
        action_scale: float = 1.0,
        velocity_scale_mps: float = 0.25,
        fine_error_scale_m: float = 0.010,
        unsafe_penalty: float = 120.0,
        settled_bonus: float = 5.0,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "none", "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        if max_steps < 1 or dwell_steps < 1:
            raise ValueError("max_steps and dwell_steps must be positive")
        if not (0.0 < success_threshold_m < 0.01):
            raise ValueError("success_threshold_m must be in (0, 0.01) metres")
        if start_strategy not in ("uniform", "near_target"):
            raise ValueError("start_strategy must be 'uniform' or 'near_target'")
        if not (0.0 < near_min_distance_m < near_max_distance_m):
            raise ValueError("near target distance limits must be positive and increasing")
        if not (0.0 < action_scale <= 1.0):
            raise ValueError("action_scale must be in (0, 1]")
        if velocity_scale_mps <= 0 or fine_error_scale_m <= 0:
            raise ValueError("observation scales must be positive")
        if min(target_margin_m, start_margin_m, boundary_tolerance_m) < 0:
            raise ValueError("margins/tolerances cannot be negative")
        if np.any(envelope.extent <= 2 * max(target_margin_m, start_margin_m)):
            raise ValueError("Margins leave no usable range on one or more axes")
        if unsafe_penalty < 0 or settled_bonus < 0:
            raise ValueError("unsafe_penalty and settled_bonus must be nonnegative")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.success_threshold_m = float(success_threshold_m)
        self.dwell_steps = int(dwell_steps)
        self.envelope = envelope
        self.target_margin_m = float(target_margin_m)
        self.start_margin_m = float(start_margin_m)
        self.boundary_tolerance_m = float(boundary_tolerance_m)
        self.start_strategy = start_strategy
        self.near_min_distance_m = float(near_min_distance_m)
        self.near_max_distance_m = float(near_max_distance_m)
        self.action_scale = float(action_scale)
        self.velocity_scale_mps = float(velocity_scale_mps)
        self.fine_error_scale_m = float(fine_error_scale_m)
        self.unsafe_penalty = float(unsafe_penalty)
        self.settled_bonus = float(settled_bonus)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # A terminal observation can lie just beyond the sampled envelope.
        # Do not clip true physical position/error to satisfy arbitrary bounds.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.sim = make_sim(
            num_agents=1,
            render=(render_mode == "human"),
            rgb_array=(render_mode == "rgb_array"),
        )
        self._target = envelope.center.copy()
        self._start = envelope.center.copy()
        self._steps = 0
        self._dwell = 0
        self._done = False

    @staticmethod
    def distance_reward(distance_m: float) -> float:
        """Bounded monotone dense signal with precision-sensitive local slope.

        World distance is METRES, not mm. Returns (-1.005, -0.005].
        The 50 mm scale helps long-range navigation. The 2 mm scale provides
        a meaningful distinction between 2 mm, 1 mm and 0.5 mm errors.
        """
        d = float(distance_m)
        if not np.isfinite(d) or d < 0:
            raise ValueError("distance_m must be finite and nonnegative")
        return -float(0.75 * d / (d + 0.05) + 0.25 * d / (d + 0.002) + 0.005)

    def _pipette(self) -> np.ndarray:
        return np.asarray(precise_pipette_position(self.sim), dtype=np.float64)

    def _physical_velocity(self) -> np.ndarray:
        # The forward pose model is p=[base_x-j0, base_y-j1, base_z+j2]+offset.
        # Therefore joint velocities on X and Y have the OPPOSITE world sign.
        v_joint = np.asarray(precise_joint_velocity(self.sim), dtype=np.float64)
        return v_joint * np.array([-1.0, -1.0, +1.0], dtype=np.float64)

    def _observe(self) -> np.ndarray:
        p = self._pipette()
        error = self._target - p
        pos = 2.0 * (p - self.envelope.low) / self.envelope.extent - 1.0
        coarse = error / self.envelope.extent
        fine = np.clip(error / self.fine_error_scale_m, -1.0, 1.0)
        velocity = np.tanh(self._physical_velocity() / self.velocity_scale_mps)
        return np.concatenate((pos, coarse, fine, velocity)).astype(np.float32)

    def _safe(self, p: np.ndarray) -> bool:
        return self.envelope.contains(p, margin=-self.boundary_tolerance_m)

    def _info(self, p: np.ndarray | None = None) -> dict[str, Any]:
        if p is None:
            p = self._pipette()
        error = self._target - p
        distance = float(np.linalg.norm(error))
        return {
            "steps": int(self._steps),
            "distance_m": distance,
            "distance_mm": 1000.0 * distance,
            "dwell": int(self._dwell),
            "is_success": bool(distance < self.success_threshold_m),
            "settled": bool(self._dwell >= self.dwell_steps),
            "unsafe": bool(not self._safe(p)),
            "pipette": p.tolist(),
            "target": self._target.tolist(),
            "start": self._start.tolist(),
            "error_xyz_m": error.tolist(),
            "physical_velocity_mps": self._physical_velocity().tolist(),
        }

    def _parse_pose(self, name: str, pose: Any) -> np.ndarray:
        p = np.asarray(pose, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"{name} must contain three finite world coordinates")
        if not self.envelope.contains(p):
            raise ValueError(f"{name} {p.tolist()} outside practical envelope")
        return p

    def _sample_near_target(self) -> np.ndarray:
        # Uniform direction on 3D sphere, bounded radius. Rejection near walls.
        for _ in range(256):
            v = self.np_random.normal(size=3)
            mag = np.linalg.norm(v)
            if mag < 1e-12:
                continue
            radius = self.np_random.uniform(
                self.near_min_distance_m, self.near_max_distance_m
            )
            start = self._target + radius * (v / mag)
            if self.envelope.contains(start, margin=self.start_margin_m):
                return start
        raise RuntimeError("Unable to sample near-target start; check margins/ranges")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else options
        self._target = (
            self._parse_pose("target", options["target"])
            if "target" in options
            else sample_target(
                self.np_random,
                envelope=self.envelope,
                margin=self.target_margin_m,
            )
        )

        if "start" in options:
            self._start = self._parse_pose("start", options["start"])
        elif self.start_strategy == "near_target":
            self._start = self._sample_near_target()
        else:
            self._start = sample_target(
                self.np_random,
                envelope=self.envelope,
                margin=self.start_margin_m,
            )

        self.sim.reset(num_agents=1)
        self.sim.run([[0.0, 0.0, 0.0, 0]], num_steps=20)
        teleport_pipette(self.sim, self._start)
        self.sim.run([[0.0, 0.0, 0.0, 0]], num_steps=1)

        self._steps = 0
        self._dwell = 0
        self._done = False
        return self._observe(), self._info()

    def step(self, action: np.ndarray):
        if self._done:
            raise RuntimeError("Episode ended. Call reset() before step().")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            raise ValueError(f"Expected finite 3D action, received {action!r}")
        a = np.clip(a, -1.0, +1.0) * self.action_scale
        self.sim.run([[float(a[0]), float(a[1]), float(a[2]), 0]], num_steps=1)
        self._steps += 1

        p = self._pipette()
        d = float(np.linalg.norm(self._target - p))
        self._dwell = self._dwell + 1 if d < self.success_threshold_m else 0
        unsafe = not self._safe(p)
        settled = self._dwell >= self.dwell_steps
        terminated = bool(unsafe or settled)
        truncated = bool(self._steps >= self.max_steps and not terminated)
        dense_reward = self.distance_reward(d)
        # Terminal terms are issued exactly once, because the episode ends.
        # Unsafe has precedence over success for safety.
        success_bonus = self.settled_bonus if settled and not unsafe else 0.0
        safety_cost = self.unsafe_penalty if unsafe else 0.0
        reward = dense_reward + success_bonus - safety_cost

        info = self._info(p)
        info["reward_dense"] = float(dense_reward)
        info["reward_success_bonus"] = float(success_bonus)
        info["reward_safety_cost"] = float(safety_cost)
        info["termination_reason"] = (
            "unsafe" if unsafe else "settled" if settled else "time_limit" if truncated else None
        )
        self._done = terminated or truncated
        return self._observe(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return getattr(self.sim, "current_frame", None)
        return None

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass
