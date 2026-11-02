"""Gymnasium entry point for the final Task 11 OT-2 controller.

The authoritative environment lives in Path_Y/path_y_env.py.  Re-exporting
that class here ensures the assessor's expected filename and the trained
SAC policy use identical physics, observations, actions, and reward logic.

Observations: 12 float32 values (position, coarse/fine target error, velocity).
Actions: three continuous commands in [-1, 1].
Success: position error < 1 mm for 10 consecutive simulation steps.

No trained model, checkpoint, or underlying environment is modified here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _directory in (_PROJECT_ROOT / "src", Path(__file__).resolve().parent / "Path_Y"):
    if not _directory.is_dir():
        raise FileNotFoundError(
            f"Required Task 11 environment dependency is missing: {_directory}"
        )
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from path_y_env import PathYEnv


class OT2Env(PathYEnv):
    """Task 11 public Gymnasium wrapper; exact Path Y behavior unchanged."""


# Backward-compatible spelling from the original Reinforcement learning control entry point.
OT2ENV = OT2Env

__all__ = ["OT2Env", "OT2ENV", "PathYEnv"]
