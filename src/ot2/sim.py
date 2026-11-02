"""Utilities for loading and interacting with the BUas OT-2 PyBullet simulator.

The simulator assets are not bundled with this project. Set ``OT2_SIM_DIR`` to
its folder when it is stored elsewhere, or place ``Y2B-2023-OT2_Twin`` next to
``src/`` in the project root.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator


def _candidate_sim_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("OT2_SIM_DIR")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    project_root = Path(__file__).resolve().parent.parent.parent
    candidates.extend(
        [
            (project_root / "Y2B-2023-OT2_Twin").resolve(),
            (project_root / "OT2_Twin").resolve(),
            (project_root / "simulation").resolve(),
        ]
    )

    # Keep order but remove duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def sim_dir() -> Path:
    """Return the simulator folder containing ``sim_class.py``.

    The function first checks the ``OT2_SIM_DIR`` environment variable and then
    a few common project-local folders. It deliberately does not crawl the
    whole filesystem.
    """
    checked: list[str] = []
    for candidate in _candidate_sim_dirs():
        checked.append(str(candidate))
        if candidate.is_dir() and (candidate / "sim_class.py").exists():
            return candidate

    raise FileNotFoundError(
        "OT-2 simulator assets were not found. Expected a folder containing "
        "sim_class.py. Set OT2_SIM_DIR=/path/to/Y2B-2023-OT2_Twin or place "
        "that folder in the project root. Checked:\n  - " + "\n  - ".join(checked)
    )


@contextlib.contextmanager
def with_sim_cwd() -> Iterator[Path]:
    """Temporarily switch to the simulator folder for relative asset loading."""
    prev = os.getcwd()
    path = sim_dir()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def _ensure_sim_importable() -> Path:
    path = sim_dir()
    sim_path = str(path)
    if sim_path not in sys.path:
        sys.path.insert(0, sim_path)
    return path


def make_sim(num_agents: int = 1, render: bool = False, rgb_array: bool = False):
    """Construct the course ``sim_class.Simulation`` from any task folder."""
    path = _ensure_sim_importable()
    os.chdir(path)  # upstream simulator resolves textures/meshes relative to cwd
    from sim_class import Simulation  # type: ignore[import-not-found]

    return Simulation(num_agents=num_agents, render=render, rgb_array=rgb_array)


def precise_pipette_position(sim, robot_index: int = 0):
    """Return pipette world position as float64 metres from PyBullet joints."""
    import numpy as np
    import pybullet as p

    robot_id = sim.robotIds[robot_index]
    base = p.getBasePositionAndOrientation(robot_id)[0]
    js = p.getJointStates(robot_id, [0, 1, 2])
    off = sim.pipette_offset
    return np.array(
        [
            base[0] - js[0][0] + off[0],
            base[1] - js[1][0] + off[1],
            base[2] + js[2][0] + off[2],
        ],
        dtype=np.float64,
    )


def precise_joint_velocity(sim, robot_index: int = 0):
    """Return joint velocities as float64 values."""
    import numpy as np
    import pybullet as p

    robot_id = sim.robotIds[robot_index]
    js = p.getJointStates(robot_id, [0, 1, 2])
    return np.array([js[i][1] for i in range(3)], dtype=np.float64)
