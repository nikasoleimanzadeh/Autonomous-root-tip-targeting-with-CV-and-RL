"""PID control PID controller builder.

The reusable PID implementation lives in ``src/ot2/control/pid.py`` so Tasks 11–13
can use the exact same controller. This module only loads the gains selected by
PID control and constructs the three-axis controller.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.control.pid import AxisPID, XYZPID

DEFAULT_GAINS_PATH = Path(__file__).resolve().parent / "results" / "gains_final.json"


def build_pid(gains_path: Path | str = DEFAULT_GAINS_PATH) -> XYZPID:
    """Build XYZ PID from the gains selected by ``tune_pid.py``."""
    path = Path(gains_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Final gains not found: {path}. Run task10_pid_controller/tune_pid.py first."
        )
    gains = json.loads(path.read_text())
    return XYZPID(
        x=AxisPID(**gains["x"]),
        y=AxisPID(**gains["y"]),
        z=AxisPID(**gains["z"]),
    )
