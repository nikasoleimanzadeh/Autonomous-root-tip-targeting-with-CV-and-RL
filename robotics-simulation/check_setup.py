"""robotics simulation setup check.

Run from the project root:
    python task09_robotics_environment/check_setup.py
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ot2.sim import sim_dir


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    try:
        path = sim_dir()
    except FileNotFoundError as exc:
        print("\n❌ Simulator not found.\n")
        print(exc)
        print(
            "\nTask 9 code is ready, but the BUas simulator assets are required to "
            "execute it. The folder must contain sim_class.py, meshes/ and textures/."
        )
        raise SystemExit(1)

    print("✅ Simulator found:", path)
    print("✅ sim_class.py:", path / "sim_class.py")


if __name__ == "__main__":
    main()
