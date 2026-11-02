"""Synthetic geometry tests; no OT-2 simulator binaries or hardware required."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_simulation_calibration import check_four_quadrants


def test_four_quadrants_square_plate():
    config = {
        "mode": "assumed_simulation", "plate_center_xy_m": [0.1827, 0.137],
        "plate_size_xy_m": [0.150, 0.150], "hover_z_m": 0.175,
    }
    result = check_four_quadrants("test_image_01", 2836, 2719, config)
    assert result["passed"]
    assert result["physically_measured"] is False
    np.testing.assert_allclose(result["quadrants"]["top_left"]["world_xy_m"],
                               [0.2202, 0.0995], atol=1e-12)
    np.testing.assert_allclose(result["quadrants"]["bottom_right"]["world_xy_m"],
                               [0.1452, 0.1745], atol=1e-12)


def test_four_quadrants_rectangular_crop():
    config = {
        "mode": "assumed_simulation", "plate_center_xy_m": [0.1827, 0.137],
        "plate_size_xy_m": [0.150, 0.150], "hover_z_m": 0.175,
    }
    result = check_four_quadrants("test_image_02", 2000, 4000, config)
    assert result["passed"]
    assert result["image_right_is_world_positive_y"]
    assert result["image_down_is_world_negative_x"]
    assert result["max_inverse_error_px"] < 1e-6


def test_four_quadrants_reject_opposite_axis_convention():
    config = {
        "mode": "measured_affine", "plate_center_xy_m": [0.1827, 0.137],
        "plate_size_xy_m": [0.150, 0.150], "hover_z_m": 0.175,
    }
    try:
        check_four_quadrants("test_image_01", 1000, 1000, config)
    except ValueError as exc:
        assert "assumed_simulation" in str(exc)
    else:
        raise AssertionError("Wrong calibration mode should be rejected")
