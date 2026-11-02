"""Task 12: check the declared image->OT-2 simulation convention.

These checks DO NOT calibrate a real camera or independently verify photographed
root-tip locations. They verify mathematical consistency and, optionally, the
loaded specimen geometry in the user's own PyBullet simulator.

Run from the robotics repo root (PowerShell):
    python task12_13_integration_benchmarking/verify_simulation_calibration.py `
      --with-sim

No robot movement, training, model inference, or liquid dispensing is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from calibration import load_config, map_for_plate
from cv_adapter import load_crop_sizes, load_tips


# Labels denote crop-local image quadrants, not actual fiducial measurements.
QUADRANTS = {
    "top_left": (0.25, 0.25),
    "top_right": (0.75, 0.25),
    "bottom_left": (0.25, 0.75),
    "bottom_right": (0.75, 0.75),
}


def check_four_quadrants(plate: str, width: int, height: int, config: dict) -> dict:
    """Test centre, signs, four distinct points, and inverse under the assumed map."""
    if config.get("mode") != "assumed_simulation":
        raise ValueError("Four-quadrant convention check requires assumed_simulation mode")

    mapping = map_for_plate(plate, width, height, config, allow_assumed=True)
    cx, cy = map(float, config["plate_center_xy_m"])
    size_x, size_y = map(float, config["plate_size_xy_m"])

    centre = mapping.pixel_to_world(width / 2, height / 2)
    centre_error_mm = float(np.linalg.norm(centre[:2] - [cx, cy]) * 1000)

    points = {}
    max_inverse_error_px = 0.0
    max_expected_world_error_mm = 0.0
    for label, (x_fraction, y_fraction) in QUADRANTS.items():
        x_px, y_px = x_fraction * width, y_fraction * height
        world = mapping.pixel_to_world(x_px, y_px)
        roundtrip = mapping.world_to_pixel(float(world[0]), float(world[1]))
        max_inverse_error_px = max(
            max_inverse_error_px,
            abs(roundtrip[0] - x_px), abs(roundtrip[1] - y_px),
        )
        expected = np.array([
            cx + (0.5 - y_fraction) * size_x,
            cy + (x_fraction - 0.5) * size_y,
        ])
        max_expected_world_error_mm = max(
            max_expected_world_error_mm,
            float(np.linalg.norm(world[:2] - expected) * 1000),
        )
        points[label] = {
            "pixel_xy": [float(x_px), float(y_px)],
            "world_xy_m": [float(world[0]), float(world[1])],
        }

    top_left = points["top_left"]["world_xy_m"]
    top_right = points["top_right"]["world_xy_m"]
    bottom_left = points["bottom_left"]["world_xy_m"]
    axis_col_to_positive_y = top_right[1] > top_left[1]
    axis_row_to_negative_x = bottom_left[0] < top_left[0]
    coords = [np.array(p["world_xy_m"]) for p in points.values()]
    min_pairwise_distance_mm = min(
        float(np.linalg.norm(coords[i] - coords[j]) * 1000)
        for i in range(len(coords)) for j in range(i + 1, len(coords))
    )
    passed = bool(
        centre_error_mm < 1e-6 and max_inverse_error_px < 1e-6
        and max_expected_world_error_mm < 1e-6
        and axis_col_to_positive_y and axis_row_to_negative_x
        and min_pairwise_distance_mm > 10.0
    )
    return {
        "plate": plate,
        "width_px": width,
        "height_px": height,
        "passed": passed,
        "centre_error_mm": centre_error_mm,
        "max_inverse_error_px": max_inverse_error_px,
        "max_expected_world_error_mm": max_expected_world_error_mm,
        "image_right_is_world_positive_y": axis_col_to_positive_y,
        "image_down_is_world_negative_x": axis_row_to_negative_x,
        "min_pairwise_separation_mm": min_pairwise_distance_mm,
        "quadrants": points,
        "physically_measured": False,
    }


def check_loaded_specimen(config: dict) -> dict:
    """Check actual PyBullet specimen pose and collision dimensions; no motion."""
    import pybullet as p
    from ot2.sim import make_sim

    sim = make_sim(num_agents=1, render=False)
    try:
        specimen_id = sim.specimenIds[0]
        position, quaternion = p.getBasePositionAndOrientation(specimen_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)
        collision_shapes = p.getCollisionShapeData(specimen_id, -1)
        box_shapes = [shape for shape in collision_shapes if shape[2] == p.GEOM_BOX]
        if not box_shapes:
            raise RuntimeError("No box collision shape for the simulator specimen")
        dims = np.asarray(box_shapes[0][3], dtype=float)
    finally:
        sim.close()

    cx, cy = map(float, config["plate_center_xy_m"])
    sx, sy = map(float, config["plate_size_xy_m"])
    top_z = float(position[2]) + float(dims[2]) / 2
    expected_top = float(config["plate_surface_z_m"])
    # The OT-2 simulator described in the coursework loads the plate at -pi/2.
    yaw_difference = math.atan2(math.sin(yaw + math.pi / 2), math.cos(yaw + math.pi / 2))
    checks = {
        "xy_centre": bool(np.allclose(position[:2], [cx, cy], atol=1e-4, rtol=0)),
        "plate_xy_size": bool(np.allclose(dims[:2], [sx, sy], atol=1e-4, rtol=0)),
        "surface_z": bool(abs(top_z - expected_top) < 1e-4),
        "orientation": bool(abs(roll) < 1e-3 and abs(pitch) < 1e-3 and abs(yaw_difference) < 1e-3),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "specimen_position_m": [float(v) for v in position],
        "specimen_euler_rad": [roll, pitch, yaw],
        "collision_box_dimensions_m": dims.tolist(),
        "computed_surface_z_m": top_z,
        "does_not_verify_photo_alignment": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path,
        default=HERE / "plate_config.example.json",
    )
    parser.add_argument(
        "--crop-metadata", type=Path,
        default=ROOT / "task12_cv_handoff" / "petri_crop_metadata.csv",
    )
    parser.add_argument(
        "--tips-csv", type=Path,
        default=ROOT / "task12_cv_handoff" / "task7_primary_root_lengths.csv",
    )
    parser.add_argument("--max-plates", type=int, default=3)
    parser.add_argument("--with-sim", action="store_true", help="Check loaded specimen pose in PyBullet")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "task12_calibration_checks.json")
    args = parser.parse_args()
    if args.max_plates < 1:
        parser.error("--max-plates must be >= 1")

    config = load_config(args.calibration)
    crop_sizes = load_crop_sizes(args.crop_metadata)
    by_plate, _ = load_tips(args.tips_csv)
    selected = list(by_plate)[:args.max_plates]
    checks = []
    for plate in selected:
        if plate not in crop_sizes:
            raise ValueError(f"No Petri crop metadata for {plate}")
        width, height = crop_sizes[plate]
        result = check_four_quadrants(plate, width, height, config)
        checks.append(result)
        print(f"{plate}: four-point geometry {'PASS' if result['passed'] else 'FAIL'} "
              f"(inverse <= {result['max_inverse_error_px']:.2e} px)")

    sim_result = None
    if args.with_sim:
        sim_result = check_loaded_specimen(config)
        print("Loaded PyBullet specimen:", "PASS" if sim_result["passed"] else "FAIL")
        print("  XYZ:", sim_result["specimen_position_m"])
        print("  Box dimensions:", sim_result["collision_box_dimensions_m"])

    all_passed = bool(all(r["passed"] for r in checks)
                      and (sim_result is None or sim_result["passed"]))
    summary = {
        "passed": all_passed,
        "calibration_mode": config["mode"],
        "simulation_specimen_geometry_checked": args.with_sim,
        "camera_to_plate_calibration_verified": False,
        "limitation": "Synthetic points check the defined pixel->world convention. No measured "
                      "correspondences establish camera/image alignment or root-tip accuracy.",
        "plates": checks,
        "loaded_simulator": sim_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", args.output)
    print("Overall:", "PASS" if all_passed else "FAIL")
    print("Camera/photo registration: UNVERIFIED (intentionally)")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
