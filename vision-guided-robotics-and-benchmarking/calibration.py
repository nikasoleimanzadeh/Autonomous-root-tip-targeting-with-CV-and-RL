"""vision-guided robotics integration: convert root architecture extraction Petri-crop pixels into OT-2 world XY (metres).

The mapping is *either* a measured affine per plate or an explicitly acknowledged
simulation assumption. An affine algebra test does not validate the actual plate
pose, camera orientation, or pipette-to-agar contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlateMap:
    image: str
    width_px: int
    height_px: int
    matrix: np.ndarray  # world_x,world_y = M @ (tip_x,tip_y,1), units m
    hover_z_m: float
    verified: bool = False
    calibration_method: str = "unknown"
    validation_error_mm: float | None = None
    plate_surface_z_m: float | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("Affine matrix must be a finite 2x3 array")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("Image dimensions must be positive")
        if not np.isfinite(self.hover_z_m):
            raise ValueError("Hover Z must be finite")
        if np.linalg.matrix_rank(matrix[:, :2]) != 2:
            raise ValueError("Affine transform is not invertible")
        if self.validation_error_mm is not None and (
            not np.isfinite(self.validation_error_mm) or self.validation_error_mm < 0
        ):
            raise ValueError("Invalid held-out calibration error")
        object.__setattr__(self, "matrix", matrix)

    def pixel_to_world(self, x_px: float, y_px: float) -> np.ndarray:
        """root architecture extraction tip_x is image column; tip_y is image row, both crop-relative."""
        if not (np.isfinite(x_px) and np.isfinite(y_px)):
            raise ValueError("Pixel coordinates are not finite")
        if not (0 <= x_px < self.width_px and 0 <= y_px < self.height_px):
            raise ValueError(f"Tip ({x_px},{y_px}) outside crop {self.width_px}x{self.height_px}")
        x_m, y_m = self.matrix @ np.array([x_px, y_px, 1.0], dtype=float)
        return np.array([x_m, y_m, self.hover_z_m], dtype=float)

    def world_to_pixel(self, world_x_m: float, world_y_m: float) -> tuple[float, float]:
        px = np.linalg.solve(self.matrix[:, :2],
                             np.array([world_x_m, world_y_m]) - self.matrix[:, 2])
        return float(px[0]), float(px[1])


def assumed_simulation_map(image: str, width_px: int, height_px: int,
                           spec: dict) -> PlateMap:
    """Explicit illustrative convention ONLY; not a physically verified calibration.

    Following the comparison reference (Ali's ZIP): rows +down -> -world X;
    columns +right -> +world Y; image crop stretched to 150 mm square.
    Coordinates are world metres. Plate dimensions/pose need independent check.
    """
    cx, cy = map(float, spec["plate_center_xy_m"])
    size_x, size_y = map(float, spec["plate_size_xy_m"])
    if size_x <= 0 or size_y <= 0:
        raise ValueError("Plate dimensions must be positive")
    matrix = np.array([
        [0.0, -size_x / height_px, cx + size_x / 2],
        [size_y / width_px, 0.0, cy - size_y / 2],
    ])
    return PlateMap(image, width_px, height_px, matrix,
                    float(spec["hover_z_m"]), verified=False,
                    calibration_method="assumed_simulation",
                    plate_surface_z_m=spec.get("plate_surface_z_m"))


def calibrated_map(image: str, width_px: int, height_px: int,
                   spec: dict) -> PlateMap:
    """Load measured image->world correspondence, fitted separately per plate.

    A held-out residual should be supplied for a real accuracy claim; fitting
    residual on the same landmarks is not an independent validation.
    """
    record = spec["plates"].get(image)
    if record is None:
        raise KeyError(f"No calibrated plate entry for {image!r}")
    if (int(record["image_width_px"]) != width_px or
            int(record["image_height_px"]) != height_px):
        raise ValueError(f"Calibration image dimensions differ for {image}")
    return PlateMap(image, width_px, height_px,
                    np.asarray(record["matrix_xy_m"], dtype=float),
                    float(record["hover_z_m"]),
                    verified=bool(record.get("verified", False)),
                    calibration_method="measured_affine",
                    validation_error_mm=record.get("held_out_max_error_mm"),
                    plate_surface_z_m=record.get("plate_surface_z_m"))


def map_for_plate(image: str, width_px: int, height_px: int,
                  config: dict, allow_assumed: bool = False) -> PlateMap:
    mode = config.get("mode")
    if mode == "measured_affine":
        return calibrated_map(image, width_px, height_px, config)
    if mode == "assumed_simulation":
        if not allow_assumed:
            raise ValueError("Assumed plate geometry requires --allow-assumed-plate-map. "
                             "It is NOT independently calibrated.")
        return assumed_simulation_map(image, width_px, height_px, config)
    raise ValueError("Calibration mode must be measured_affine or assumed_simulation")


def fit_measured_affine(pixels_xy: np.ndarray, world_xy_m: np.ndarray
                        ) -> tuple[np.ndarray, float]:
    """Fit 2x3 affine; return matrix and TRAINING residual in mm.

    >=3 non-collinear points. Check held-out locations separately.
    """
    pts = np.asarray(pixels_xy, float)
    out = np.asarray(world_xy_m, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3 or out.shape != pts.shape:
        raise ValueError("Need >=3 matched pixel_xy and world_xy_m pairs")
    if not np.isfinite(pts).all() or not np.isfinite(out).all():
        raise ValueError("Nonfinite calibration points")
    A = np.column_stack([pts, np.ones(len(pts))])
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("Calibration landmarks must not be collinear")
    coefficients, _, _, _ = np.linalg.lstsq(A, out, rcond=None)
    residuals_m = np.linalg.norm(A @ coefficients - out, axis=1)
    return coefficients.T, float(np.sqrt(np.mean(residuals_m ** 2)) * 1000)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
