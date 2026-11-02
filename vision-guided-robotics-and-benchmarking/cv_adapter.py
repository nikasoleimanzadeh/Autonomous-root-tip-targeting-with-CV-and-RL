"""Reuse the completed root architecture extraction CSV instead of running a different root model."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class RootTip:
    plate: str
    plant: int
    x_px: float  # root architecture extraction tip_x = crop column
    y_px: float  # root architecture extraction tip_y = crop row
    status: str


def _stem(name: str) -> str:
    return Path(str(name).replace('\\', '/')).stem.replace('_petri', '')


def load_tips(csv_path: str | Path, allowed_statuses: tuple[str, ...] = ("OK",)
              ) -> tuple[dict[str, list[RootTip]], dict[str, int]]:
    """Load unique root architecture extraction tips; skip CHECK/FAILED unless explicitly allowed."""
    required = {"image", "plant", "tip_x", "tip_y", "status"}
    by_plate: dict[str, list[RootTip]] = {}
    stats = {"total": 0, "selected": 0, "excluded_status": 0}
    seen: set[tuple[str, int]] = set()
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected Task 7 columns {sorted(required)}")
        for r in reader:
            stats["total"] += 1
            status = r["status"].strip().upper()
            if status not in allowed_statuses:
                stats["excluded_status"] += 1
                continue
            plate = _stem(r["image"])
            plant = int(r["plant"])
            key = (plate, plant)
            if key in seen:
                raise ValueError(f"Duplicate target for {plate}, plant {plant}")
            seen.add(key)
            x, y = float(r["tip_x"]), float(r["tip_y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Invalid OK tip coordinates for {key}")
            by_plate.setdefault(plate, []).append(RootTip(plate, plant, x, y, status))
            stats["selected"] += 1
    if not by_plate:
        raise ValueError("No usable Task 7 root tips found")
    for tips in by_plate.values():
        tips.sort(key=lambda t: t.plant)
    return dict(sorted(by_plate.items())), stats


def load_crop_sizes(metadata_csv: str | Path) -> dict[str, tuple[int, int]]:
    """Use crop width/height from the *original* Petri crop metadata.

    root architecture extraction pixel coordinates are crop-local; never add bbox x/y here.
    Bbox x/y would be necessary only for mapping back to the raw photograph.
    """
    out: dict[str, tuple[int, int]] = {}
    with Path(metadata_csv).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = _stem(row["image"])
            size = (int(row["width"]), int(row["height"]))
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"Bad crop size for {stem}: {size}")
            if stem in out and out[stem] != size:
                raise ValueError(f"Different crop sizes for {stem}")
            out[stem] = size
    return out


def build_targets(tips: dict[str, list[RootTip]], crop_sizes: dict[str, tuple[int, int]],
                  calibration: dict, allow_assumed: bool, plate_limit: int | None = None,
                  ) -> tuple[list[dict], dict[str, object]]:
    from calibration import map_for_plate

    rows = []
    plate_summaries = {}
    chosen = list(tips.items())[:plate_limit]
    for plate, per_plate in chosen:
        if plate not in crop_sizes:
            raise KeyError(f"No Petri crop metadata for {plate}")
        w, h = crop_sizes[plate]
        plate_map = map_for_plate(plate, w, h, calibration, allow_assumed=allow_assumed)
        from ot2.envelope import PRACTICAL_ENVELOPE
        plate_summaries[plate] = {"status": plate_map.calibration_method,
                                  "calibration_verified": plate_map.verified,
                                  "held_out_max_error_mm": plate_map.validation_error_mm,
                                  "crop_width": w, "crop_height": h}
        for tip in per_plate:
            world = plate_map.pixel_to_world(tip.x_px, tip.y_px)
            if not PRACTICAL_ENVELOPE.contains(world):
                raise ValueError(f"{plate}, plant {tip.plant} maps outside Task 9 envelope: {world}")
            rows.append({"plate": plate, "plant": tip.plant,
                         "tip_x_px": tip.x_px, "tip_y_px": tip.y_px,
                         "target_x_m": float(world[0]), "target_y_m": float(world[1]),
                         "target_z_m": float(world[2]),
                         "calibration_verified": plate_map.verified,
                         "calibration_method": plate_map.calibration_method})
    return rows, plate_summaries
