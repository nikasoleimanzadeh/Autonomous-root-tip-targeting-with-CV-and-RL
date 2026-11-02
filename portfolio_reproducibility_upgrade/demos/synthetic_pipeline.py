"""Public-safe, standard-library demonstration of a vision-to-robot coordinate handoff.

This is not the trained U-Net, the OT-2 physics engine, or a validated controller.
All input geometry is generated synthetically and motion targets are illustrative.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

PLATE_PIXELS = 600
PLATE_SIZE_METRES = 0.150
PLATE_ORIGIN_XY_METRES = (0.1077, 0.062)
HOVER_Z_METRES = 0.18


def synthetic_roots() -> list[list[tuple[float, float]]]:
    """Return deterministic synthetic root polylines in plate-local pixels."""
    roots = []
    for i in range(5):
        x0 = 60 + i * 118
        roots.append([(x0 + 6 * math.sin((y - 70) / 62 + i * .55), y)
                      for y in range(70, 440 + (i % 3) * 30, 12)])
    return roots


def detect_tips(roots: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Demo only: root endpoint localization from known synthetic polylines.

    A real CV pipeline must first infer the roots from pixels. No such inference
    is claimed or implemented here.
    """
    if not roots or any(not root for root in roots):
        raise ValueError('Every synthetic plant must contain at least one point')
    return [root[-1] for root in roots]


def pixel_to_robot(x_px: float, y_px: float) -> tuple[float, float, float]:
    """Illustrative mapping using the documented plate corner and axis convention.

    Image row -> robot X; image column -> robot Y. Assumes aligned 150-mm plate;
    this is NOT a measured or validated camera-to-robot calibration.
    """
    if not (math.isfinite(x_px) and math.isfinite(y_px)):
        raise ValueError('Coordinates must be finite')
    if not (0 <= x_px < PLATE_PIXELS and 0 <= y_px < PLATE_PIXELS):
        raise ValueError('Pixel outside synthetic plate')
    scale = PLATE_SIZE_METRES / PLATE_PIXELS
    return (PLATE_ORIGIN_XY_METRES[0] + (y_px + .5) * scale,
            PLATE_ORIGIN_XY_METRES[1] + (x_px + .5) * scale,
            HOVER_Z_METRES)


def make_svg(roots: list[list[tuple[float, float]]], tips: list[tuple[float, float]]) -> str:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 690" role="img" aria-label="Synthetic root shapes and target locations">',
        '<rect width="800" height="690" fill="#f7f8fa"/>',
        '<text x="60" y="43" font-size="23" font-family="Arial" fill="#17243b">Synthetic root-tip targeting demonstration</text>',
        '<text x="60" y="67" font-size="14" font-family="Arial" fill="#536279">Generated geometry — no client images or trained inference</text>',
        '<rect x="80" y="82" width="600" height="600" rx="9" fill="white" stroke="#9daabb" stroke-width="2"/>',
    ]
    for i, (root, tip) in enumerate(zip(roots, tips), 1):
        points = ' '.join(f'{x+80:.1f},{y+82:.1f}' for x, y in root)
        lines.append(f'<polyline points="{escape(points)}" fill="none" stroke="#287759" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
        lines.append(f'<circle cx="{tip[0]+80:.1f}" cy="{tip[1]+82:.1f}" r="8" fill="#d14040" stroke="white" stroke-width="2"/>')
        lines.append(f'<text x="{tip[0]+92:.1f}" y="{tip[1]+79:.1f}" font-family="Arial" font-size="14" fill="#17243b">{i}</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    roots = synthetic_roots()
    tips = detect_tips(roots)
    rows = []
    for i, (x, y) in enumerate(tips, 1):
        rx, ry, rz = pixel_to_robot(x, y)
        rows.append(dict(plant=i, pixel_x=round(x, 3), pixel_y=round(y, 3),
                         robot_x_m=round(rx, 6), robot_y_m=round(ry, 6), robot_z_m=rz))
    with (output / 'targets.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'synthetic_roots.svg').write_text(make_svg(roots, tips), encoding='utf-8')
    report = {
        'demonstration': 'synthetic coordinate handoff only',
        'plant_count': len(rows),
        'calibration_verified': False,
        'real_cv_inference': False,
        'robot_simulator_executed': False,
        'robot_motion_executed': False,
        'units': 'pixel coordinates and robot-space metres',
        'target_positions': rows,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    result = run(args.output)
    print(f"Generated {result['plant_count']} synthetic target coordinates in {args.output.resolve()}")
    print('Demo only: no trained inference, physical calibration or robot movement.')


if __name__ == '__main__':
    main()
