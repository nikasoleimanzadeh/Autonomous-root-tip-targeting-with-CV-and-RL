"""Fit per-plate measured pixel->robot affine matrices from measured landmarks.

CSV fields:
  image,split,pixel_x,pixel_y,robot_x_m,robot_y_m,width,height,hover_z_m
split = fit or holdout; holdout rows are independently checked, not used in fitting.
At least 3 non-collinear 'fit' rows and >=1 'holdout' row per plate.
All points must use *cropped Petri image* coordinates and actual world metres.

Usage:
  python task12_13_integration_benchmarking/fit_plate_calibration.py \
    --landmarks landmarks.csv --output calibrated_plates.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import fit_measured_affine


def fit_file(path: Path, max_holdout_error_mm: float = 0.5) -> dict:
    grouped = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            grouped[Path(row['image']).stem].append(row)
    if not grouped:
        raise ValueError('No landmarks supplied')
    config = {'mode':'measured_affine','plates':{}}
    for plate, rows in grouped.items():
        widths = {int(r['width']) for r in rows}
        heights = {int(r['height']) for r in rows}
        zs = {float(r['hover_z_m']) for r in rows}
        if len(widths) != 1 or len(heights) != 1 or len(zs) != 1:
            raise ValueError(f'{plate}: image sizes and hover Z must match')
        fit = [r for r in rows if r['split'].strip().lower() == 'fit']
        holdout = [r for r in rows if r['split'].strip().lower() == 'holdout']
        if len(fit) < 3 or not holdout:
            raise ValueError(f'{plate}: require >=3 fit points and >=1 independent holdout point')
        def xy(data, px: bool):
            colx,coly = ('pixel_x','pixel_y') if px else ('robot_x_m','robot_y_m')
            values = np.array([[float(r[colx]), float(r[coly])] for r in data])
            if px and (np.any(values[:,0] < 0) or np.any(values[:,0] >= next(iter(widths))) or
                       np.any(values[:,1] < 0) or np.any(values[:,1] >= next(iter(heights)))):
                raise ValueError(f'{plate}: landmark outside image')
            return values
        matrix, fit_rmse_mm = fit_measured_affine(xy(fit,True),xy(fit,False))
        actual = xy(holdout,False)
        predicted = np.c_[xy(holdout,True), np.ones(len(holdout))] @ matrix.T
        holdout_errors = np.linalg.norm(predicted-actual,axis=1)*1000
        max_error = float(max(holdout_errors))
        config['plates'][plate] = {
            'image_width_px': next(iter(widths)), 'image_height_px': next(iter(heights)),
            'hover_z_m': next(iter(zs)), 'matrix_xy_m': matrix.tolist(),
            'fit_rmse_mm': fit_rmse_mm, 'held_out_max_error_mm': max_error,
            'held_out_errors_mm': holdout_errors.tolist(),
            'verified': bool(max_error <= max_holdout_error_mm),
            'verification_note': 'Held-out affine landmark mapping check only; additional CV tip and dispensing error remain.',
        }
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--landmarks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-holdout-error-mm', type=float, default=0.5)
    args=p.parse_args()
    config=fit_file(args.landmarks,args.max_holdout_error_mm)
    args.output.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    for plate,r in config['plates'].items():
        print(f'{plate}: fit RMSE {r["fit_rmse_mm"]:.4f}mm, '
              f'holdout max {r["held_out_max_error_mm"]:.4f}mm, '
              f'validated={r["verified"]}')


if __name__=='__main__':
    main()
