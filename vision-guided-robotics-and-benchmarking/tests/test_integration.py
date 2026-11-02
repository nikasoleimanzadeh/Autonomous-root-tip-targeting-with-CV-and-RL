"""Tests requiring no OT-2 binary/assets. Run from repo root: pytest -q ..."""
from pathlib import Path
import sys
import csv
import json
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
for p in (str(HERE),str(ROOT),str(ROOT/'src')):
    if p not in sys.path: sys.path.insert(0,p)

from calibration import assumed_simulation_map, map_for_plate, fit_measured_affine
from cv_adapter import load_tips, load_crop_sizes, build_targets


def test_assumed_map_center_and_inverse():
    spec = {'plate_center_xy_m':[0.1827,0.137], 'plate_size_xy_m':[0.150,0.150], 'hover_z_m':0.175}
    m = assumed_simulation_map('plate1', 3000, 2700, spec)
    np.testing.assert_allclose(m.pixel_to_world(1500,1350), [0.1827,0.137,0.175], atol=1e-12)
    for px,py in [(0,0),(300,260),(2999,2699)]:
        world = m.pixel_to_world(px,py)
        xx,yy = m.world_to_pixel(*world[:2])
        np.testing.assert_allclose((xx,yy),(px,py),atol=1e-8)


def test_assumed_map_explicit_guard():
    spec = {'mode':'assumed_simulation','plate_center_xy_m':[0.1827,0.137], 'plate_size_xy_m':[0.150,0.150], 'hover_z_m':0.175}
    with pytest.raises(ValueError, match='allow-assumed'):
        map_for_plate('plate1', 3000, 2700, spec)


def test_fit_requires_noncollinear():
    with pytest.raises(ValueError, match='collinear'):
        fit_measured_affine(np.array([[0,0],[1,1],[2,2]]),np.array([[0,0],[1,1],[2,2]]))


def test_fit_affine_known_points():
    pixel = np.array([[0,0],[100,0],[0,100],[100,100],[50,50]])
    truth = np.array([[0.0003,-0.00005,0.025],[0.00002,0.0004,-0.019]])
    world = np.c_[pixel, np.ones(len(pixel))] @ truth.T
    M, residual = fit_measured_affine(pixel,world)
    np.testing.assert_allclose(M,truth,atol=1e-12)
    assert residual < 1e-8


def test_cv_pipeline_filters_status_and_uses_crop_pixels(tmp_path):
    tipfile = tmp_path / 'tips.csv'
    with tipfile.open('w',newline='') as f:
        w = csv.writer(f); w.writerow(['image','plant','tip_x','tip_y','status']);
        w.writerows([['test_image_01',1,1500,1350,'OK'], ['test_image_01',2,1600,1480,'CHECK'],
                     ['test_image_02',1,750,700,'OK']])
    cropfile = tmp_path / 'crop.csv'
    with cropfile.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image','x','y','width','height'])
        w.writerows([['test_image_01.png',20,34,3000,2700],['test_image_02.png',33,17,1500,1400]])
    byplate,stats = load_tips(tipfile)
    assert stats == {'total':3,'selected':2,'excluded_status':1}
    assert byplate['test_image_01'][0].x_px==1500
    sizes = load_crop_sizes(cropfile)
    spec={'mode':'assumed_simulation','plate_center_xy_m':[0.1827,0.137],
          'plate_size_xy_m':[0.150,0.150],'hover_z_m':0.175}
    rows,details=build_targets(byplate,sizes,spec,allow_assumed=True)
    assert len(rows)==2 and len(details)==2
    np.testing.assert_allclose([rows[0]['target_x_m'],rows[0]['target_y_m']], [0.1827,0.137], atol=1e-12)


def test_fit_plate_calibration_heldout(tmp_path):
    from fit_plate_calibration import fit_file
    f = tmp_path/'landmarks.csv'
    matrix=np.array([[0.00004,-0.00001,0.055],[0.00002,0.00003,0.061]])
    pts=[(1,5),(500,12),(20,550),(530,570),(240,230)]
    with f.open('w',newline='') as h:
        w=csv.writer(h)
        w.writerow(['image','split','pixel_x','pixel_y','robot_x_m','robot_y_m','width','height','hover_z_m'])
        for i,(x,y) in enumerate(pts):
            out=matrix@np.array([x,y,1]); w.writerow(['plate1','fit' if i<4 else 'holdout',x,y,*out,600,600,0.175])
    out=fit_file(f)
    assert out['plates']['plate1']['verified']
    np.testing.assert_allclose(out['plates']['plate1']['matrix_xy_m'],matrix,atol=1e-12)
