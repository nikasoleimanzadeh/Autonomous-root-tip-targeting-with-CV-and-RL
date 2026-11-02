"""Pure-data unit tests; PyBullet and stable-baselines3 needed only for live benchmark."""
from pathlib import Path
import importlib.util
import csv
import sys
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'benchmark_controllers.py'
spec = importlib.util.spec_from_file_location('benchmark_task13', MODULE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def record(controller, error=0.5, settled=True, steps=100, plate='test_image_01', plant='1'):
    return {'controller': controller, 'plate': plate, 'plant': plant,
            'final_error_mm': error, 'settled': settled, 'unsafe': False,
            'steps': steps, 'simulated_time_s': steps/240.,
            'rollout_wall_time_s': 0.2}


def test_summary_both_pass_fail():
    s = b.summarize([record('pid'), record('pid', error=3.5, settled=False, steps=600)])
    assert s['n_targets'] == 2
    assert s['settled_pct'] == 50
    assert s['final_below_1mm'] == 1
    assert s['mean_final_error_mm'] == 2.0
    assert s['max_final_error_mm'] == 3.5


def test_pairing_matches_targets():
    p = [record('pid')]
    q = [record('sac', error=.75)]
    assert b.paired_rows(p,q)[0]['sac_minus_pid_error_mm'] == .25
    with pytest.raises(ValueError, match='Mismatch'):
        b.paired_rows(p,[record('sac',plant='2')])


def test_summary_reject_empty():
    with pytest.raises(ValueError):
        b.summarize([])


def test_bad_controller_configuration_rejected():
    with pytest.raises(ValueError):
        b.evaluate_controller(None, [], name='x')


def test_reports(tmp_path):
    p = [record('pid')]; r = [record('sac', error=.7)]
    summary = {'n_targets': 1, 'plates': 1,
               'controllers': {'pid': b.summarize(p), 'sac': b.summarize(r)}}
    b.make_plots(p,r,tmp_path)
    b.write_report(tmp_path,summary,p,r)
    assert (tmp_path/'benchmark_results.md').exists()
    assert len(list(tmp_path.glob('*.png'))) == 3
