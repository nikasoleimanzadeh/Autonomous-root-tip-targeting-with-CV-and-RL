"""Task 13 matched PID-vs-Path-Y-SAC evaluation on Task 12 CV targets.

This compares the ORIGINAL Task 10 tuned PI controller and the ORIGINAL user's
learned Path Y SAC model in the SAME Path Y simulator, start pose, 12-D
observation context (for SAC), time limit and ten-step sub-mm settling rule.

This is a *single-target controller* benchmark. It must NOT be confused with
Task 12's multi-waypoint transit/hover/retreat timings or actual root-tip error.
The plate-photo -> robot transformation is an unverified simulation assumption.
No training or model modification occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for folder in (REPO, REPO / 'src', REPO / 'task11_rl_controller' / 'Path_Y'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

DEFAULT_MODEL = (REPO / 'task11_rl_controller' / 'Path_Y' / 'runs'
                 / 'final_push_20261012_155636_757034' / 'best_learned_sac_model.zip')
DEFAULT_MANIFEST = HERE / 'results' / 'target_manifest.csv'
DT = 1.0 / 240.0
MAX_STEPS = 600
THRESHOLD_MM = 1.0
DWELL = 10


def read_cases(manifest: Path, start_xyz: np.ndarray, limit: int | None = None) -> list[dict]:
    """Load ONLY existing Task 12 mapped CV targets, with finite in-envelope poses."""
    from ot2.envelope import PRACTICAL_ENVELOPE

    if not manifest.is_file():
        raise FileNotFoundError(f'Cannot find Task 12 manifest: {manifest}')
    rows = []
    with manifest.open(newline='', encoding='utf-8-sig') as fp:
        for row in csv.DictReader(fp):
            goal = np.array([float(row[k]) for k in
                             ('target_x_m', 'target_y_m', 'target_z_m')], dtype=np.float64)
            if not np.all(np.isfinite(goal)) or not PRACTICAL_ENVELOPE.contains(goal):
                raise ValueError(f'Out-of-bounds CV target: {row}')
            if not PRACTICAL_ENVELOPE.contains(start_xyz):
                raise ValueError('Benchmark starting position outside practical envelope')
            rows.append({'plate': row['plate'], 'plant': str(row['plant']),
                         'target': goal, 'start': start_xyz.copy(),
                         'calibration_verified': str(row.get('calibration_verified', 'false')).lower() == 'true'})
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No valid mapped CV targets in manifest')
    if len({(r['plate'], r['plant']) for r in rows}) != len(rows):
        raise ValueError('Duplicate plate/plant target IDs')
    return rows


def summarize(rows: list[dict]) -> dict:
    """Report settled and final-error success separately, including failures."""
    if not rows:
        raise ValueError('Empty benchmark')
    errs = np.asarray([r['final_error_mm'] for r in rows], dtype=float)
    simtime = np.asarray([r['simulated_time_s'] for r in rows], dtype=float)
    walltime = np.asarray([r['rollout_wall_time_s'] for r in rows], dtype=float)
    settled = sum(bool(r['settled']) for r in rows)
    under = sum(r['final_error_mm'] < THRESHOLD_MM for r in rows)
    return {
        'n_targets': len(rows), 'settled': settled,
        'settled_pct': 100.0 * settled / len(rows),
        'final_below_1mm': under, 'final_below_1mm_pct': 100.0 * under / len(rows),
        'unsafe': sum(bool(r['unsafe']) for r in rows),
        'mean_final_error_mm': float(np.mean(errs)),
        'median_final_error_mm': float(np.median(errs)),
        'max_final_error_mm': float(np.max(errs)),
        'p95_final_error_mm': float(np.percentile(errs, 95)),
        'mean_simulated_time_s': float(np.mean(simtime)),
        'median_simulated_time_s': float(np.median(simtime)),
        'mean_rollout_wall_time_s': float(np.mean(walltime)),
        'mean_steps': float(np.mean([r['steps'] for r in rows])),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for data in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def evaluate_controller(env, cases: list[dict], *, name: str, policy=None,
                        pid=None, max_steps: int = MAX_STEPS) -> list[dict]:
    """Evaluate a locked model or original tuned PID; reset BOTH identically.

    Position for PID is the precise PyBullet position, exactly as Task 10 uses.
    SAC receives the original Path Y 12-D observation; no PID intervention.
    """
    if (pid is None) == (policy is None):
        raise ValueError('Provide exactly one of policy and pid')
    from ot2.sim import precise_pipette_position

    out = []
    for n, case in enumerate(cases, 1):
        obs, info = env.reset(options={'start': case['start'], 'target': case['target']})
        if pid is not None:
            pid.reset()
        start_dist = float(info['distance_mm'])
        min_dist = start_dist
        begin = time.perf_counter()
        reason = 'time_limit'
        for step in range(1, max_steps + 1):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                position = precise_pipette_position(env.sim)
                action = pid.step(case['target'], position, dt=DT)
            action = np.asarray(action, dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(action)):
                raise FloatingPointError(f'{name}: non-finite action on {case["plate"]}/{case["plant"]}')
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise FloatingPointError(f'{name}: invalid observation/reward')
            min_dist = min(min_dist, float(info['distance_mm']))
            if terminated or truncated:
                reason = str(info.get('termination_reason', 'terminated'))
                break
        elapsed = time.perf_counter() - begin
        out.append({
            'controller': name, 'plate': case['plate'], 'plant': case['plant'],
            'start_x_m': float(case['start'][0]), 'start_y_m': float(case['start'][1]),
            'start_z_m': float(case['start'][2]),
            'target_x_m': float(case['target'][0]),
            'target_y_m': float(case['target'][1]), 'target_z_m': float(case['target'][2]),
            'start_distance_mm': start_dist,
            'min_error_mm': min_dist,
            'final_error_mm': float(info['distance_mm']),
            'settled': bool(info['settled']),
            'final_below_1mm': bool(info['distance_mm'] < THRESHOLD_MM),
            'unsafe': bool(info['unsafe']), 'reason': reason,
            'steps': int(step), 'simulated_time_s': step * DT,
            'rollout_wall_time_s': elapsed,
            'calibration_verified': case['calibration_verified'],
        })
        print(f'{name:3s} {n:2d}/{len(cases)} | {case["plate"]} plant={case["plant"]}'
              f' | error={info["distance_mm"]:.4f} mm | settled={info["settled"]}'
              f' | steps={step}', flush=True)
    return out


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(pid_rows: list[dict], rl_rows: list[dict]) -> list[dict]:
    if len(pid_rows) != len(rl_rows):
        raise ValueError('Different number of targets')
    out = []
    for p, r in zip(pid_rows, rl_rows):
        if (p['plate'], p['plant']) != (r['plate'], r['plant']):
            raise ValueError('Mismatch in benchmark case ordering')
        out.append({
            'plate': p['plate'], 'plant': p['plant'],
            'pid_final_error_mm': p['final_error_mm'],
            'sac_final_error_mm': r['final_error_mm'],
            'pid_settled': p['settled'], 'sac_settled': r['settled'],
            'pid_steps': p['steps'], 'sac_steps': r['steps'],
            'pid_simulated_time_s': p['simulated_time_s'],
            'sac_simulated_time_s': r['simulated_time_s'],
            'sac_minus_pid_error_mm': r['final_error_mm'] - p['final_error_mm'],
        })
    return out


def make_plots(pid_rows: list[dict], sac_rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    groups = [pid_rows, sac_rows]
    names = ['Task 10 PID', 'Path Y SAC']
    positions = np.arange(2)
    summaries = [summarize(g) for g in groups]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(positions, [v['settled_pct'] for v in summaries])
    ax.set(xticks=positions, xticklabels=names, ylabel='Settled within 1 mm (%)',
           title='Matched CV targets — 10 consecutive steps inside 1 mm', ylim=(0, 105))
    for i, stat in enumerate(summaries):
        ax.text(i, stat['settled_pct'] + 1, f"{stat['settled']}/{stat['n_targets']}", ha='center')
    fig.tight_layout()
    fig.savefig(path / 'controller_success_comparison.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(pid_rows))
    ax.plot(x, [r['final_error_mm'] for r in pid_rows], marker='o', label='Task 10 PID')
    ax.plot(x, [r['final_error_mm'] for r in sac_rows], marker='x', label='Path Y SAC')
    ax.axhline(THRESHOLD_MM, linestyle='--', label='1 mm requirement')
    ax.set(xlabel='Matched CV target index', ylabel='Final 3-D positioning error (mm)',
           title='Same start and mapped targets; full errors shown')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'target_errors.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(positions, [s['mean_simulated_time_s'] for s in summaries])
    ax.set(xticks=positions, xticklabels=names, ylabel='Mean simulated time (s / target)',
           title='Single-target time, including failures/timeouts')
    fig.tight_layout()
    fig.savefig(path / 'execution_times.png', dpi=160)
    plt.close(fig)


def write_report(path: Path, summary: dict, pid_rows: list[dict], sac_rows: list[dict]) -> None:
    p = summary['controllers']['pid']; r = summary['controllers']['sac']
    text = f'''# Task 13 — Matched controller benchmark (auto-generated)

This document reports **measured results** from the locked Task 10 controller
and the user's final Task 11 Path Y SAC model, without retraining.

| Metric | Task 10 tuned PID | Path Y SAC |
|---|---:|---:|
| CV-derived targets | {p['n_targets']} | {r['n_targets']} |
| Settled within 1 mm for 10 steps | {p['settled']}/{p['n_targets']} ({p['settled_pct']:.1f}%) | {r['settled']}/{r['n_targets']} ({r['settled_pct']:.1f}%) |
| Final below 1 mm | {p['final_below_1mm']}/{p['n_targets']} | {r['final_below_1mm']}/{r['n_targets']} |
| Mean final 3-D error (mm) | {p['mean_final_error_mm']:.4f} | {r['mean_final_error_mm']:.4f} |
| Median final 3-D error (mm) | {p['median_final_error_mm']:.4f} | {r['median_final_error_mm']:.4f} |
| Worst final 3-D error (mm) | {p['max_final_error_mm']:.4f} | {r['max_final_error_mm']:.4f} |
| Mean simulated time (s/target) | {p['mean_simulated_time_s']:.3f} | {r['mean_simulated_time_s']:.3f} |
| Unsafe terminations | {p['unsafe']} | {r['unsafe']} |

## Evaluation protocol

Both controllers use the **same `PathYEnv`**, identical Task 12 CV mapped targets,
identical fixed initial pose (`PRACTICAL_ENVELOPE.center`), a 600-step budget,
a 240 Hz simulator and termination after **10 consecutive steps within 1 mm**.
SAC uses its trained 12-D observation and deterministic actions; PID uses the
original Task 10 `build_pid()` tuned gains and precise physical position. PID
integrators are reset for each trial. This is a *single-target* head-to-head
benchmark; reset between cases is identical for both. The user-trained SAC
checkpoint is untouched.

## Interpretation and limitations

- **Mapped coordinate error:** error measures the pipette versus the requested
  3-D coordinate produced by Task 12, *not* a measured biological root-tip error.
  Photo/plate registration remains assumed, not independently calibrated.
- No liquid dispensing or infection biology is simulated. Task 12 deposits are
  visual success markers, not real inoculation.
- `simulated_time_s = steps / 240`; it is not physical hardware elapsed time.
  The CSV also contains inference/simulation wall-clock rollout duration, which
  depends on hardware and is not a fair estimate of robot actuation speed.
- This single-target comparison must **not** be conflated with Task 12's
  separate approach/hover/retreat timings or three-plate GIFs.
- The SAC model contains expert-imitation-guided training; inference uses only
  SAC policy actions. No PID teacher assists at inference.

## Visualizations

![Settled success](controller_success_comparison.png)

![Matched target errors](target_errors.png)

![Execution times](execution_times.png)

## Recommendations

1. Preserve the tuned PID as the reliable baseline when sub-mm stability is
   required; use the observed matched comparison to decide if SAC is suitable.
2. Reduce unproductive long RL trajectories and improve robust precision near
   1 mm if the recorded SAC error/timeout distribution warrants it.
3. Calibrate photo-to-robot coordinates using independent measured landmarks
   before making real root-tip accuracy claims.
4. Benchmark actual dispensing accuracy with measured droplet locations before
   translating this simulated waypoint performance to laboratory deployment.

The comparison uses **{summary['plates']} distinct plates** and **{summary['n_targets']} unique CV-derived root targets**. Full per-target results are in `matched_targets.csv`.
'''
    (path / 'benchmark_results.md').write_text(text, encoding='utf-8')


def run(model_path: Path, manifest: Path, output: Path, limit: int | None):
    from stable_baselines3 import SAC
    from ot2.envelope import PRACTICAL_ENVELOPE
    from path_y_env import PathYEnv
    from task10_pid_controller.controller import build_pid

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f'Final SAC checkpoint not found: {model_path}')
    cases = read_cases(manifest, PRACTICAL_ENVELOPE.center, limit)
    if output.exists():
        raise FileExistsError(f'Output folder already exists; will not overwrite: {output}')
    model_hash = sha256(model_path)
    print(f'VALID MODEL: {model_path.name} | SHA256={model_hash}')
    model = SAC.load(str(model_path), device='cpu')
    if tuple(model.observation_space.shape) != (12,) or tuple(model.action_space.shape) != (3,):
        raise ValueError('Wrong checkpoint! Expected Path Y 12-D observation and 3-D action')
    output.mkdir(parents=True, exist_ok=False)
    env = PathYEnv(render_mode=None, max_steps=MAX_STEPS, dwell_steps=DWELL,
                   success_threshold_m=THRESHOLD_MM / 1000.0)
    try:
        pid = build_pid()
        pid_rows = evaluate_controller(env, cases, name='PID', pid=pid)
        sac_rows = evaluate_controller(env, cases, name='SAC', policy=model)
    finally:
        env.close()
    if sha256(model_path) != model_hash:
        raise RuntimeError('MODEL CHECKPOINT CHANGED — STOP')
    matched = paired_rows(pid_rows, sac_rows)
    save_rows(output / 'pid_cases.csv', pid_rows)
    save_rows(output / 'sac_cases.csv', sac_rows)
    save_rows(output / 'matched_targets.csv', matched)
    summary = {'n_targets': len(cases), 'plates': len(set(c['plate'] for c in cases)),
               'model_sha256': model_hash, 'model_path': str(model_path),
               'manifest_path': str(manifest.resolve()),
               'start_xyz_m': PRACTICAL_ENVELOPE.center.tolist(),
               'protocol': 'Single-goal, same start, 600 steps, 1mm/10 dwell, 240 Hz',
               'controllers': {'pid': summarize(pid_rows), 'sac': summarize(sac_rows)},
               'photo_registration_verified': all(c['calibration_verified'] for c in cases)}
    (output / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    make_plots(pid_rows, sac_rows, output)
    write_report(output, summary, pid_rows, sac_rows)
    print('\nFINAL COMPARISON')
    for label, stat in summary['controllers'].items():
        print(f"{label.upper():3} | settled {stat['settled']}/{stat['n_targets']}"
              f" | mean {stat['mean_final_error_mm']:.4f} mm"
              f" | max {stat['max_final_error_mm']:.4f} mm"
              f" | avg simulated {stat['mean_simulated_time_s']:.3f}s")
    print('Outputs:', output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=HERE / 'results' /
                        ('benchmark_run_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--limit', type=int, help='Only the first N CV targets (smoke test)')
    a = parser.parse_args()
    run(a.model, a.manifest, a.output, a.limit)


if __name__ == '__main__':
    main()
