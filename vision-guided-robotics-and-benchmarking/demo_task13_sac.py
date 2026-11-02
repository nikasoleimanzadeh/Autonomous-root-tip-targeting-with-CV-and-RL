"""Task 13 SAC multi-waypoint simulation GIFs — never trains a controller.

Actual learned Path Y SAC moves between transit, hover and retreat goals.
A simulated deposit is displayed ONLY if the RL pipette settles inside 1 mm
for ten steps on both transit and hover. No real dispensing is modelled.

The observation formula exactly reproduces PathYEnv._observe; each new goal
updates only the requested coordinate, NOT the actual robot position. There is
no teleport, teacher action or PID assistance between waypoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for folder in (REPO, REPO / 'src', REPO / 'task11_rl_controller' / 'Path_Y', HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from benchmark_task13 import DEFAULT_MODEL, DEFAULT_MANIFEST, DT, MAX_STEPS, DWELL
from calibration import load_config, map_for_plate
from cv_adapter import load_crop_sizes
from demo_gifs import PlateGif, preview_image
from ot2.envelope import PRACTICAL_ENVELOPE
from ot2.sim import make_sim, precise_joint_velocity, precise_pipette_position
from ot2.teleport import teleport_safe

TRANSIT_Z = 0.245


def sac_observation(sim, goal, fine_scale=0.010, vel_scale=.25):
    """The 12-dimensional inputs of the *trained* Path Y SAC policy."""
    p = np.asarray(precise_pipette_position(sim), dtype=float)
    error = np.asarray(goal, dtype=float) - p
    position = 2.0 * (p - PRACTICAL_ENVELOPE.low) / PRACTICAL_ENVELOPE.extent - 1.0
    coarse = error / PRACTICAL_ENVELOPE.extent
    fine = np.clip(error / fine_scale, -1, 1)
    physical_velocity = np.asarray(precise_joint_velocity(sim), dtype=float) * [-1., -1., 1.]
    vel = np.tanh(physical_velocity / vel_scale)
    return np.concatenate((position, coarse, fine, vel)).astype(np.float32)


class SACPlateGif(PlateGif):
    """Reuse Task 12 measured-position visualization, but label it correctly."""
    def _frame(self, world_xyz, target_xyz, steps, error_mm, event='MOVING'):
        frame = super()._frame(world_xyz, target_xyz, steps, error_mm, event)
        xoff = 600 if self.show_3d else 0
        d = ImageDraw.Draw(frame)
        d.rectangle((xoff, 0, frame.width, 21), fill=(16, 24, 38))
        d.text((xoff + 8, 5), f'{self.plate} | Path Y SAC / Task 13 SIMULATION | {event}',
               font=self.font, fill=(230, 242, 255))
        return frame


def move_sac(sim, policy, goal, gif=None, max_steps=MAX_STEPS):
    """Reach waypoint using pure SAC inference; keep physics continuous."""
    goal = np.asarray(goal, dtype=float)
    if not PRACTICAL_ENVELOPE.contains(goal):
        raise ValueError(f'Unsafe waypoint outside practical work envelope: {goal}')
    dwell = 0
    best = float('inf')
    last_distance_mm = float('inf')
    reason = 'time_limit'
    for step in range(1, max_steps + 1):
        obs = sac_observation(sim, goal)
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(3)
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite SAC action')
        sim.run([[float(v) for v in np.clip(action, -1, 1)] + [0.0]], num_steps=1)
        reached = np.asarray(precise_pipette_position(sim), dtype=float)
        last_distance_mm = float(np.linalg.norm(reached - goal) * 1000.0)
        best = min(best, last_distance_mm)
        if gif is not None:
            gif.on_step(reached, goal, step, last_distance_mm)
        dwell = dwell + 1 if last_distance_mm < 1.0 else 0
        if not PRACTICAL_ENVELOPE.contains(reached, margin=-.010):
            reason = 'unsafe'
            break
        if dwell >= DWELL:
            reason = 'settled'
            break
    return {'settled': reason == 'settled', 'reason': reason, 'steps': step,
            'simulated_time_s': step * DT, 'final_error_mm': last_distance_mm,
            'min_error_mm': best, 'reached_xyz_m': reached.tolist()}


def group_manifest(path: Path, max_plates: int):
    groups = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            row = dict(row)
            for name in ('target_x_m', 'target_y_m', 'target_z_m', 'tip_x_px', 'tip_y_px'):
                row[name] = float(row[name])
            groups.setdefault(row['plate'], []).append(row)
    return dict(list(groups.items())[:max_plates])


def main():
    from stable_baselines3 import SAC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--metadata', type=Path, default=REPO / 'task12_cv_handoff' / 'petri_crop_metadata.csv')
    p.add_argument('--calibration', type=Path, default=HERE / 'plate_config.example.json')
    p.add_argument('--previews', type=Path, default=REPO / 'task12_cv_handoff' / 'previews')
    p.add_argument('--gif-dir', type=Path, default=HERE / 'gifs')
    p.add_argument('--output', type=Path, default=HERE / 'results' / 'task13_sac_route_results.csv')
    p.add_argument('--max-plates', type=int, default=3)
    p.add_argument('--gif-3d', action='store_true')
    p.add_argument('--allow-assumed-plate-map', action='store_true',
                   help='Acknowledge photo registration has not been measured')
    a = p.parse_args()
    if a.max_plates < 1:
        p.error('--max-plates must be positive')
    cfg = load_config(a.calibration)
    if cfg.get('mode') == 'assumed_simulation' and not a.allow_assumed_plate_map:
        p.error('Assumed mapping requires --allow-assumed-plate-map')
    if not a.model.is_file():
        p.error(f'Checkpoint not found: {a.model}')
    groups = group_manifest(a.manifest, a.max_plates)
    sizes = load_crop_sizes(a.metadata)
    model = SAC.load(str(a.model), device='cpu')
    if tuple(model.observation_space.shape) != (12,) or tuple(model.action_space.shape) != (3,):
        raise ValueError('Not a Path Y SAC model (needs 12D obs, 3D actions)')
    records = []
    for plate, rows in groups.items():
        w, h = sizes[plate]
        mapping = map_for_plate(plate, w, h, cfg, allow_assumed=a.allow_assumed_plate_map)
        image = preview_image(a.previews, plate)
        if image is None:
            raise FileNotFoundError(f'No plate preview for {plate} in {a.previews}')
        gif = SACPlateGif(plate, rows, mapping, image,
                          a.gif_dir / f'{plate}_sac.gif',
                          every=45, show_3d=a.gif_3d)
        sim = make_sim(num_agents=1, render=False, rgb_array=False)
        try:
            # Match Task 12 initialization, then keep continuous joint state
            # for all waypoints within this plate.
            sim.run([[0., 0., 0., 0.]], num_steps=20)
            teleport_safe(sim, PRACTICAL_ENVELOPE.center)
            sim.run([[0., 0., 0., 0.]], num_steps=1)
            for row in rows:
                goal = np.array([row['target_x_m'], row['target_y_m'], row['target_z_m']])
                transit = np.array([goal[0], goal[1], TRANSIT_Z])
                approach = move_sac(sim, model, transit, gif)
                hover = (move_sac(sim, model, goal, gif) if approach['settled']
                         else {'settled': False, 'reason': 'approach_failed', 'steps': 0,
                               'simulated_time_s': 0, 'final_error_mm': float('nan')})
                mark = bool(approach['settled'] and hover['settled'])
                if mark:
                    gif.mark_deposit(goal, row)
                # Attempt safe retreat even after a timeout, unless unsafe.
                retreat = (move_sac(sim, model, transit, gif)
                           if approach['reason'] != 'unsafe' and hover['reason'] != 'unsafe'
                           else {'settled': False, 'reason': 'unsafe_abort', 'steps': 0,
                                 'simulated_time_s': 0})
                records.append({'plate': plate, 'plant': row['plant'],
                                'simulated_deposit': mark,
                                'approach_settled': approach['settled'],
                                'hover_settled': hover['settled'],
                                'retreat_settled': retreat['settled'],
                                'position_error_mm': hover['final_error_mm'],
                                'approach_reason': approach['reason'],
                                'hover_reason': hover['reason'],
                                'retreat_reason': retreat['reason'],
                                'simulated_motion_time_s': sum(
                                    t['simulated_time_s'] for t in (approach, hover, retreat))})
                print(f'{plate} plant {row["plant"]}: SAC mark={mark} '
                      f'hover={hover["final_error_mm"]:.3f} mm '
                      f'({hover["reason"]})', flush=True)
                if retreat['reason'] == 'unsafe' or retreat['reason'] == 'unsafe_abort':
                    print('Unsafe state: stopping this plate')
                    break
        finally:
            sim.close()
        gif.save()
        print('Saved GIF:', gif.gif_path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8') as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print('REAL SAC ROUTE: simulated deposits', sum(r['simulated_deposit'] for r in records),
          '/', len(records), '\nCSV:', a.output)
    print('Calibration: photo-to-world remains assumed; no liquid dispensing')


if __name__ == '__main__':
    main()
