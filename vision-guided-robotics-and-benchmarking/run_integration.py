"""CLI: Task 7 CSV -> explicit plate calibration -> Task 10 PID -> OT-2 GIFs.

From the *repository root*:
  python task12_13_integration_benchmarking/run_integration.py \
    --tips-csv path/to/task7_primary_root_lengths.csv \
    --crop-metadata path/to/petri_crop_metadata.csv \
    --calibration task12_13_integration_benchmarking/plate_config.example.json \
    --allow-assumed-plate-map --max-plates 3

Add --execute --previews DIR to run the actual OT-2 simulator and generate GIFs.
Requires your existing simulator assets and $OT2_SIM_DIR; no new model or PID.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for p in (str(ROOT), str(SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from calibration import load_config, map_for_plate
from cv_adapter import load_tips, load_crop_sizes, build_targets


def save_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tips-csv', type=Path, required=True)
    p.add_argument('--crop-metadata', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--allow-assumed-plate-map', action='store_true',
                   help='Explicitly acknowledge that assumed geometry is NOT measured')
    p.add_argument('--max-plates', type=int, default=3)
    p.add_argument('--execute', action='store_true', help='Run existing Task 9/10 sim & PID')
    p.add_argument('--gif-3d', action='store_true', help='Add real PyBullet camera view to each plate GIF')
    p.add_argument('--previews', type=Path,
                   help='Path to Task 5 _petri.png/jpeg images for animation')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    p.add_argument('--gif-dir', type=Path, default=Path(__file__).resolve().parent/'gifs')
    args = p.parse_args()
    args.tips_csv = args.tips_csv.resolve()
    args.crop_metadata = args.crop_metadata.resolve()
    args.calibration = args.calibration.resolve()
    args.output = args.output.resolve()
    args.gif_dir = args.gif_dir.resolve()
    if args.previews:
        args.previews = args.previews.resolve()
    if args.max_plates <= 0:
        p.error('--max-plates must be a positive integer')

    tips, stats = load_tips(args.tips_csv)
    sizes = load_crop_sizes(args.crop_metadata)
    cfg = load_config(args.calibration)
    targets, calib_info = build_targets(tips, sizes, cfg,
                                       allow_assumed=args.allow_assumed_plate_map,
                                       plate_limit=args.max_plates)
    if not targets:
        raise RuntimeError('No valid targets selected')
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output/'task12_target_manifest.csv', targets)
    manifest = {'source': 'User Task 7 output (not Ali model)',
                'tip_status_counts': stats,
                'selected_plates': list(calib_info), 'calibrations': calib_info,
                'n_targets': len(targets), 'executed_in_simulator': bool(args.execute),
                'calibration_note': 'Assumed simulation plate map is not independently validated.'}
    (args.output/'task12_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Task 7: {stats['selected']} OK / {stats['total']} total tips; "
          f"{len(targets)} selected across {len(calib_info)} plates.")
    for plate, info in calib_info.items():
        print(f'  {plate}: {info["status"]}; verified={info["calibration_verified"]}')
    print('Wrote:', args.output/'task12_target_manifest.csv')
    if not args.execute:
        print('DRY RUN ONLY. Add --execute when the simulator and plate calibration are ready.')
        return

    from ot2.sim import make_sim
    from ot2.teleport import teleport_safe
    from ot2.envelope import PRACTICAL_ENVELOPE
    from task10_pid_controller.controller import build_pid
    from simulate_pid import run_plate
    from demo_gifs import PlateGif, preview_image

    pid = build_pid()  # ORIGINAL Task 10 tuned gains; never replace
    results = []
    for plate in calib_info:
        selected = [r for r in targets if r['plate'] == plate]
        w, h = sizes[plate]
        calibration = map_for_plate(plate, w, h, cfg, args.allow_assumed_plate_map)
        gif = None
        if args.previews:
            image_path = preview_image(args.previews, plate)
            if image_path:
                gif = PlateGif(plate, selected, calibration,
                               image_path, args.gif_dir/f'{plate}_pid.gif', show_3d=args.gif_3d)
            else:
                print(f'  Missing preview photo for {plate}: no GIF for this plate')
        sim = make_sim(num_agents=1, render=False, rgb_array=False)
        try:
            # Same validated centering technique as Task 10 standalone evaluation.
            teleport_safe(sim, PRACTICAL_ENVELOPE.center)
            result = run_plate(sim, pid, selected, renderer=gif)
            results.extend(result)
        finally:
            sim.close()
        if gif is not None:
            gif.save()
            print('  GIF:', gif.gif_path)
        print(f'  {plate}: {sum(r["simulated_deposit"] for r in result)}/{len(result)} '
              'positioned and marked with simulated deposit')

    save_csv(args.output/'task12_pid_position_results.csv', results)
    numerical = [r['position_error_mm'] for r in results if np.isfinite(r['position_error_mm'])]
    complete = sum(bool(r['simulated_deposit']) for r in results)
    summary = {'n_targets': len(results), 'n_simulated_deposits': complete,
               'success_percent': 100.0 * complete / len(results),
               'mean_pid_position_error_mm': float(np.mean(numerical)) if numerical else None,
               'max_pid_position_error_mm': float(np.max(numerical)) if numerical else None,
               'error_definition': '3-D pipette pose minus requested *mapped* target; '
                                   'NOT independently measured root-tip error',
               'calibration_verified_all': all(i['calibration_verified'] for i in calib_info.values())}
    (args.output/'task12_pid_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
