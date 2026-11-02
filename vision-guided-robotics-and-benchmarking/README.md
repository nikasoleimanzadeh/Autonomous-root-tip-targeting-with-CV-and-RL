# Vision-Guided Robot Targeting and Benchmarking

This module links measured root-tip positions to robot motion commands in the OT-2 simulation. It also compares control strategies and records where accuracy may be lost along the way.

## Integration flow

1. Read detected root-tip coordinates and their quality flags.
2. Load the Petri dish crop dimensions and plate mapping assumptions.
3. Transform image coordinates to robot-space targets.
4. Reject targets outside the measured work envelope.
5. Approach and settle above each target with the selected controller.
6. Record the simulated trajectory, target error, and optional deposit visualization.

The core scripts are [`cv_adapter.py`](cv_adapter.py), [`calibration.py`](calibration.py), [`run_integration.py`](run_integration.py), and [`benchmark_task13.py`](benchmark_task13.py). The [`tests/`](tests/) and [`results/`](results/) directories contain the corresponding experiments and recorded outcomes.

## A critical distinction: mapping vs. physical calibration

The implementation uses an **assumed image-to-world mapping**, with `calibration_verified = false`. The reported simulator positions and simulated deposit markers do **not** show that photographed root tips were correctly targeted on physical agar plates. The code explicitly guards the assumed mapping with an opt-in flag rather than silently claiming verified calibration.

The implementation also separates **positioning error** from **root-tip detection error**, **coordinate-transform error**, and **liquid-placement error**. A sub-millimetre controller result does not make the entire pipeline sub-millimetre accurate.

## Benchmarking

The scripts collect timing and positioning metrics and compare PI/PID and RL behavior under the selected test cases. Consult the stored `results/` summaries and figures for specific run configurations; do not interpret a simulator GIF as independent measurement of biological inoculation.

## Next steps

Measure non-collinear image-to-robot landmark pairs, evaluate held-out landmarks, propagate uncertainty through the coordinate transform, and benchmark physical deposition separately from motion control. Client-derived CSVs and plate previews should be excluded from a public release.
