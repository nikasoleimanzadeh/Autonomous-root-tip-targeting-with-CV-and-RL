# PI Motion Control for the OT-2

This component implements independent feedback controllers for the simulated pipette's X, Y, and Z axes. Although the implementation supports PID terms, the best tested gains have **Kd = 0** on all three axes, so the final tuned configuration is a **PI controller**.

## How it works

[`controller.py`](controller.py) runs the controller, [`tune_pid.py`](tune_pid.py) searches candidate gains, and [`test_pid.py`](test_pid.py) evaluates randomly sampled reachable targets. The implementation includes output limits and anti-windup handling. A visual example is available in [`results/pid_controller.gif`](results/pid_controller.gif).

## Selected gains

| Axis | Kp | Ki | Kd |
|---|---:|---:|---:|
| X | 58.1873 | 1323.1537 | 0 |
| Y | 77.0519 | 1782.3085 | 0 |
| Z | 103.5513 | 1860.0442 | 0 |

## Evaluation

The recorded initial test reports **20/20 targets within 1 mm** in the configured simulation, with a **mean settling time of 0.240 s**. Full per-target results and additional seeded summaries are stored in [`results/`](results/). These unusually small final position errors depend on simulator fidelity, stopping conditions, and test setup; they should not be treated as physical robot accuracy.

## Running

Install the OT-2 simulator and the project's dependencies, then run `python pid-motion-control/test_pid.py` from the repository root after checking simulator imports and paths. The scripts retain some historical local environment assumptions.

## What remains

Broader robustness testing, disturbance analysis, and validation on hardware would be required before operational use.
