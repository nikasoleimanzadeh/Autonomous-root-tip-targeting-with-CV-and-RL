# Reinforcement Learning for Pipette Positioning

The control problem is to move a pipette to a requested three-dimensional position and let it settle before any simulated deposit. This folder explores reinforcement learning as an alternative to conventional PI/PID control.

## Environment and controllers

[`ot2_gym_wrapper.py`](ot2_gym_wrapper.py) adapts the OT-2 simulator to a Gymnasium-style interaction loop, exposing position, velocity, and goal information. Training and evaluation scripts include [`train_rl.py`](train_rl.py), [`evaluate_rl.py`](evaluate_rl.py), and [`compare_final_controllers.py`](compare_final_controllers.py).

The project contains two research tracks:

- **PPO experiments:** baselines, tuning, and longer training runs, with evaluation files under `results/` and `../runs/`.
- **SAC / Path Y:** [`Path_Y/`](Path_Y/) contains reward-design experiments, guided training, behavior-cloning-related work, and independent comparisons.

## Evaluation approach

The experiments compare final distance to target, settled-target success, and episode duration under defined start/goal cases. Diagnostic runs and controller comparison figures are stored in [`evidence/`](evidence/) and [`results/`](results/). Not every experiment succeeded, and a short-distance success should not be generalized to the full reachable workspace.

## Reproducing experiments

Start with `test_wrapper.py` to inspect the environment interface, then review the configurations and training scripts. Results depend on the exact simulator build, reward version, observation/action scaling, and random seeds. The historical run archives are useful as research records but are not a curated, small public release.

## Limitations

The trained agents do not establish a production-ready controller. Generalization across target positions, safety constraints, and simulation-to-hardware transfer remain open. Check saved checkpoints and data files for redistribution rights before publication.
