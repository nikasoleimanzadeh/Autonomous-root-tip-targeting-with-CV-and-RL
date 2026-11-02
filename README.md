# Autonomous Root-Tip Targeting with Computer Vision and Robotic Control

**From microscope images to target coordinates for a simulated laboratory robot.**

This project explores how a vision system can locate and measure plant roots, then translate detected root-tip positions into motion targets for an Opentrons OT-2 digital twin. It combines classical image processing, a residual U-Net segmentation model, root-architecture analysis, and two approaches to robotic control: tuned PI control and reinforcement learning.

The motivation is practical: root phenotyping and targeted treatment involve repetitive image analysis and precise positioning. This prototype investigates which parts of that workflow can be automated—and where measurement, calibration, and controller errors make human verification essential.

> **Scope:** The robotic demonstrations run in simulation. Positioning accuracy in the digital twin is **not** evidence of successful physical inoculation. The image-to-robot calibration in the integration work has not been independently validated against measured landmarks.

## How the system fits together

```text
Petri dish image
    ↓  classical image processing
Dish detection and crop
    ↓  residual U-Net inference
Root segmentation mask
    ↓  plant assignment + root tracing
Primary-root lengths and tip coordinates
    ↓  coordinate transformation and validation
OT-2 simulator target positions
    ↓  PI or reinforcement-learning controller
Simulated pipette movement + evaluation
```

## Explore the implementation

| Component | What it demonstrates |
|---|---|
| [Petri dish detection](petri-dish-detection/) | Square region-of-interest extraction using intensity profiles and edge checks |
| [Plant instance segmentation](plant-instance-segmentation/) | Separating neighboring plants with conventional computer vision |
| [Residual U-Net root segmentation](unet-root-segmentation/) | Patch-based training and inference for thin root structures |
| [Root-length prediction](root-length-prediction-pipeline/) | Root-start refinement, root tracing, length estimates, and experiment tracking |
| [Robotics simulation](robotics-simulation/) | OT-2 state observations, motor control, and work-envelope measurement |
| [PI motion control](pid-motion-control/) | Per-axis gain tuning and position-error evaluation |
| [Reinforcement-learning control](reinforcement-learning-control/) | Gymnasium environment, PPO/SAC experiments, and controller comparisons |
| [Vision-guided integration](vision-guided-robotics-and-benchmarking/) | Coordinate conversion, target validation, simulated trajectories, and benchmarking |
| [Calibration interfaces](cv-to-robot-calibration-data/) | Input schema and coordinate-frame handoff; protected measurements are not redistributed |
| [Project presentation](project-showcase/) | A walkthrough of the proposed workflow, findings, and limitations |

The shared OT-2 utilities live in [`src/ot2/`](src/ot2/); experiment logs are organized under [`runs/`](runs/).

## Selected results

These numbers come from the documented experiments and should be read with their specific evaluation conditions:

- **Root segmentation:** the residual U-Net training record reports a best validation F1 of **0.8024**, on the specified validation patches.
- **Primary-root length prediction:** the final competition pipeline reports **16.159% private sMAPE** and **4th place**. Its public sMAPE was **3.713%**; the gap is a useful reminder not to treat the public leaderboard as held-out validation.
- **PI control:** the simulator evaluation README reports **20/20 targets within 1 mm** in its initial test and gives the underlying error and settling-time measurements.
- **System integration:** target navigation and simulated deposit markers were evaluated; physical calibration and liquid placement are **not independently established**.

See each component's README for methods, data splits, metrics, and limitations. These are results of different tests, not a single end-to-end accuracy score.

## Technology

Python · OpenCV · NumPy · Jupyter · TensorFlow / tf-keras · Matplotlib · PyBullet · Gymnasium · Stable-Baselines3 · MLflow · Git

## Reproducibility and demonstrations

A separate synthetic-data demo is available as an optional reproducibility upgrade. It illustrates coordinate handling without relying on protected project assets; it does not replace the trained segmentation model or the OT-2 simulator.

## Running and reproducing work

The notebooks and robotics scripts were developed in different environments. The included `pyproject.toml` describes the OT-2 Python package, but **the complete vision-to-robot workflow is not currently a one-command install**.

1. Create a Python environment appropriate for the component you want to run.
2. Follow that component's README and check its dependency and input-data expectations.
3. Use the OT-2 simulator setup described under [`robotics-simulation/`](robotics-simulation/) for robotic control.
4. Supply independently licensed public or synthetic images for vision demonstrations, and configure paths locally. The original project dataset is not redistributed.

Some scripts still contain historical filesystem assumptions; portability is an ongoing improvement. This repository documents the implementation and experiments rather than claiming a turnkey deployment.

## Data use and responsible publication

The original project used restricted course/client material. Those images, labels, client-derived outputs, and trained model checkpoints must not appear in a public portfolio. A public release should include only self-authored code and independently licensed or synthetic demonstrations, with notebook outputs and Git history audited before publication.

**Before publishing this repository, remove the restricted assets identified in the [publication audit](docs/publication-audit.md).** This README does not itself make the current archive safe to publish.

## Limitations and next steps

The main open questions are robustness across imaging conditions, accurate extraction of individual root tips, independent camera-to-robot calibration, and control success under realistic disturbances. Future work would include a portable test dataset, deterministic end-to-end tests, calibrated robot-space targets, and a reproducible environment lockfile.

This work was developed as an academic engineering project. The technical design, experimental decisions, and conclusions are presented here as a portfolio case study rather than an assignment submission.
