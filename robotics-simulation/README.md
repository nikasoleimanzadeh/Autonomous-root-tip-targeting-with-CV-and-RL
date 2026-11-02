# OT-2 Robotics Simulation

The project uses an Opentrons OT-2 digital twin to study 3-D pipette positioning before integrating image-derived targets. This module establishes how to command the simulated motors, inspect state observations, and determine reachable workspace boundaries.

## Implementation

- [`check_setup.py`](check_setup.py): environment sanity checks
- [`robotics_interaction.py`](robotics_interaction.py): movement commands and state observations
- [`eight_corners.py`](eight_corners.py) and [`eight_corners_gui.py`](eight_corners_gui.py): work-envelope exploration and visualization
- [`results/`](results/): recorded coordinates and a simulation GIF

The scripts use the PyBullet-backed OT-2 twin and shared helpers from [`src/ot2/`](../src/ot2/). Environment setup depends on access to the simulator implementation; historical references may point to a local `OT2_SIM_DIR`.

## Documented workspace

The recorded practical pipette range is X **−0.1870 to 0.2530 m**, Y **−0.1705 to 0.2195 m**, and Z **0.1700 to 0.2895 m**. These are measured for the particular simulated setup, not universal hardware limits.

## Why it matters

Subsequent controllers reject unreachable goals and evaluate error in the same coordinate frame. Simulation is a useful development environment, but its dynamics and geometry still require verification before transfer to hardware.
