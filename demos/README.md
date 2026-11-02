# Synthetic Coordinate-Mapping Demonstration

This demonstration uses generated root-shaped traces to illustrate the interface between image analysis and robot-space target planning. It makes no use of restricted client images or trained model weights.

From the repository root, run:

```powershell
python demos/synthetic_pipeline.py --output demo-output
python -m unittest discover -s tests -v
```

The example produces an SVG visualization and tabular target coordinates. It demonstrates coordinate handling, not neural-network inference, real-world calibration, or successful inoculation. The commands require `demos/synthetic_pipeline.py` and its accompanying tests to be installed from the separate synthetic-demo package.
