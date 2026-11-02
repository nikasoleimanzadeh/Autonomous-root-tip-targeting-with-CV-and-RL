# Synthetic vision-to-robot handoff

This small demonstration makes the coordinate transformation part of the project easy to inspect without requiring private datasets, model checkpoints, or the OT-2 simulator. It generates five fictional root traces, identifies the endpoint of each trace, converts those locations into illustrative robot-space targets, and writes an SVG and two machine-readable outputs.

It runs on **Python 3.10+ using only the standard library**:

```powershell
python demos/synthetic_pipeline.py --output demo-output
python -m unittest discover -s tests -v
```

Open `demo-output/synthetic_roots.svg` in a browser. See `targets.csv` for the coordinates and `report.json` for the explicit scope of the demonstration.

## What this demonstrates, and what it doesn't

- **Does:** illustrate root endpoints, axis conventions, coordinate conversion, deterministic export, and basic input validation.
- **Doesn't:** segment real images, run the trained U-Net, validate physical plate calibration, operate the OT-2 digital twin, or execute PID/RL control. The polylines are generated in memory; their endpoints are known directly rather than inferred from an image.

The scale assumes a square 150 mm plate represented by 600 × 600 generated pixels. The plate origin and image-to-robot axis convention are illustrative and drawn from the project's documented integration geometry, **not verified measurements**. Do not use these generated coordinates to operate equipment.

The full engineering implementations remain in the computer-vision, robotics, and integration folders. This demo is deliberately a safe public entry point, not a substitute for those systems.
