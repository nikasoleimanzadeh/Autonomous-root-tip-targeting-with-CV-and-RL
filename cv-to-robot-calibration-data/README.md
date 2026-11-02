# Coordinate Handoff and Calibration Inputs

This component defines the bridge between root-tip detections in image coordinates and the position requests consumed by the simulated OT-2 controller.

Root-tip tables may contain image identifiers, plant identifiers, tip coordinates, and a detection-quality status. Crop metadata provides the image dimensions needed for coordinate conversion. The integration logic that consumes these values lives in [`../vision-guided-robotics-and-benchmarking/`](../vision-guided-robotics-and-benchmarking/).

**Publication note:** The original CSVs and preview images in this directory appear derived from the restricted project dataset and should not be included in a public portfolio. Replace them with synthetic examples of the expected schema, or document the format without shipping actual measurements.

The mapping used for the recorded demonstration was not independently calibrated against measured landmarks, so this folder should not be represented as proof of physical positioning accuracy.
