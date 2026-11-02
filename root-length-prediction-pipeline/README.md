# Primary Root-Length Prediction Pipeline

A binary root mask is useful, but plant analysis also needs measurements for individual root systems. This pipeline converts microscopy images into estimated primary-root lengths by combining image preprocessing, U-Net inference, root-start localization, instance assignment, and root tracing.

## Pipeline

The [main notebook](task_8.ipynb) accepts a directory of images and performs dish cropping, overlapping-patch inference, mask reconstruction, root-start detection, root grouping, and length estimation. The **V5.1 refinement** is a rule-based improvement to root-start detection, not a separately trained neural network. It filters implausible starts, recovers candidates, and merges duplicates before tracing.

The [`development/`](development/) notebooks record experiments with patch size, root-system architecture, and pipeline iterations. [`experiment_tracking/`](experiment_tracking/) contains the tracking structure and historical metrics.

## Measured results

The recorded batch processed **19/19 images** and returned **87 root measurements**: 76 marked `OK` and 11 flagged for review. These quality labels are internal checks, not ground-truth correctness labels.

The competition record reports **3.713% public sMAPE**, **16.159% private sMAPE**, and **4th place**. The private score is the more useful summary of performance on the competition's held-out split.

## Reproduction notes

Set `INPUT_IMAGE_DIR`, `ROOT_MODEL_PATH`, and output locations in the notebook. The original server paths and task-numbered helper names are historical and may need updating. The original model checkpoint, competition inputs, and client-derived notebook outputs are not redistributable; use permitted public or synthetic data for an independently reproducible example.

## Limits

Performance remains sensitive to segmentation errors, overlapping roots, root-start thresholds, and imaging conditions. The competition result is not equivalent to validated accuracy on arbitrary new datasets.
