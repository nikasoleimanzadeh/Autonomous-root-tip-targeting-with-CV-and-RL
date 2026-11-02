# Task 8 — reconstructed experiment history (2026-10-30)

This directory summarizes **16 distinct historical diagnostic / development / training / pipeline experiment records** extracted from saved notebooks and the existing Task 8 README. It is an evidence reconstruction completed after the original experiments, **not** a contemporaneous W&B or MLflow export. It does **not** establish that 10 Task 8 runs were formally tracked during the competition or that all records were Kaggle-submitted pipeline versions. Do not describe all 16 rows as Kaggle submissions or independently trained models.

## Contents

- `runs.csv` — 16 verifiable historical stage evaluations; blank metrics mean the notebook did not report a value.
- `training_epoch_metrics.csv` — 21 epochs from **one** residual U-Net training run, not 21 separate training runs; best validation F1 0.802433 at epoch 13.
- `threshold_comparison.csv` — numeric comparisons across the 19 test images for segmentation thresholds 0.50 versus 0.35.
- `dashboard.png` — locally recreated plot of the results, prominently identified as **not a screenshot of W&B/MLflow**.
- `import_historical_to_mlflow.py` — optional local-only importer; creates new MLflow records tagged `reconstructed` without retraining. Do not pass off the backfilled records as live historical tracking.
- `../development/` — five unmodified notebook copies with preserved outputs. The two identically hashed copies named `2026fin` and `20256att` were deduplicated.

## Numbers taken from saved results

- Task 4 patch sizes: 128px 90.4268% empty, 256px 83.2605% empty, 512px 70.9837% empty.
- Task 5 U-Net: 21 epochs; validation F1 peaked at 0.802433, restored epoch 13.
- Task 6 gap test on one image: 36px assigned 101,291 of 103,428 root pixels; 60px assigned 103,292 of 103,428.
- Task 8 detection: base 86 starts, early V5.1 85, revised V5.1 87 on 19 images. These are **different notebook iterations**, not an accuracy comparison against ground truth.
- Threshold test: cutoff 0.50 produced 493,844 predicted root pixels; cutoff 0.35 produced 501,198 across 19 images (no independent accuracy scores).
- Final project: 19 images, 87 measured roots, 76 marked OK, 11 for review, 95 submission rows; existing `Task 8/README.md` reports public sMAPE 3.713%, private sMAPE 16.159% and private rank 4. The Kaggle screenshots remain under `Task 8/evidence/` in your repo.

## To make **actual MLflow tracking evidence** without rerunning any experiments

From your repo root with MLflow installed:

```powershell
python -m pip install mlflow
python .\deliverables\task08_kaggle_competition\experiment_tracking\import_historical_to_mlflow.py
mlflow ui --backend-store-uri .\mlruns
```

Then open `http://127.0.0.1:5000` and select the experiment `NPEC-Task8-Historical-Reconstruction`. Capture a **real screenshot** of the MLflow run comparison interface and replace `dashboard.png` only if your assignment requires the original tool dashboard specifically. The included plot is not one. Avoid committing `mlruns/`, and never upload images, masks, model weights or predictions to a cloud experiment service.

**Scope warning:** Importing these historical results now does not prove that you tracked them live during the competition or linked Kaggle submissions to run names at the time. It does make the recovered experiments inspectable without new training. 
