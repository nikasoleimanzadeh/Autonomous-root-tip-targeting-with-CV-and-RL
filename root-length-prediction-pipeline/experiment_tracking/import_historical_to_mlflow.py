"""Create LOCAL MLflow runs from historical notebook records; never pretend contemporaneous logging.
No images, masks, weights, client samples, identifiers or raw predictions are sent anywhere.
Requires mlflow: python -m pip install mlflow
"""
from pathlib import Path
import csv
import math
import mlflow

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd()
# Default file-based backend inside local repository; do not commit mlruns/.
URI=(ROOT/'mlruns').resolve().as_uri()
mlflow.set_tracking_uri(URI)
mlflow.set_experiment('NPEC-Task8-Historical-Reconstruction')

safe_parameters={'patch_size','fragment_gap_px','threshold','train_patches','val_patches','epochs_recorded','best_epoch'}
metric_fields={'patches_total','patches_nonempty','patches_empty','empty_patch_percent','best_val_f1','best_val_loss',
'root_pixels_total','assigned_root_pixels','assigned_root_percent','plants_detected','image_count','root_starts',
'roots_measured','roots_ok','roots_review','kaggle_rows','public_smape_percent','private_smape_percent','private_rank'}

with (HERE/'runs.csv').open(newline='',encoding='utf-8') as f:
    experiments=list(csv.DictReader(f))
with (HERE/'training_epoch_metrics.csv').open(newline='',encoding='utf-8') as f:
    epochs=list(csv.DictReader(f))
for row in experiments:
    with mlflow.start_run(run_name=row['run_name']):
        mlflow.set_tags({'historical_reconstruction':'true','provenance':'saved_notebooks_and_readme',
                         'stage':row['stage'],'verification':'not_logged_during_original_competition'})
        mlflow.log_param('variant',row['variant'])
        for name in safe_parameters:
            if row.get(name):mlflow.log_param(name,row[name])
        for name in metric_fields:
            if row.get(name):mlflow.log_metric(name,float(row[name]))
        if row['run_name']=='unet_training_512':
            for e in epochs:
                step=int(e['epoch'])
                for k in ('train_loss','train_f1','val_loss','val_f1'):
                    mlflow.log_metric(k,float(e[k]),step=step)
print(f'Imported {len(experiments)} reconstructed records into local {URI}')
print('Launch: mlflow ui --backend-store-uri ./mlruns')
