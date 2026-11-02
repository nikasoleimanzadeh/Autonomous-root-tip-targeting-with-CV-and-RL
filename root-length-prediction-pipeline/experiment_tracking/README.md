# Experiment Tracking and Model Evaluation

The root-length prediction pipeline went through several iterations. This directory preserves the recorded settings, comparisons, and metric exports used to understand which changes improved performance and which did not generalize.

## Contents

- [`runs.csv`](runs.csv): recorded experiment configurations and results.
- [`threshold_comparison.csv`](threshold_comparison.csv): comparison of post-processing threshold choices.
- [`training_epoch_metrics.csv`](training_epoch_metrics.csv): available model-training metrics by epoch.
- [`import_historical_to_mlflow.py`](import_historical_to_mlflow.py): utility for importing historical numeric experiment records into MLflow.

## Interpreting the results

Use the configuration and run identifiers when comparing results. Validation F1 and leaderboard sMAPE measure different stages of the system and should not be treated as interchangeable. The final private leaderboard score provides a more conservative view of generalization than the public score.

The records are an engineering history, not a benchmark on new independent data. Numeric metrics can be suitable for publication, but verify that the exports contain no restricted identifiers or client-derived fields before sharing.
