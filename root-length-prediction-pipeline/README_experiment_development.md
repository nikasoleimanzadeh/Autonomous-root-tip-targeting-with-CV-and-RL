# V5.1 iteration and research notebooks

These notebooks preserve the development process and the saved output of actual executed experiments. They supplement the production `Root-length prediction/root_length_prediction.ipynb` pipeline without modifying or replacing it.

**Development evidence path:** `root-length-prediction-pipeline/development/`.

`v5_1_pipeline_experiments.ipynb` documents the first root-start detector, early V5.1 refinement, thresholds of 0.50/0.35 and root assignment checks. `v5_1_pipeline_prototype.ipynb` contains the revised refinement, individual-root assignment, primary tracing and 95-row Kaggle output. `generalized_root_length_pipeline.ipynb` documents the input-folder generalization and 87 measured roots on the saved test run. `root_architecture_exploration.ipynb` contains earlier segmentation, fragment gap (36/60), RootNav and bottom-up tracing experiments. `image_patch_size_study.ipynb` supports the 128/256/512 patch-size choice.

**Source provenance:** all notebooks are unmodified copies of the uploaded originals. Their saved outputs are historical runs, not fresh executions. The notebook code contains absolute paths from the original school server; run paths would need configuring before execution elsewhere.

The experiment table at `experiment_tracking/runs.csv` is a source-based historical reconstruction, not an original W&B/MLflow export.
