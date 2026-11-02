# V5.1 iteration and research notebooks

These notebooks preserve the development process and the saved output of actual executed experiments. They supplement the production `Task 8/task_8.ipynb` pipeline without modifying or replacing it.

**Development evidence path:** `deliverables/task08_kaggle_competition/development/`.

`Task8_V51_experiments.ipynb` documents the first root-start detector, early V5.1 refinement, thresholds of 0.50/0.35 and root assignment checks. `Task8_V51_clean_pipeline.ipynb` contains the revised refinement, individual-root assignment, primary tracing and 95-row Kaggle output. `Task8_generalized_candidate.ipynb` documents the input-folder generalization and 87 measured roots on the saved test run. `Task6_7_RSA_exploration.ipynb` contains earlier segmentation, fragment gap (36/60), RootNav and bottom-up tracing experiments. `Task4_patch_size_study.ipynb` supports the 128/256/512 patch-size choice.

**Source provenance:** all notebooks are unmodified copies of the uploaded originals. Their saved outputs are historical runs, not fresh executions. The notebook code contains absolute paths from the original school server; run paths would need configuring before execution elsewhere.

The experiment table at `experiment_tracking/runs.csv` is a source-based historical reconstruction, not an original W&B/MLflow export.
