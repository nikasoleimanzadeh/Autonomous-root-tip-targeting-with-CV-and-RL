# Task 8 — Primary root length prediction

For Task 8, I put together the steps from my earlier tasks into one pipeline. I give it a folder of raw Hades images, and it predicts the primary root length for each root it detects. There isn't any model training in this notebook.

I used my 512 × 512 root segmentation U-Net from Task 5. The weights are already in this repository, so I didn't upload a second copy here:

[Root segmentation model — Task 5](../Task%205/NikaSoleimanzadeh_244258_unet_model_512px.h5)

## What I did

I first crop the Petri dish and run the U-Net on overlapping image patches. I combine the predictions into a root mask. After that, I detect root starting points, separate individual root systems, and trace their primary roots to estimate length in pixels.

I call my root-start refinement **V5.1**. It's not another trained model. The original detector uses the root mask to find likely starts near the top of the plate. V5.1 filters out starts that are too low, tries to find small roots that were missed, and merges duplicate detections. Those start coordinates help me assign the root pixels to individual plants before tracing their primary roots.

The main file is [`task_8.ipynb`](task_8.ipynb). I kept the processing, prediction and measurement steps in one notebook instead of splitting them into helper notebooks. I also kept the original tracing calculations and their `OK`/`CHECK`/`FAILED` quality labels. A quality flag is a warning for review, not a guarantee that a root was measured correctly.

## Running it

I set `INPUT_IMAGE_DIR` and `ROOT_MODEL_PATH` in the first code cell. On the school server, I used `/home/y2a/Kaggle` and `/home/y2a/NikaSoleimanzadeh_244258_unet_model_512px.h5`. If I run the notebook from this repository on another machine, I need to point `ROOT_MODEL_PATH` to the Task 5 model linked above.

The notebook accepts a directory of PNG, JPG, TIFF or BMP images. It saves the main root-length results to `outputs/07_primary_roots/task7_primary_root_lengths.csv` inside the configured pipeline output directory. It also saves `outputs/batch_summary.csv`, with a row for every input image, including images where no roots were detected.

`GENERATE_KAGGLE_SUBMISSION` is disabled by default. If I enable it, the notebook uses the original 19-image competition format to generate a 95-row submission. That formatting is only for Kaggle; it isn't used to produce the general root-length CSV.

## Results and limitations

In the saved run, the pipeline processed **19 of 19 images**, measured **87 roots**, and finished without notebook errors. Of those measurements, **76 were marked `OK`** and **11 were flagged for review**. The notebook contains the output figures and batch summary from that run.

The pipeline ran on the competition image folder, but I haven't independently verified the predictions against annotated root lengths in a separate unseen batch. The root-start and RSA steps still use thresholds I selected while developing the pipeline. I wouldn't claim that the successful run alone proves accuracy on all possible Hades images.

## Kaggle evidence

My submitted file was `submission_task8.csv`. Kaggle shows a **public sMAPE of 3.713%** and a **final private sMAPE of 16.159%**, with a **final leaderboard position of 4th**. The private score falls between 10% and 45%, so it is in the assignment's partial-points bracket rather than the full 20-point bracket.

[Submission score screenshot](evidence/kaggle_submission_scores.png)  
[Final private leaderboard screenshot](evidence/kaggle_private_leaderboard.png)

The final block presentation is a separate assignment deliverable.
