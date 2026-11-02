# Petri Dish Detection and ROI Extraction

Before analyzing roots, the pipeline isolates the plate from the surrounding image. This reduces background noise and gives later stages a consistent region of interest.

## Approach

The [notebook](task_2.ipynb) uses grayscale intensity profiles, smoothing, and Otsu thresholding to estimate dish boundaries. Scharr gradients provide an independent edge estimate; the selected crop is adjusted to a square. This stage uses classical image processing rather than a trained detector.

## What the experiment shows

Four recorded examples from different imaging batches produced square crops. The largest difference between the selected boundary and the automatic Scharr estimate was **13 pixels**. Because both measurements are automatic, this is an internal consistency check—not a claim of 13-pixel error against manually measured ground truth.

## How to explore

Open `task_2.ipynb` and set `IMAGE_DIR` to a folder of images you are allowed to use. The notebook contains historical local path assumptions that may need updating. Do not distribute restricted source images or embedded outputs.

## Limitations

The reported results cover the four examples shown, not a comprehensive benchmark across every image or lighting condition. A future evaluation should use manually annotated dish boundaries and report failure cases.
