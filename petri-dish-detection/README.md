# Task 2 — Petri dish detection

For this task, I used traditional computer vision to find and crop Petri dishes from the root images. I wanted to use the same method for Y2B_23, Y2B_24 and Y2B_25 instead of writing separate code for each dataset.

My work is in [`task_2.ipynb`](task_2.ipynb). The notebook has the code, example outputs and the checks I ran.

## What I did

I converted each image to grayscale and looked at brightness across the middle rows and columns. After smoothing those values, I used Otsu thresholding to find the dish area. I then adjusted the crop around its centre so the final image was square.

I also used Scharr gradients as a second way of estimating the dish edges. I compared those positions with my final crop and checked whether the difference was within 30 pixels on each side.

## My results

I included one example from Y2B_23, one from Y2B_24 and two from Y2B_25. All four square crops passed the checks in the notebook. The largest difference compared with the automatic Scharr estimate was 13 pixels.

The Scharr check is still an automatic estimate, not a hand-labelled measurement of the real edge. The notebook shows how the same code performed on these four examples; it isn't a full test of every dataset image.

## Running it

I used Python with `opencv-python`, `numpy`, `matplotlib` and Jupyter. The raw images aren't included in this repository. The notebook looks for an `images` folder in the task or project folder first, then my original `Desktop/flowerpainting/Task2` folder. If the images are stored somewhere else, I can change `IMAGE_DIR` in the setup cell.

The four example filenames are already in the notebook, and the saved outputs are kept so the results can still be viewed without the raw dataset.
