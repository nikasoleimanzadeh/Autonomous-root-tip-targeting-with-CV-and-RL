# Task 3 — Plant instance segmentation

For Task 3, I used traditional computer vision to separate five plants in each of the two images provided for the assignment. I wanted each plant to have its own label, including its shoot and roots, rather than separating individual plant organs.

## What I did

I started by finding the five shoots near the top of the images using thresholding and connected components. Then I enhanced the thin roots using background subtraction and line-shaped morphological kernels at different angles.

I detect the root networks across the whole image rather than cutting it into five vertical strips. This matters because some roots grow diagonally, and a strip-based approach can cut them off or give them to the wrong plant. I match the networks to the plants based on where each root component starts, then combine each shoot and root network into one instance mask.

I only used OpenCV, NumPy and Matplotlib. There is no machine-learning or deep-learning model.

## Notebook

`task_3.ipynb` contains my code, function explanations and the saved results for both assignment images:

- `task_3_image_1.png`
- `task_3_image_2.png`

The notebook shows each original image next to its coloured segmentation. Label 0 is the background, and labels 1 to 5 are the plants from left to right.

## How I ran it

I used Python with `opencv-python`, `numpy` and `matplotlib` installed.

The two PNGs are available in the course's Task 3 materials. To run the notebook again, put the images in `Task 3/images/` or change `IMAGE_DIR` in the first code cell to the folder where they are saved. The notebook accepts both the normal filenames and copies ending in ` (1).png`.

I then ran the notebook from top to bottom. The input images aren't included in this folder, but the results from my original run are saved in the notebook.

## Results

Both supplied images produced five non-empty plant masks. The notebook prints the plant positions and pixel counts and shows the results side by side for a visual comparison.

For the first image, the five saved plant areas were **7,376, 5,756, 4,726, 4,936 and 4,820 pixels**. For the second image, they were **41,805, 44,888, 48,404, 40,187 and 39,961 pixels**.

I kept the same overall segmentation approach for both images. Some very faint roots can still be difficult to capture, but my main goal here was to separate the five plants rather than trace every individual root.
