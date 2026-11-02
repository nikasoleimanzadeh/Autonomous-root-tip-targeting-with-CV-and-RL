# Plant Instance Segmentation

This component distinguishes neighboring plants before root measurements are assigned to an individual specimen. It uses classical computer vision rather than a separate instance-segmentation network.

## Approach

The [notebook](task_3.ipynb) locates shoots with thresholding and connected components, enhances thin roots using background subtraction and oriented morphological kernels, and links root networks back to their likely shoot locations. The method avoids simply splitting an image into vertical strips, which can cut across diagonal roots.

## Results

Both recorded example images yielded **five non-empty instance masks**. The outputs were inspected visually; this is not an independently labeled instance-IoU benchmark.

## How to explore

Provide permitted input images, configure `IMAGE_DIR` in the notebook, and run the cells in order. Historical filenames and paths are still used inside the notebook and should be generalized in a future portability pass.

## Limitations

Weak, overlapping, or disconnected root structures can be missed or assigned incorrectly. The procedure assumes a relatively consistent plate layout and should be validated on a broader, independently labeled image set.
