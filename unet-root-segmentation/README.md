# Task 5 – Root segmentation

For this task I trained a residual U-Net to detect roots in Petri dish images and wrote an inference notebook to use it on a raw image.

## What I used

I used the 512 × 512 image and mask patches I prepared in Task 4. My training set has 9,600 patches from Y2B_23 and Y2B_24, and the validation set has 612 patches from the Y2B_25 `val_` images. I kept the validation images out of training.

I used a residual U-Net with group normalization and a loss combining Dice, BCE and soft clDice. I also balanced the training patches by root density because a lot of patches contain very few root pixels.

## My result

My best recorded validation F1 was **0.8024**. Early stopping restored the weights from epoch 13. The training notebook includes the original training log, validation scores and performance plots.

## Files

`task_5_training.ipynb` contains the training code and saved results. `task_5_inference.ipynb` loads my model, processes `task5_test_image.png` and saves the final prediction.

`NikaSoleimanzadeh_244258_unet_model_512px.h5` is the trained model I use for inference. I kept the filename the same in both notebooks.

`task5_test_image_prediction_nikasoleimanzadeh_244258.png` is the binary prediction for the test image. It should be the PNG saved directly by the inference code, with the same dimensions as the original image (4112 × 3006 pixels) and only 0 and 255 pixel values. It should not be a resized preview or screenshot.

## Running it

I run the training notebook using the prepared dataset at `/home/y2a/dataset/`. For inference, I put the `.h5` file and `task5_test_image.png` in the notebook's working directory, then run the inference notebook from top to bottom. The last code cell saves the prediction PNG.

I used `tf_keras` for both training and inference so the custom group normalization layer loads correctly. The image preprocessing, mask reconstruction and output checks are in the inference notebook.
