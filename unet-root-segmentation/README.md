# Residual U-Net for Root Segmentation

Fine plant roots occupy very few pixels and are easy to lose during preprocessing. This component trains a residual U-Net to predict binary root masks from image patches and reconstructs full-resolution predictions during inference.

## Model and data preparation

The [training notebook](task_5_training.ipynb) uses **512 × 512** patches, residual blocks, group normalization, and a combined Dice, binary cross-entropy, and soft-clDice loss. Root-density-aware sampling helps prevent nearly empty patches from dominating optimization.

The documented training experiment used **9,600 training patches** and **612 validation patches** drawn from distinct imaging groups. The validation protocol and grouping are described in the notebook.

## Recorded result

The best reported validation **F1 is 0.8024**, with early stopping restoring the checkpoint from epoch 13. This is the documented validation result, not a guarantee of performance on unrelated plant species or imaging setups.

## Files

- [`task_5_training.ipynb`](task_5_training.ipynb): training configuration, loss design, and learning curves
- [`task_5_inference.ipynb`](task_5_inference.ipynb): raw-image inference, prediction reconstruction, and mask checks

The original NPEC-data-trained `.h5` checkpoint and client-derived prediction images **must not be published**. To reproduce a public demonstration, train a new model on permitted public or synthetic data and document its license and evaluation separately.

## Limitations

Some scripts assume the original server's `/home/y2a/...` paths. Full reproducibility also requires a clearly versioned environment, externally available data, and held-out full-image testing.
