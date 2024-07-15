import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging


def write_patches_as_png(
    gt_patch: np.ndarray,
    pred_patch: np.ndarray,
    slice_idx: int,
    y: int,
    x: int,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save patches with low SSIM to file for visual inspection.

    Parameters:
    - gt_patch (np.ndarray): Ground truth image patch.
    - pred_patch (np.ndarray): Predicted image patch.
    - slice_idx (int): Index of the current slice.
    - y (int): y-coordinate of the top-left corner of the patch.
    - x (int): x-coordinate of the top-left corner of the patch.
    - output_dir (Path): Directory to save the output images.
    - logger (logging.Logger): Logger instance for logging messages.
    """

    assert gt_patch.shape == pred_patch.shape, "Ground truth and prediction patches must have the same shape."
    assert gt_patch.ndim == 2, "Input patches must be 2D."

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_patch, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(pred_patch, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Save the figure
    patch_output_path = output_dir / f"low_ssim_patch_slice{slice_idx}_y{y}_x{x}.png"
    fig.savefig(patch_output_path)
    plt.close(fig)

    logger.info(f"Saved a pair of patches with low SSIM to {patch_output_path}")