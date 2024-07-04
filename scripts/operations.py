import numpy as np
import SimpleITK as sitk
import logging

from multiprocessing import Pool
from functools import partial
from skimage.metrics import structural_similarity
from typing import Tuple
from pathlib import Path


def load_seg_from_dcm_like(seg_fpath: Path, ref_nifti: sitk.Image) -> tuple:
    """
    Load the segmentation from a .dcm file.

    Parameters:
    `seg_fpath`: The path to the .dcm file.
    `ref_nifti`: The reference NIfTI file for resampling.

    Returns:
    `seg`: The segmentation as a NumPy array.
    """
    seg = sitk.ReadImage(str(seg_fpath))
    seg = resample_to_reference(image=seg, ref_img=ref_nifti)
    seg = sitk.GetArrayFromImage(seg)
    return seg


def load_nifti_as_array(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI file as a NumPy array.
    
    Parameters:
    `nifti_path`: The path to the NIfTI file.
    
    Returns:
    `img`: The NIfTI file as a NumPy array.
    """
    img = sitk.ReadImage(str(nifti_path))
    img = sitk.GetArrayFromImage(img)
    return img


def calculate_bounding_box(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the bounding box around the lesion.
    """
    # Find the coordinates where ROI has non-zero values
    non_zero_coords = np.argwhere(roi)
    
    # Calculate the bounding box
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    
    return min_coords, max_coords


def resample_to_reference(
    image: sitk.Image, 
    ref_img: sitk.Image, 
    interpolator = sitk.sitkNearestNeighbor, 
    default_pixel_value: float = 0
) -> sitk.Image:
    """
    Automatically aligns, resamples, and crops an SITK image to a
    reference image. Can be either from .mha or .nii.gz files.

    Parameters:
    `image`: The moving image to align
    `ref_img`: The reference image with desired spacing, crop size, etc.
    `interpolator`: SITK interpolator for resampling
    `default_pixel_value`: Pixel value for voxels outside of original image
    """
    resampled_img = sitk.Resample(image, ref_img, 
            sitk.Transform(), 
            interpolator, default_pixel_value, 
            ref_img.GetPixelID())
    return resampled_img


def extract_sub_volume_with_padding(
    image: np.ndarray,
    min_coords: np.ndarray,
    max_coords: np.ndarray,
    padding: int,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Extract the sub-volume from the image with optional padding around the bounding box.
    The padding will be reduced if it exceeds the image dimensions.
    Padding is not applied in the slice (first) dimension.
    
    Parameters:
    `image`: The image from which to extract the sub-volume
    `min_coords`: The minimum coordinates of the bounding box
    `max_coords`: The maximum coordinates of the bounding box
    `padding`: The padding to apply around the bounding box
    
    Returns:
    `sub_volume`: The sub-volume with padding
    """
    # Initialize padded min and max coordinates
    padded_min_coords = min_coords.copy()
    padded_max_coords = max_coords.copy()
    
    # Apply padding to x and y dimensions (1 and 2), not slice (0)
    for dim in [1, 2]:
        # Reduce padding if it goes out of the image dimensions
        while True:
            if padded_min_coords[dim] - padding < 0 or padded_max_coords[dim] + padding >= image.shape[dim]:
                padding = max(padding // 2, 0)  # Halve the padding, minimum is 0
            else:
                break  # Exit the loop if padding is within limits
        
        # Apply padding
        padded_min_coords[dim] -= padding
        padded_max_coords[dim] += padding

    # Extract the sub-volume with padding
    sub_volume = image[padded_min_coords[0]:padded_max_coords[0]+1, 
                       padded_min_coords[1]:padded_max_coords[1]+1, 
                       padded_min_coords[2]:padded_max_coords[2]+1]
    
    if logger:
        logger.info(f"\t\t\tExtracted sub-volume with padding: {sub_volume.shape}")

    return sub_volume


def generate_ssim_map_3d_parallel(
    target: np.ndarray, 
    pred: np.ndarray, 
    window_size: int, 
    stride: int, 
    num_workers: int, 
    logger: logging.Logger
) -> np.ndarray:
    """
    Generate an SSIM map for a 3D volume in parallel across multiple slices.

    Parameters:
        target (np.ndarray): Ground truth 3D volume.
        pred (np.ndarray): Predicted 3D volume.
        window_size (int): The size of the window for SSIM calculation.
        stride (int): The stride of the sliding window.
        num_workers (int): The number of worker processes to use.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        np.ndarray: The SSIM map for the 3D volume.
    """
    logger.info("\t\t\tStarting parallel SSIM map generation.")
    ssim_map = np.zeros_like(target)

    with Pool(processes=num_workers) as pool:
        func = partial(calculate_ssim_for_slice, target_volume=target, pred_volume=pred, window_size=window_size, stride=stride)
        results = pool.map(func, range(target.shape[0]))

    for slice_index, ssim_map_slice in results:
        ssim_map[slice_index, :, :] = ssim_map_slice

    logger.info("\t\t\tCompleted parallel SSIM map generation.")
    return ssim_map


def calculate_ssim_for_slice(
    slice_index: int, 
    target_volume: np.ndarray, 
    pred_volume: np.ndarray, 
    window_size: int, 
    stride: int
) -> Tuple[int, np.ndarray]:
    """
    Calculate the SSIM for a single slice of the 3D volume.

    Parameters:
        slice_index (int): Index of the slice in the volume.
        target_volume (np.ndarray): Ground truth volume.
        pred_volume (np.ndarray): Predicted volume.
        window_size (int): Size of the window for SSIM calculation.
        stride (int): Stride of the sliding window.

    Returns:
        Tuple[int, np.ndarray]: Index of the slice and its SSIM map.
    """
    height, width = target_volume.shape[1], target_volume.shape[2]
    ssim_map_slice = np.zeros((height, width))

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            gt_patch = target_volume[slice_index, y:y + window_size, x:x + window_size]
            pred_patch = pred_volume[slice_index, y:y + window_size, x:x + window_size]

            # Compute SSIM for the current patch. The data_range is the maximum possible value of the image.
            ssim_value = structural_similarity(gt_patch, pred_patch, data_range=target_volume.max() - target_volume.min())

            # Fill the SSIM map for the current window location
            ssim_map_slice[y:y + window_size, x:x + window_size] = ssim_value

    return slice_index, ssim_map_slice