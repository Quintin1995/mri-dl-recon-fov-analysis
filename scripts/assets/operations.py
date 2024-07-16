import numpy as np
import SimpleITK as sitk
import logging
import random

from multiprocessing import Pool
from functools import partial
from skimage.metrics import structural_similarity
from typing import Tuple, List
from pathlib import Path


def extract_label_patches(
    multi_label: np.ndarray,
    label: int, 
    patch_size: Tuple[int, int],
    max_attempts: int = 500,
    threshold: float = 0.9,
    seed: int = 42,
    logger: logging.Logger = None
) -> List[Tuple[int, int, int, int, int]]:
    """
    Get random 2D patches of the specified size from slices in the 3D image where at least a given percentage of values are equal to the given label.
    
    Parameters:
    `multi_label`: The 3D segmentation image as a NumPy array.
    `label` : The label to search for in the segmentation image.
    `patch_size`: The size of the patch to extract (height, width).
    `max_attempts` : Maximum number of attempts to find a suitable patch per slice.
    `threshold` : The minimum percentage of the patch that must be the given label.
    `seed` : Random seed for reproducibility.
    `logger` : Logger instance for logging information.

    Returns:
    A list of tuples with the coordinates of the extracted patches (y_min, y_max, x_min, x_max, z).
    
    Raises:
    ValueError: If no patch with the specified label is found within the maximum number of attempts.
    """
    np.random.seed(seed)  # Ensure reproducibility
    assert multi_label.ndim == 3, "Input must be a 3D image."

    patch_half_size = (patch_size[0] // 2, patch_size[1] // 2)
    
    # Identify slices that contain the label
    slices_with_label = [z for z in range(multi_label.shape[0]) if np.any(multi_label[z] == label)]
    
    if not slices_with_label:
        raise ValueError(f"No slices with label {label} found in the image.")
    
    successful_patches = []
    for z in slices_with_label:
        for attempt in range(max_attempts):
            label_coords = np.argwhere(multi_label[z] == label)
            
            if len(label_coords) == 0:
                if logger:
                    logger.info(f"No valid label found in slice {z} at attempt {attempt + 1}.")
                continue  # Try another slice if no valid label is found
            
            # Select a random point from the label coordinates
            y, x = label_coords[np.random.randint(len(label_coords))]
            
            # Calculate the patch bounds
            y_min = max(0, y - patch_half_size[0])
            y_max = min(multi_label[z].shape[0], y + patch_half_size[0])
            x_min = max(0, x - patch_half_size[1])
            x_max = min(multi_label[z].shape[1], x + patch_half_size[1])
            
            # Check if the patch meets the threshold requirement
            if np.mean(multi_label[z, y_min:y_max, x_min:x_max] == label) >= threshold:
                successful_patches.append((y_min, y_max, x_min, x_max, z))
                if logger:
                    logger.info(f"\t\t\t\tFound a valid patch in slice {z} at attempt {attempt + 1}.")
                break  # Found a valid patch, move to the next slice
        
        if len(successful_patches) == 0 and logger:
            logger.info(f"Failed to find a valid patch in slice {z} after {max_attempts} attempts.")
    
    if len(successful_patches) == 0:
        if logger:
            logger.info(f"No valid patch found with label {label} in any of the slices.")
        raise ValueError(f"No patch found with label {label} in any of the slices after {max_attempts} attempts.")

    return successful_patches


def percentile_clipping(image: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    """
    Clip the intensities of the image to the specified percentiles.

    Parameters:
    `image`: The image to clip. Dims: (Z, H, W)
    `lower_percentile`: The lower percentile for clipping.
    `upper_percentile`: The upper percentile for clipping.

    Returns:
    `image`: The clipped image.
    """

    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    image = np.clip(image, lower_bound, upper_bound)
    return image


def clip_image_sitk(image: sitk.Image, lower_percentile: float, upper_percentile: float) -> sitk.Image:
    img_array = sitk.GetArrayFromImage(image)
    lower_bound = np.percentile(img_array, lower_percentile)
    upper_bound = np.percentile(img_array, upper_percentile)
    clipped_array = np.clip(img_array, lower_bound, upper_bound)
    clipped_image = sitk.GetImageFromArray(clipped_array)
    clipped_image.CopyInformation(image)  # Preserve the original image metadata
    return clipped_image


def extract_2d_patch(img: np.ndarray, z: int, y_min: int, y_max: int, x_min: int, x_max: int) -> np.ndarray:
    """
    Extract a 2D patch from a 3D image. The patch is defined by the start and end indices in the y and x dimensions.

    Parameters:
    `img`: The 3D image. Dims: (Z, H, W)
    `y_min`: The starting index in the y dimension.
    `y_max`: The ending index in the y dimension.
    `x_min`: The starting index in the x dimension.
    `x_max`: The ending index in the x dimension.
    `z`: The slice index.

    Returns:
    `patch`: The extracted 2D patch. Dims: (H, W)
    """
    return img[z, y_min:y_max, x_min:x_max]


def select_random_nonzero_region(seg: np.ndarray, rectangle_size: Tuple[int, int], seed: int = 42) -> Tuple[int, int, int, int]:
    """
    Select a random non-zero region from the segmentation mask. The region is defined by
    the rectangle size. The function will keep selecting random regions until a region
    with no non-zero values is found.

    Parameters:
    `seg`: The segmentation mask. Dims: (H, W)
    `rectangle_size`: The size of the rectangle to select. (H, W)
    `seed`: The seed for the random number generator.

    Returns:
    `x_start`: The starting index in the x dimension.
    `x_end`: The ending index in the x dimension.
    `y_start`: The starting index in the y dimension.
    `y_end`: The ending index in the y dimension.
    """

    non_zero_coords = np.argwhere(seg)
    print(f"Number of non-zero coordinates found: {len(non_zero_coords)}")
    print(f"rectangle_size: {rectangle_size}")

    iteration_count = 0
    while True:
        iteration_count += 1
        if iteration_count % 1000 == 0:
            print(f"Iteration: {iteration_count}")
            
        random_idx = random.choice(non_zero_coords)
        x, y = random_idx[0], random_idx[1]
        
        x_start = max(0, x - rectangle_size[0] // 2)
        y_start = max(0, y - rectangle_size[1] // 2)
        x_end = min(seg.shape[0], x_start + rectangle_size[0])
        y_end = min(seg.shape[1], y_start + rectangle_size[1])
        
        print(f"Selected coordinates: x={x}, y={y}")
        print(f"Rectangle coordinates: x_start={x_start}, x_end={x_end}, y_start={y_start}, y_end={y_end}")
        
        if np.any(seg[x_start:x_end, y_start:y_end] != 0):
            print(f"Found non-zero region at iteration {iteration_count}")
            return x_start, x_end, y_start, y_end
        else:
            print(f"No non-zero region found at iteration {iteration_count}")


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
    img = sitk.ReadImage(str(nifti_path))
    return sitk.GetArrayFromImage(img)


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


def compute_error_map(gt: np.ndarray, pred: np.ndarray, scaling: float = 1.0, clip_range: tuple = None) -> np.ndarray:
    """
    Compute the error map between the ground truth and the predicted image, optionally scaling the result.

    Parameters:
        `gt`: The ground truth image.
        `pred`: The predicted image.
        `scaling`: Scaling factor to amplify the error map. Default is 1.0.
        `clip_range`: Tuple (min, max) to clip the error map values. Default is None (no clipping).

    Returns:
        error_map: The computed error map.
    """
    assert gt.shape == pred.shape, "Shape mismatch between ground truth and predicted image."
    
    error_map = np.abs(gt - pred)
    error_map *= scaling
    
    if clip_range is not None:
        error_map = np.clip(error_map, clip_range[0], clip_range[1])
    
    return error_map
