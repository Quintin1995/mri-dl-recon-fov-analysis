import SimpleITK as sitk
import numpy as np
from pathlib import Path
import random
from typing import Tuple, Dict


def load_image_as_array(image_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(image_path))
    return sitk.GetArrayFromImage(img)


def get_random_patch(image: np.ndarray, label: int, patch_size: Tuple[int, int], max_attempts: int = 500, threshold: float = 0.9) -> Tuple[np.ndarray, int]:
    """
    Get a random 2D patch of the specified size from a random slice in the 3D image where at least a given percentage of values are equal to the given label.
    
    Parameters:
    image (np.ndarray): The 3D segmentation image as a NumPy array.
    label (int): The label to search for in the segmentation image.
    patch_size (Tuple[int, int]): The size of the patch to extract.
    max_attempts (int): Maximum number of slices to try before giving up.
    threshold (float): The minimum percentage of the patch that must be the given label.
    
    Returns:
    Tuple[np.ndarray, int]: The extracted patch and the slice index.
    
    Raises:
    ValueError: If no slice with the specified label is found within the maximum number of attempts.
    """
    patch_half_size = (patch_size[0] // 2, patch_size[1] // 2)
    
    # Identify slices that contain the label
    slices_with_label = [z for z in range(image.shape[0]) if np.any(image[z, :, :] == label)]
    
    if not slices_with_label:
        raise ValueError(f"No slices with label {label} found in the image.")
    
    attempts = 0
    while attempts < max_attempts:
        # Select a random slice from those that contain the label
        z = random.choice(slices_with_label)
        image_slice = image[z, :, :]
        
        # Find all coordinates of the given label in the slice
        label_coords = np.argwhere(image_slice == label)
        
        if len(label_coords) == 0:
            attempts += 1
            continue  # Try another slice if no valid label is found
        
        # Select a random point from the label coordinates
        random_point = label_coords[random.randint(0, len(label_coords) - 1)]
        y, x = random_point
        
        # Calculate the patch bounds
        y_min = max(0, y - patch_half_size[0])
        y_max = min(image_slice.shape[0], y + patch_half_size[0])
        x_min = max(0, x - patch_half_size[1])
        x_max = min(image_slice.shape[1], x + patch_half_size[1])
        
        # Extract the patch
        patch = image_slice[y_min:y_max, x_min:x_max]
        
        # Check if the patch meets the threshold requirement
        if np.mean(patch == label) >= threshold:
            return patch, z

        attempts += 1

    raise ValueError(f"No patch found with label {label} in any of the slices after {max_attempts} attempts.")


def get_cfg() -> Dict[str, any]:
    """
    Return a configuration dictionary.
    
    Returns:
    Dict[str, any]: Configuration parameters.
    """
    return {
        "patch_sizes":          (55, 55),
        "labels_total_mr":      {'prostate': 17, 'femure_left': 34},
        "ml_total_mr":          Path("/scratch/hb-pca-rad/projects/03_reader_set_v2/segs/0088_ANON9892116_mlseg_total_mr.nii.gz"),
        "labels_tissue_types":  {'subcutaneous_fat': 1, 'skeletal_muscle': 3},
        "ml_tissue_types":      Path("/scratch/hb-pca-rad/projects/03_reader_set_v2/segs/0088_ANON9892116_mlseg_tissue_types_mr.nii.gz"),
        "threshold":            0.8
    }


def main():
    cfg = get_cfg()
    
    patch_sizes = cfg["patch_sizes"]
    
    arr_total_mr     = load_image_as_array(cfg["ml_total_mr"])
    arr_tissue_types = load_image_as_array(cfg["ml_tissue_types"])

    for label_name, label_value in cfg["labels_total_mr"].items():
        patch, z = get_random_patch(arr_total_mr, label_value, patch_sizes, threshold=cfg["threshold"])
        print(f"Extracted patch for {label_name} with label {label_value}:")
        print(patch)
        
    for label_name, label_value in cfg["labels_tissue_types"].items():
        patch, z = get_random_patch(arr_tissue_types, label_value, patch_sizes, threshold=cfg["threshold"])
        print(f"Extracted patch for {label_name} with label {label_value}:")
        print(patch)


if __name__ == "__main__":
    main()
