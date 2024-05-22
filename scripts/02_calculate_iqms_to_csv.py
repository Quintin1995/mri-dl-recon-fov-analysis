import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from metrics import fastmri_ssim_qvl, fastmri_psnr_qvl, fastmri_nmse_qvl, blurriness_metric, hfen
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from typing import Tuple, List
from scipy import stats
from skimage.metrics import structural_similarity


def setup_logger(log_dir: Path, use_time: bool = True, part_fname: str = None) -> logging.Logger:
    """
    Configure logging to both console and file.
    This function sets up logging based on the specified logging directory.
    It creates a log file named with the current timestamp and directs log 
    messages to both the console and the log file.
    Parameters:
    - log_dir (Path): Directory where the log file will be stored.
    Returns:
    - logging.Logger: Configured logger instance.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if use_time: 
        log_file = log_dir / f"log_{current_time}.log"
    elif part_fname is not None and use_time: 
        log_file = log_dir / f"log_{part_fname}_{current_time}.log"
    elif part_fname is not None and not use_time:
        log_file = log_dir / f"log_{part_fname}.log"
    else:
        log_file = log_dir / "log.log"

    l = logging.getLogger()
    l.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    l.addHandler(console_handler)

    return l


def filter_patient_dirs(rootdir: Path, include_list: list, logger: logging.Logger = None, do_sort: bool = True) -> list:
    """
    Get the patient directories that are relevant for the analysis.
    
    Parameters:
    - rootdir (Path): The root directory where the patient directories are stored.
    - include_list (list): The list of strings to include in the patient directory names.
    
    Returns:
    - patients_dirs (list): The list of patient directories that are relevant for the analysis.
    """
    
    patients_dirs = [x for x in rootdir.iterdir() if x.is_dir()]
    
    # only consider directories of the type '0053_ANON123456789'
    patients_dirs = [x for x in patients_dirs if x.name not in ['archive', 'exclusie']]
    patients_dirs = [x for x in patients_dirs if len(x.name.split('_')) == 2]
    
    # Filter out  patients not in the include_list
    if include_list is not None:
        patients_dirs = [x for x in patients_dirs if any([y in x.name for y in include_list])]
    
    if do_sort:     # sort them based ont he x.name.split[0] as integer values
        patients_dirs = sorted(patients_dirs, key=lambda x: int(x.name.split('_')[0]))
        
    if logger:
        logger.info(f"Found {len(patients_dirs)} patient directories in {rootdir}")

    return patients_dirs


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


def extract_sub_volume_with_padding(image: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray, padding: int) -> np.ndarray:
    """
    Extract the sub-volume from the image with optional padding around the bounding box.
    The padding will be reduced if it exceeds the image dimensions.
    Padding is not applied in the slice (first) dimension.
    Parameters:
    - image (np.ndarray): The image from which to extract the sub-volume
    - min_coords (np.ndarray): The minimum coordinates of the bounding box
    - max_coords (np.ndarray): The maximum coordinates of the bounding box
    - padding (int): The padding to apply around the bounding box
    Returns:
    - sub_volume (np.ndarray): The sub-volume with padding
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
    
    return sub_volume


def save_slices_as_images(
    seg_bb: np.ndarray,
    recon_bb: np.ndarray,
    target_bb: np.ndarray,
    pat_dir: Path,
    output_dir: str,
    acceleration: int,
    lesion_num: int,
    is_mirror: bool = False,
):
    """
    Save each slice from seg_bb, recon_bb, and target_bb as images side by side.
    
    Parameters:
        seg_bb (np.ndarray): The bounding box extracted from the segmentation.     # 3d array
        recon_bb (np.ndarray): The bounding box extracted from the reconstruction. # 3d array
        target_bb (np.ndarray): The bounding box extracted from the target.        # 3d array
        pat_dir (Path): The patient directory.
        output_dir (str): The directory where the images will be saved.
        acceleration (int): The acceleration factor.
        lesion_num (int): The lesion number.
        is_mirror (bool): Whether the bounding box is mirrored.

    Returns:
        None
    """
    assert recon_bb.shape == target_bb.shape == seg_bb.shape, "Mismatch in shape among bounding boxes"
    assert seg_bb.ndim == 3, "Bounding box should be a 3D array"
    assert recon_bb.ndim == 3, "Bounding box should be a 3D array"
    assert target_bb.ndim == 3, "Bounding box should be a 3D array"

    # make the dirs if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\t\t\tROI shape: {seg_bb.shape}, mean: {round(seg_bb.mean(), 4)}")
    logger.info(f"\t\t\tRecon shape: {recon_bb.shape}, mean: {round(recon_bb.mean(), 4)}")
    logger.info(f"\t\t\tTarget shape: {target_bb.shape}, mean: {round(target_bb.mean(), 4)}")

    # Number of slices should be the same for all bounding boxes
    
    for slice_idx in range(seg_bb.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plotting each slice side by side
        axes[0].imshow(seg_bb[slice_idx], cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Segmentation Slice {slice_idx}')
        
        axes[1].imshow(recon_bb[slice_idx], cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'Reconstruction Slice {slice_idx}')

        axes[2].imshow(target_bb[slice_idx], cmap='gray')
        axes[2].axis('off')
        axes[2].set_title(f'Target Slice {slice_idx}')
        
        # Save the figure
        if not is_mirror:
            fpath = os.path.join(output_dir, f"RIM_R{acceleration}_{pat_dir.name}_lesion{lesion_num}_slice{slice_idx+1}.png")
        else:
            fpath = os.path.join(output_dir, f"RIM_R{acceleration}_{pat_dir.name}_lesion{lesion_num}_slice{slice_idx+1}_mirrored.png")

        plt.savefig(fpath)
        plt.close(fig)
        logger.info(f"\t\t\tSaved slice {slice_idx+1}/{seg_bb.shape[0]} to {fpath}")


def calculate_image_quality_metrics(
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acceleration: int,
    iqms: List[str],
    fov: str = None,
    decimals: int = 3,
    logger: logging.Logger = None,
) -> dict:
    
    metrics = {}
    for iqm in iqms:
        if iqm == 'ssim':
            metrics['ssim'] = round(fastmri_ssim_qvl(gt=target, pred=recon), decimals)
        elif iqm == 'psnr':
            metrics['psnr'] = round(fastmri_psnr_qvl(gt=target, pred=recon), decimals)
        elif iqm == 'nmse':
            metrics['nmse'] = round(fastmri_nmse_qvl(gt=target, pred=recon), decimals)
        elif iqm == 'vofl':
            metrics['vofl'] = round(blurriness_metric(image=recon), decimals)
        elif iqm == 'rmse':
            metrics['rmse'] = round(np.sqrt(np.mean((recon - target) ** 2)), decimals)
        elif iqm == 'hfen':
            metrics['hfen'] = round(hfen(pred=recon, gt=target), decimals)
        else:
            raise ValueError(f"Invalid IQM: {iqm}")

    if logger is not None:
        log_msg = "\t\t" + ", ".join([f"{iqm.upper()}: {metrics[iqm]:.{decimals}f}" for iqm in iqms if iqm in metrics])
        logger.info(f"{log_msg}")

    data = {
        'pat_id':       pat_dir.name,
        'acceleration': acceleration,
    }
    data.update(metrics)  # Add the computed metrics to the data dictionary
    if fov is not None:
        data['roi'] = fov
    
    return data


def calculate_iqm_and_add_to_df(
    df: pd.DataFrame,
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acc: int,
    iqms: List[str],
    fov: str,
    decimals: int = 3,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Calculate the image quality metrics (IQMs) for the DLRecon images. The accelerated image are compared to the
    target images. We calculate the IQMs for the images and add them to the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to which the IQMs will be added
    - recon (np.ndarray): The reconstruction volume
    - target (np.ndarray): The target volume
    - pat_dir (Path): The patient directory
    - acc (int): The acceleration factor
    - iqms (List[str]): The list of IQMs to calculate
    - fov (str): The field of view (abfov, prfov, lsfov)
    - decimals (int): The number of decimals to round the IQMs to
    - logger (logging.Logger): The logger instance
    """	
    iqms_dict = calculate_image_quality_metrics(
        recon        = recon,
        target       = target,
        pat_dir      = pat_dir,
        acceleration = acc,
        iqms         = iqms,
        fov          = fov,
        decimals     = decimals,
        logger       = logger,
    )
    new_row = pd.DataFrame([iqms_dict])
    return pd.concat([df, new_row], ignore_index=True)


def load_seg_from_dcm_like(seg_fpath: Path, ref_nifti: sitk.Image, pat_dir: Path, acc: int) -> tuple:
    """
    Load the segmentation from a .dcm file.
    """
    # Load images from file as SimpleITK images
    seg = sitk.ReadImage(str(seg_fpath))

    # Resample first to the same size as the recon before obtaining an array
    seg = resample_to_reference(image=seg, ref_img=ref_nifti)

    # Convert to NumPy arrays
    seg = sitk.GetArrayFromImage(seg)

    return seg


def load_nifti_as_array(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI file as a NumPy array.
    
    Parameters:
    - nifti_path (Path): The path to the NIfTI file.
    
    Returns:
    - img (np.ndarray): The NIfTI file as a NumPy array.
    """
    img  = sitk.ReadImage(str(nifti_path))
    img  = sitk.GetArrayFromImage(img)
    return img


def process_lesion_fov(
    df: pd.DataFrame,
    seg_idx: int,
    recon: np.ndarray,
    target: np.ndarray,
    seg_fpath: Path,
    ref_nifti: sitk.Image,
    pat_dir: Path,
    acc: int,
    decimals: int = 3,
    is_mirror: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process the lesion FOV. This includes:
    - Extracting the sub-volume around the lesion
    - Saving the slices as images
    - Calculating the IQMs on each slice
    - Adding the IQMs to the DataFrame
    Parameters:
    - df (DataFrame): The DataFrame to which the IQMs will be added
    - seg_idx (int): The index of the segmentation file
    - recon (np.ndarray): The reconstruction volume
    - target (np.ndarray): The target volume
    - seg_fpath (Path): The path to the segmentation file
    - ref_nifti (sitk.Image): The reference NIfTI file
    - pat_dir (Path): The patient directory
    - acc (int): The acceleration factor
    - decimals (int): The number of decimals to round the IQMs to
    Returns:
    - df (DataFrame): The updated DataFrame with the IQMs
    - seg_bb (np.ndarray): The bounding box around the lesion
    """ 
    seg = load_seg_from_dcm_like(seg_fpath=seg_fpath, ref_nifti=ref_nifti, pat_dir=pat_dir, acc=acc)

    # Bounding box around the lesion
    min_coords, max_coords = calculate_bounding_box(roi=seg)
    seg_bb    = extract_sub_volume_with_padding(seg, min_coords, max_coords, padding=10)
    recon_bb  = extract_sub_volume_with_padding(recon, min_coords, max_coords, padding=10)
    target_bb = extract_sub_volume_with_padding(target, min_coords, max_coords, padding=10)

    save_slices_as_images(
        seg_bb       = seg_bb,
        recon_bb     = recon_bb,
        target_bb    = target_bb,
        pat_dir      = pat_dir,
        output_dir   = str(pat_dir / "lesion_bbs"),
        acceleration = acc,
        lesion_num   = seg_idx+1,
        is_mirror    = is_mirror,
    )

    # Each slice with a lesion IQM calculation and add to the dataframe
    for slice_idx in range(seg_bb.shape[0]):
        data = {
            'pat_id':       pat_dir.name,
            'acceleration': acc,
            'ssim':         round(fastmri_ssim_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'psnr':         round(fastmri_psnr_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'nmse':         round(fastmri_nmse_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'vofl':         round(blurriness_metric(image=recon_bb[slice_idx]), decimals),
            'roi':          f"lsfov" if not is_mirror else f"lsfov_mirrored",
        }
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    return df, seg_bb


def write_patches_to_file(
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


def calculate_and_save_ssim_map_3d(
    target: np.ndarray,
    recon: np.ndarray,
    output_dir: Path,
    patient_id: str,
    acc_factor: int,
    window_size: int = 11,
    stride: int = 4,
    logger: logging.Logger = None,
    ref_nifti: sitk.Image = None,
    metric: str = 'ssim',
):
    """
    Calculates an SSIM map for 3D volumes and saves it as a NIfTI file.

    Parameters:
    - target: The ground truth 3D volume.
    - reconstruction: The predicted 3D volume.
    - output_dir: Directory where the SSIM map NIfTI file will be saved.
    - patient_id: Identifier for the patient. In this case like 0001_ANON0123456
    - acceleration_factor: Acceleration factor (e.g., R2, R4, etc.) used in the file name.
    - window_size: The size of the window to compute SSIM, default is 11.
    - stride: The stride with which to apply the sliding window, default is 4.
    - logger: Logger for logging information.
    """
    if logger:
        logger.info(f"\t\t\tCalculating SSIM map with window size {window_size} and stride {stride}.")

    # Call to a function to generate the SSIM map
    # ssim_map = generate_ssim_map_3d(target, recon, window_size, stride, logger)
    ssim_map = generate_ssim_map_3d_parallel(target, recon, window_size, stride, num_workers=8, logger=logger)

    # Convert the SSIM map to a NIfTI image
    ssim_map_nifti = sitk.GetImageFromArray(ssim_map)

    # Define the output file path
    output_file_path = output_dir / f"RIM_R{acc_factor}_{patient_id}_dcml_{metric}_map_stride{stride}.nii.gz"

    # copy data from the ref nitfi to the ssim map nifti
    if ref_nifti:
        ssim_map_nifti.CopyInformation(ref_nifti)

    # Write the SSIM map NIfTI image to file
    sitk.WriteImage(ssim_map_nifti, str(output_file_path))
    
    if logger:
        logger.info(f"\t\t\tSaved SSIM map to {output_file_path}")


def calculate_iqms_on_all_patients(
    df: pd.DataFrame,
    patients_dir: Path,
    include_list: list,
    accelerations: list,
    fovs: List[str],
    iqms: List[str],
    do_ssim_map: bool = False,
    decimals: int = 3,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Calculate the IQMs for all patients in the patients_dir. On three FOVs: abfov, prfov, and lsfov.
    We add the IQMs to the DataFrame and return it.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the IQMs will be added
    - patients_dir (Path): The path to the directory containing the patient directories
    - include_list (list): The list of patients to include
    - accelerations (list): The list of accelerations to process
    - logger (logging.Logger): The logger instance
    - do_ssim_map (bool): Whether to calculate and save the SSIM map
    - decimals (int): The number of decimals to round the IQMs to
    - fov1 (str): The first field of view to process
    
    Returns:
    - df (pd.DataFrame): The updated DataFrame with the IQMs
    """
    pat_dirs = filter_patient_dirs(patients_dir, include_list, logger)
    for pat_idx, pat_dir in enumerate(pat_dirs):
        logger.info(f"Processing patient {pat_idx+1}/{len(pat_dirs)}: {pat_dir.name}")

        # Load the target, ROIs and then the reconstructions
        # roi_fpaths = [x for x in pat_dir.iterdir() if "roi" in x.name.lower()]
        target_fpath = [x for x in pat_dir.iterdir() if "rss_target_dcml" in x.name.lower()][0]
        target = load_nifti_as_array(target_fpath)
        for acc in accelerations:
            logger.info(f"\tProcessing acceleration: {acc}")
            
            for fov in fovs:
                recon_fpath = [x for x in pat_dir.iterdir() if f"vsharp_r{acc}_recon_dcml" in x.name.lower()][0]
                recon  = load_nifti_as_array(recon_fpath)
                df = calculate_iqm_and_add_to_df(df, recon, target, pat_dir, acc, iqms, fov, decimals, logger)
            
            # Calculate an SSIM map of the reconstruction versus the target
            if do_ssim_map:
                ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
                calculate_and_save_ssim_map_3d(
                    target      = target,
                    recon       = recon,
                    output_dir  = pat_dir / "metric_maps",
                    patient_id  = pat_dir.name,
                    acc_factor  = acc,
                    window_size = 10,
                    stride      = 10,
                    logger      = logger,
                    ref_nifti   = ref_nifti,
                    metric      = 'ssim',
                )
    return df


def plot_iqm_vs_accs_scatter_trend(
    df: pd.DataFrame,
    metric    = 'ssim',
    save_path = None,
    palette   = 'bright'
):
    # Dictionary for renaming ROIs
    rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}

    # Rename the 'roi' categories for better readability
    df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))

    # Count the number of datapoints for each trendline, grouped by 'roi_grouped' and 'acceleration'
    datapoints_count = df.groupby(['roi_grouped', 'acceleration']).size().reset_index(name='Number of Datapoints')

    # Create a dictionary to map ROI and acceleration to the number of datapoints
    datapoints_dict = {(roi, acc): count for roi, acc, count in zip(datapoints_count['roi_grouped'], datapoints_count['acceleration'], datapoints_count['Number of Datapoints'])}

    # Create a new legend label incorporating the number of datapoints
    df['legend_label'] = df.apply(lambda row: f"{row['roi_grouped']} (n={datapoints_dict[(row['roi_grouped'], row['acceleration'])]})", axis=1)

    # Create the scatter plot with trend lines
    plt.figure(figsize=(12, 8))
    color_palette = sns.color_palette(palette, n_colors=len(df['roi_grouped'].unique()))

    # Scatter plot
    sns.scatterplot(data=df, x='acceleration', y=metric, hue='legend_label', palette=color_palette, s=100)

    # Trend lines
    sns.lineplot(data=df, x='acceleration', y=metric, hue='roi_grouped', palette=color_palette, estimator=np.mean, legend=None)

    # Add titles and labels
    plt.title(f'DLRecon Image Quality: ({metric.upper()}) Degradation over Acceleration for Different Prostate ROIs')
    plt.xlabel('Acceleration')
    plt.ylabel(metric.upper())

    # Enhance the legend
    plt.legend(title='ROIs', title_fontsize='16', labelspacing=1.2, fontsize='12')
    plt.grid(True)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")


def plot_all_iqms_vs_accs_violin1(
    df: pd.DataFrame,
    metrics=['ssim', 'psnr', 'nmse', 'vofl'],
    save_path=None,
    palette='muted',  # Using a predefined palette that is visually appealing
    logger: logging.Logger = None,
) -> None:
    if 'roi' in df.columns:
        rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}
        df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))
        hue = 'roi_grouped'
    else:
        hue = None

    # df['acceleration'] = df['acceleration'].astype(int)  # Ensure 'acceleration' is integer

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.violinplot(data=df, x='acceleration', y=metric, hue=hue, ax=ax, palette=palette, inner='quartile')
        if hue:
            sns.stripplot(data=df, x='acceleration', y=metric, hue=hue, ax=ax, dodge=True, jitter=0.1, palette=palette, color='k', alpha=0.5, size=4)
            ax.legend().remove()
        else:
            sns.stripplot(data=df, x='acceleration', y=metric, ax=ax, color='k', jitter=0.1, size=4)

        ax.set_title(f"{metric.upper()}")
        ax.set_xlabel("R value")
        ax.set_ylabel(metric.upper())
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Dotted line, lighter width, lower alpha for subtlety


    if hue and idx == len(metrics) - 1:  # add legend only on the last subplot for clarity
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[:len(df[hue].unique())], labels[:len(df[hue].unique())], title='ROI Types', loc='upper right')

    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved figure to {save_path}")
            

def plot_all_iqms_vs_accs_scatter_trend(
    df: pd.DataFrame,
    metrics=['ssim', 'psnr', 'nmse', 'vofl'],
    save_path=None,
    palette='bright',
    logger: logging.Logger = None,
) -> None:
    """
    Create a 2x2 grid of subplots for the given quality metrics.

    Parameters:
    - df (DataFrame): The data to plot
    - metrics (list): The list of quality metrics to use
    - save_path (str): Optional path to save the generated plot
    """

    # Dictionary for renaming ROIs
    rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}
    df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))

    # Create the scatter plot with trend lines 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    color_palette = sns.color_palette(palette, n_colors=len(df['roi_grouped'].unique()))

    # Count the number of datapoints for each trendline, grouped by 'roi_grouped' and 'acceleration'
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.scatterplot(data=df, x='acceleration', y=metric, hue='roi_grouped', style='roi_grouped', s=100, ax=ax, palette=palette)
        
        # Adding trend lines for each ROI
        for jdx, roi_grouped in enumerate(df['roi_grouped'].unique()):
            roi_grouped_data = df[df['roi_grouped'] == roi_grouped]
            sns.regplot(data=roi_grouped_data, x='acceleration', y=metric, scatter=False, ax=ax, color=color_palette[jdx])
        
        # Add title and labels
        ax.set_title(f"IQM: {metric.upper()}")
        ax.set_xlabel("Acceleration Factor")
        ax.set_ylabel(metric.upper())
        ax.grid(True)
        ax.legend().remove()
    
    # Increase the legend size
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='ROI Types', fontsize='12')
    fig.suptitle("Image Quality Metrics Across Accelerations and FOVs", fontsize=16)
    
    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}") if logger else None


def init_empty_dataframe(iqms: List[str], logger: logging.Logger = None) -> pd.DataFrame:
    """
    Create an empty DataFrame with the specified columns and data types.
    """
    # Define columns and data types for the DataFrame
    types = {'pat_id': 'str', 'acceleration': 'int'}
    types.update({iqm: 'float64' for iqm in iqms})
    cols = ['pat_id', 'acceleration'] + iqms

    # Create DataFrame with specified columns and data types
    df = pd.DataFrame(columns=cols).astype(types)
    
    if logger:
        logger.info(f"Initialized an empty DataFrame with columns: {cols}")
    
    return df


def calc_or_load_iqms_df(csv_out_fpath: Path,
    force_new_csv: bool,
    iqms: List[str],
    logger: logging.Logger,
    **cfg,
    # patients_dir: Path,
    # include_list: List[str],
    # accelerations: List[int],
    # fovs: List[str],
    # do_ssim_map: bool,
    # decimals: int,
) -> pd.DataFrame:
    if not csv_out_fpath.exists() or force_new_csv:
        df = init_empty_dataframe(iqms, logger)
        df = calculate_iqms_on_all_patients(
            df              = df,
            logger          = logger,
            **cfg)
        df.to_csv(csv_out_fpath, index=False, sep=';')
        logger.info(f"Saved DataFrame to {csv_out_fpath}")
    else:
        df = pd.read_csv(csv_out_fpath, sep=';')
        logger.info(f"Loaded DataFrame from {csv_out_fpath}")
    return df


def plot_all_iqms_vs_accs_violin(
    df: pd.DataFrame,
    metrics=['ssim', 'psnr', 'nmse', 'vofl'],
    save_path=None,
    palette='muted',  # Using a predefined palette that is visually appealing
    logger: logging.Logger = None,
) -> None:
    if 'roi' in df.columns:
        rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}
        df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))
        hue = 'roi_grouped'
    else:
        hue = None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Changed to 1 row, 4 columns
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.violinplot(data=df, x='acceleration', y=metric, hue=hue, ax=ax, palette=palette, inner='quartile')
        if hue:
            sns.stripplot(data=df, x='acceleration', y=metric, hue=hue, ax=ax, dodge=True, jitter=0.1, palette=palette, color='k', alpha=0.5, size=4)
            ax.legend().remove()
        else:
            sns.stripplot(data=df, x='acceleration', y=metric, ax=ax, color='k', jitter=0.1, size=4)

        ax.set_title(f"{metric.upper()}")  # Y-labels are removed, titles are used instead
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Maintain grid settings
        axes[idx].set_ylabel('')
        axes[idx].set_xlabel('')

    # Set a global X-label
    fig.text(0.5, 0.04, 'R value', ha='center', va='center', fontsize=12)

    plt.tight_layout(pad=1.0)  # Adjust spacing to be relatively tight

    if hue and idx == len(metrics) - 1:  # Add legend only on the last subplot for clarity
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[:len(df[hue].unique())], labels[:len(df[hue].unique())], title='ROI Types', loc='upper right')

    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Saved figure to {save_path}")


def make_iqms_plots(df: pd.DataFrame, fig_dir: Path, iqms: List[str], debug: bool, logger: logging.Logger = None) -> None:
    str_id = "debug" if debug else ""
    # for iqm in iqms:
    #     plot_iqm_vs_accs_scatter_trend(
    #         df         = df,
    #         metric     = iqm,
    #         save_path  = fig_dir / f"{iqm}_vs_accs{str_id}.png",
    #     )
    # plot_all_iqms_vs_accs_scatter_trend(
    #     df         = df,
    #     metrics    = iqms,
    #     save_path  = fig_dir / f"all_iqms_vs_accs{str_id}.png",
    #     palette    = 'bright',
    # )
    plot_all_iqms_vs_accs_violin(
        df         = df,
        metrics    = iqms,
        save_path  = fig_dir / f"all_iqms_vs_accs_violin{str_id}_v2.png",
        logger     = logger,
    )
    

def make_table_mean_std(df: pd.DataFrame, logger: logging.Logger, iqms: List[str]) -> pd.DataFrame:
    """
    Computes the mean and standard deviation for each metric per acceleration value in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the metrics and acceleration values.
    logger (logging.Logger): The Logger object for logging messages.

    Returns:
    pd.DataFrame: A new DataFrame with the mean and standard deviation of each metric per acceleration value.
    """
    # Group the data by 'acceleration' and calculate the mean and std for each metric
    stats_df = df.drop(columns=[col for col in df.columns if col not in ['acceleration'] + iqms]).groupby('acceleration').agg(['mean', 'std'])

    # Simplify the multi-level column names
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    # Log the creation of the table
    if logger:
        logger.info("Table of statistics has been created.")
        print(stats_df.head(100))

    return stats_df


def bootstrap_ci(group: np.array, num_boots: int, ci: int) -> Tuple[float, float]:
    """
    Compute the confidence interval using bootstrapping.
    
    Parameters:
    - group (np.array): The array of values to bootstrap.
    - num_boots (int): The number of bootstrap samples to generate.
    - ci (int): The confidence interval percentage.

    Returns:
    - lower_bound (float): The lower bound of the confidence interval.
    - upper_bound (float): The upper bound of the confidence interval.
    """
    boot_means = []
    for _ in range(num_boots):
        boot_sample = np.random.choice(group, size=len(group), replace=True)
        boot_means.append(np.mean(boot_sample))
    lower_bound = np.percentile(boot_means, (100-ci)/2)
    upper_bound = np.percentile(boot_means, 100-(100-ci)/2)
    return lower_bound, upper_bound


def make_table_median_ci(df: pd.DataFrame, iqms: List[str], csv_stats_out_fpath: Path, logger: logging.Logger=None, decimals: int=2) -> pd.DataFrame:
    """
    Computes the median, standard deviation, and 95% confidence intervals for each metric per acceleration value in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the metrics and acceleration values.
    - iqms (List[str]): List of image quality metrics to analyze.
    - csv_stats_out_fpath (Path): The file path to output the statistics DataFrame.
    - logger (logging.Logger, optional): The Logger object for logging messages. Defaults to None.
    - decimals (int): Number of decimal places for rounding the results. Defaults to 2.

    Returns:
    - pd.DataFrame: A new DataFrame with the median, standard deviation, and 95% confidence intervals of each metric per acceleration value.
    """
    z = 1.96  # Correct Z-score for a 95% CI
    stats_df = pd.DataFrame()

    for metric in iqms:
        acc_grouped = df.groupby('acceleration')[metric]
        # stats_df[metric + '_median'] = acc_grouped.median().round(decimals)
        # stats_df[metric + '_std']    = acc_grouped.std().round(decimals)
        stats_df[metric + '_mean']   = acc_grouped.mean().round(decimals)
        
        ci95_lo, ci95_hi = [], []
        for name, acc_group in acc_grouped:
            clean_group = acc_group.dropna()  # Drop NaN values for accurate CI calculation
            mu = clean_group.mean()
            std = clean_group.std()
            n = len(clean_group)
            ci_low = mu - (z * (std / np.sqrt(n)))
            ci_high = mu + (z * (std / np.sqrt(n)))
            logger.info(f"Stats: {metric} - {name}: mu={mu}, std={std}, n={n}, CI=({ci_low}, {ci_high})")
            ci95_lo.append(round(ci_low, decimals))
            ci95_hi.append(round(ci_high, decimals))
        stats_df[metric + '_ci95_lo'] = ci95_lo
        stats_df[metric + '_ci95_hi'] = ci95_hi
    stats_df.to_csv(csv_stats_out_fpath, index=False, sep=';')
    if logger:
        logger.info(f"Saved DataFrame to {csv_stats_out_fpath}")
    return stats_df


def main(
    iqms: List[str]           = ['ssim', 'psnr'],
    fig_dir: Path             = None,
    logger: logging.Logger    = None,
    debug: bool               = False,
    csv_stats_out_fpath: Path = None,
    **cfg,
) -> None:
    """
    Here we calculate the image quality metrics (IQMs) for the DLRecon images. The accelerated image are compared to the
    target images. We calculate the IQMs for the images. We also calculate the SSIM map for the
    prostate FOV images. We save the results to a CSV file and optionally plot the metrics.
    
    Parameters:
    - patients_dir (Path): The directory containing the patient directories. Such as 0002_ANON0123456.
    - accelerations (List[int]): The list of accelerations to process.
    - iqms (List[str]): The image quality metrics to calculate. Options are: ['ssim', 'psnr', 'nmse', 'vofl']
    - csv_out_fpath (Path): The path to the CSV file where the results will be saved.
    - fig_dir (Path): The directory where the figures will be saved.
    - include_list (List[str]): The list of patients to include. Must inluce sequantial pat_ids
    - logger (logging.Logger): The logger instance.
    - force_new_csv (bool): Whether to overwrite the existing CSV file.
    - do_ssim_map (bool): Whether to calculate and save the SSIM map.
    """
    df = calc_or_load_iqms_df(**cfg, iqms=iqms, logger=logger)
    if False:
        make_iqms_plots(df=df, fig_dir=fig_dir, iqms=iqms, debug=debug, logger=logger)
    if False: 
        make_table_mean_std(df, logger, iqms)
    if True:
        make_table_median_ci(df=df, iqms=iqms, csv_stats_out_fpath=csv_stats_out_fpath, logger=logger)
    
    if False:
        # Group the data by 'roi' and 'acceleration' and calculate the mean and standard deviation of SSIM
        grouped_df = df.groupby(['roi', 'acceleration'])['ssim'].agg(['mean', 'std']).reset_index()

        logger.info(f"These are the mean and standard deviation of SSIM for each ROI and acceleration factor:\n")
        logger.info(grouped_df)

    
def get_configurations() -> dict:
    return {
        "csv_out_fpath":      Path('data/final/iqms_vsharp_r1r3r6_v2.csv'),
        "csv_stats_out_fpath":Path('data/final/metrics_table_v1.csv'),
        "patients_dir":       Path('/scratch/hb-pca-rad/projects/03_reader_set_v2/'),
        "log_dir":            Path('logs'),
        "temp_dir":           Path('temp'),
        "fig_dir":            Path('figures'),
        'include_list_fpath': Path('lists/include_ids.lst'),     # List of patient_ids to include.
        'accelerations':      [3, 6],                            # Accelerations included for post-processing.                            #[1, 3, 6],
        'iqms':               ['ssim', 'psnr', 'rmse', 'hfen'],  # Image quality metrics to calculate.                                    #['ssim', 'psnr', 'nmse', ],
        'decimals':           3,                                 # Number of decimals to round the IQMs to.
        'do_consider_rois':   True,                              # Whether to consider the different ROIs for the IQM calculation.
        'do_ssim_map':        False,                             # Whether to calculate and save the SSIM map.
        'fovs':               ['abfov', 'prfov', 'lsfov'],       # The field of views to process. Options are: ['abfov', 'prfov', 'lsfov']
        'debug':              True,                              # Whether to run in debug mode.
        'force_new_csv':      True,                              # Whether to overwrite the existing CSV file.
    }


if __name__ == "__main__":
    cfg = get_configurations()
    
    log_fname = 'calc_iqms_debug' if cfg['debug'] else 'calc_iqms'
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname=log_fname)
    
    # Load the inclusion list if specified in the configuration
    if cfg.get('include_list_fpath'):
        try:
            with open(cfg['include_list_fpath'], 'r') as f:
                cfg['include_list'] = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f"Inclusion list file not found: {cfg['include_list_fpath']}")
            exit(1)
    
    # Override include_list if in debug mode
    if cfg['debug']:
        cfg['include_list'] = ['0053_ANON5517301', '0032_ANON7649583', '0120_ANON7275574']
    
    main(logger=logger, **cfg)