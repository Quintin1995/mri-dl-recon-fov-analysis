import pandas as pd
import numpy as np
import SimpleITK as sitk
import logging

from pathlib import Path
from typing import Tuple, List, Dict

from assets.metrics import fastmri_ssim, fastmri_psnr, fastmri_nmse, blurriness_metric, hfen, rmse
from assets.visualization import save_slice_metrics_image, plot_all_iqms_vs_accs_vs_fovs_boxplot
from assets.visualization import plot_all_iqms_vs_accs_vs_fovs_violinplot
from assets.operations import calculate_bounding_box, extract_sub_volume_with_padding
from assets.operations import load_seg_from_dcm_like, load_nifti_as_array
from assets.operations import generate_ssim_map_3d_parallel
from assets.operations import extract_2d_patch
from assets.util import setup_logger, summarize_dataframe
from assets.operations import extract_label_patches


# Define the mapping between string names and actual functions
IQM_FUNCTIONS = {
    'ssim': fastmri_ssim,
    'psnr': fastmri_psnr,
    'nmse': fastmri_nmse,
    'blurriness': blurriness_metric,
    'hfen': hfen,
    'rmse': rmse,
}

REF_REGION_MAPPING = {
    'SFR': 'subcutaneous_fat',
    'MR': 'skeletal_muscle',
    'PR': 'prostate',
    'FR': 'femur_left',
}


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


def get_metric_list(iqm: str, recon: np.ndarray, target: np.ndarray, iqm_mode: str = '3d') -> List[float]:
    if iqm == 'ssim':
        return fastmri_ssim(gt=target, pred=recon, iqm_mode=iqm_mode)
    elif iqm == 'psnr':
        return fastmri_psnr(gt=target, pred=recon, iqm_mode=iqm_mode)
    elif iqm == 'nmse':
        return fastmri_nmse(gt=target, pred=recon, iqm_mode=iqm_mode)
    elif iqm == 'vofl':
        return blurriness_metric(image=recon, iqm_mode=iqm_mode)
    elif iqm == 'rmse':
        return rmse(gt=target, pred=recon, iqm_mode=iqm_mode)
    elif iqm == 'hfen':
        return hfen(gt=target, pred=recon, iqm_mode=iqm_mode)
    else:
        raise ValueError(f"Invalid IQM: {iqm}")


def calc_image_quality_metrics2d(
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acceleration: int,
    iqms: List[str],
    fov: str = None,
    decimals: int = 3,
    iqm_mode: str = '2d',
    logger: logging.Logger = None,
) -> List[Dict]:

    all_metrics = []    # list to store the metrics for each slice
    data = {}           # temporary storage for the metrics
    for iqm in iqms:
        data[iqm] = get_metric_list(iqm, recon, target, iqm_mode)

    for slice_idx in range(recon.shape[0]):
        metrics = {
            'pat_id': pat_dir.name,
            'acceleration': acceleration,
            'roi': fov,
            'slice': slice_idx,
        }
        for iqm in iqms:
            metrics[iqm] = round(data[iqm][slice_idx], decimals)
        all_metrics.append(metrics)

    if logger is not None:
        log_entries = []
        for metric in all_metrics:
            log_entries.append("\t\t" + " | ".join([f"{key.upper()}: {value:.{decimals}f}" for key, value in metric.items() if key in iqms]))

        logger.info("\n".join(log_entries))
    
    return all_metrics


def calc_iqm_and_add_to_df(
    df: pd.DataFrame,
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acc: int,
    iqms: List[str],
    fov: str,
    decimals: int = 3,
    iqm_mode: str = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Calculate the image quality metrics (IQMs) for the DLRecon images. The accelerated image are compared to the
    target images. We calculate the IQMs for the images and add them to the DataFrame.
    
    Parameters:
    - df: The DataFrame to which the IQMs will be added
    - recon: The reconstruction volume
    - target: The target volume
    - pat_dir: The patient directory
    - acc: The acceleration factor
    - iqms: The list of IQMs to calculate
    - fov: The field of view (FAV, CPV, TLV)
    - decimals: The number of decimals to round the IQMs to
    - logger: The logger instance
    
    Returns:
    - The updated DataFrame with the IQMs added
    """
    iqms_list = calc_image_quality_metrics2d(
        recon        = recon,
        target       = target,
        pat_dir      = pat_dir,
        acceleration = acc,
        iqms         = iqms,
        fov          = fov,
        decimals     = decimals,
        iqm_mode     = iqm_mode,
        logger       = None,
    )
    for iqms_dict in iqms_list:
        condition = (
            (df['pat_id'] == iqms_dict['pat_id']) &
            (df['acceleration'] == iqms_dict['acceleration']) &
            (df['slice'] == iqms_dict['slice']) &
            (df['roi'] == iqms_dict['roi'])
        )
        if df[condition].empty:
            df = pd.concat([df, pd.DataFrame([iqms_dict])], ignore_index=True)
        else:
            for key, value in iqms_dict.items():
                if key not in ['pat_id', 'acceleration', 'slice', 'roi']:
                    df.loc[condition, key] = value

    if logger:
        logger.info(f"\T\TAdded IQMs for {fov} FOV with acceleration {acc} to the DataFrame.")
    return df


def update_dataframe(df: pd.DataFrame, data: dict, pat_id: str, acc: int, roi: str) -> pd.DataFrame:
    """
    Update the DataFrame with calculated IQMs.

    Parameters:
    `df`: The DataFrame to update
    `data`: The dictionary of calculated IQMs
    `pat_id`: The patient ID
    `acc`: The acceleration factor
    `roi`: The region of interest (FOV)
    
    Returns:
    `df`: The updated DataFrame
    """
    # add the patient ID, acceleration factor, and ROI to the data dictionary
    # data.update({'pat_id': pat_id, 'acceleration': acc, 'roi': roi})

    # convert the data dictionary to a DataFrame and add it to the existing DataFrame
    new_row = pd.DataFrame([data])

    # concatenate the new row to the existing DataFrame and ignore the index
    return pd.concat([df, new_row], ignore_index=True)


def calc_all_iqms(data: dict, target_bb: np.ndarray, recon_bb: np.ndarray, slice_idx: int, iqms: List[str], iqm_mode: str, decimals: int) -> dict:
    """
    Calculate all specified IQMs (Image Quality Metrics) for the given data.

    Parameters:
    data: Dictionary to store the calculated IQM values.
    target_bb: Ground truth image array.
    recon_bb: Reconstructed image array.
    slice_idx: Index of the slice to calculate IQMs for (if 3D).
    iqms: List of IQMs to calculate.
    iqm_mode: Mode of IQM calculation ('2d' or '3d').
    decimals: Number of decimal places to round the IQM values to.

    Returns:
    dict: Updated dictionary with the calculated IQM values.
    """
    for iqm in iqms:
        iqm_func = IQM_FUNCTIONS[iqm]
        
        if target_bb.ndim == 2 and recon_bb.ndim == 2:
            # Directly compute IQM for 2D input
            iqm_value = iqm_func(gt=target_bb, pred=recon_bb, iqm_mode=iqm_mode)
        else:
            # Compute IQM for the specified slice in 3D input
            iqm_value = iqm_func(gt=target_bb[slice_idx], pred=recon_bb[slice_idx], iqm_mode=iqm_mode)
        
        if isinstance(iqm_value, list):
            data[iqm] = float(np.round(iqm_value[slice_idx] if target_bb.ndim == 3 else iqm_value, decimals))
        elif isinstance(iqm_value, (np.float32, np.float64, float)):
            data[iqm] = float(np.round(iqm_value, decimals))
        else:
            raise ValueError(f"Invalid IQM value: {iqm_value}")
    
    return data


def process_lesion_fov(
    df: pd.DataFrame,
    seg_idx: int,
    recon: np.ndarray,
    target: np.ndarray,
    seg_fpath: Path,
    ref_nifti: sitk.Image,
    pat_dir: Path,
    acc: int,
    iqms: List[str],
    iqm_mode: str,
    padding = 20,
    decimals: int = 3,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process the lesion FOV. This includes:
    1. Extracting the sub-volume around the lesion
    2. Saving the slices as images
    3. Calculating the IQMs on each slice
    4. Adding the IQMs to the DataFrame

    Parameters:
    `df`: The DataFrame to which the IQMs will be added
    `seg_idx`: The index of the segmentation file
    `recon`: The reconstruction volume
    `target`: The target volume
    `seg_fpath`: The path to the segmentation file
    `ref_nifti`Image): The reference NIfTI file
    `pat_dir`: The patient directory
    `acc`: The acceleration factor
    `decimals`: The number of decimals to round the IQMs to
    
    Returns:
    `df`: The updated DataFrame with the IQMs
    `seg_bb`: The bounding box around the lesion
    """
    seg = load_seg_from_dcm_like(seg_fpath=seg_fpath, ref_nifti=ref_nifti)

    # Bounding box around the lesion
    min_coords, max_coords = calculate_bounding_box(roi=seg)
    seg_bb    = extract_sub_volume_with_padding(seg, min_coords, max_coords, padding=padding)
    recon_bb  = extract_sub_volume_with_padding(recon, min_coords, max_coords, padding=padding)
    target_bb = extract_sub_volume_with_padding(target, min_coords, max_coords, padding=padding)

    output_dir = pat_dir / "lesion_bbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for slice_idx in range(seg_bb.shape[0]):
        data = {
            'pat_id':       pat_dir.name,
            'acceleration': acc,
            'roi':          'TLV',
            'slice':        slice_idx,
        }
        
        iqm_values_dict = calc_all_iqms(data, target_bb, recon_bb, slice_idx, iqms, iqm_mode, decimals)
        data.update(iqm_values_dict)

        if DO_SAVE_LESION_SEGS:
            save_slice_metrics_image(
                seg_bb       = seg_bb[slice_idx],
                recon_bb     = recon_bb[slice_idx],
                target_bb    = target_bb[slice_idx],
                iqm_values   = data,  # Using 'data' here for both purposes
                x_coords     = (int(min_coords[1]), int(max_coords[1])),
                y_coords     = (int(min_coords[2]), int(max_coords[2])),
                output_dir   = output_dir,
                acceleration = acc,
                iqms         = iqms,
                lesion_num   = seg_idx + 1,
                slice_idx    = slice_idx + 1,
                scaling      = 5,
                logger       = logger
            )
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    if logger:
        logger.info(f"\t\tAdded IQMs for TLV FOV with acceleration {acc} to the DataFrame.")

    return df, seg_bb


def process_ref_region(
    df: pd.DataFrame,                           # DataFrame to store the IQMs
    recon: np.ndarray,                          # Accelerated Reconstruction 3d
    target: np.ndarray,                         # Target reconstruction 3d
    ml_array: np.ndarray,                       # multi-label array
    pat_dir: Path,                              # Example: '0053_ANON123456789'   
    acc: int,                                   # Example: 3, 6
    iqm_mode: str,                              # Example: '2d', '3d'
    iqms: List[str],                            # Example: ['ssim', 'psnr', 'nmse', 'blurriness', 'hfen', 'rmse']
    region_name: str,                           # Example: 'SFR', 'MR', 'PR', 'FR'
    label: int = None,                          # Example: 1, 3, 17, 34
    max_attempts: int = 500,                    # max attempts to get a random patch
    label_threshold: float = 0.9,                     # Example 0.9 = 90% of the patch should be the label
    patch_size: Tuple[int, int] = (45, 45),     # Example: (64, 64)
    padding: int = 20,                          # padding around the lesion
    decimals: int = 3, 
    logger: logging.Logger = None,
) -> pd.DataFrame:
    
    logger.info(f"\t\t\tProcessing {region_name} FOV")

    # List of bounding boxes for the label patches in the multi-label array
    label_patches = extract_label_patches(
        multi_label  = ml_array,
        label        = label,
        patch_size   = (patch_size[0] + padding, patch_size[1] + padding),  # add padding to the patch size
        max_attempts = max_attempts,
        threshold    = label_threshold,
        logger       = logger,
    )
    if label_patches == None:
        logger.warning(f"\t\t\tNo label patches found for {region_name.upper()} FOV.")
        return df

    output_dir = pat_dir / "ref_region_bbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # loop slices for referece region with correct label and extract 2d patch and add IQMs to df.
    for y_min, y_max, x_min, x_max, slice_idx in label_patches:
        recon_bb  = extract_2d_patch(recon, slice_idx, y_min, y_max, x_min, x_max)
        target_bb = extract_2d_patch(target, slice_idx, y_min, y_max, x_min, x_max)
        data = {
            'pat_id':       pat_dir.name,
            'acceleration': acc,
            'roi':          region_name,
            'slice':        slice_idx,
        }

        data = calc_all_iqms(data, recon_bb, target_bb, None, iqms, iqm_mode, decimals)
        df = update_dataframe(df, data, pat_dir.name, acc, region_name)

        if DO_SAVE_REF_REGIONS:
            save_slice_metrics_image(
                recon_bb     = recon_bb,
                target_bb    = target_bb,
                iqm_values   = data,
                x_coords     = (x_min, x_max),
                y_coords     = (y_min, y_max),
                output_dir   = output_dir,
                acceleration = acc,
                iqms         = iqms,
                slice_idx    = slice_idx + 1,
                scaling      = 5,
                region_name  = region_name,
                logger       = logger
            )

    if logger:
        logger.info(f"\t\t\tProcessed {region_name.upper()} FOV with IQMs: {data}")

    return df


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


def calc_iqms_on_all_patients(
    df: pd.DataFrame,
    pat_dir: Path,
    accelerations: list,
    fovs: List[str],
    iqms: List[str],
    padding: int = 20,
    max_attempts: int = 500,
    label_threshold: float = 0.9,
    labels: Dict[str, int] = None,
    patch_size: Tuple[int, int] = (45, 45),
    decimals: int = 3,
    iqm_mode: str = '3d',
    logger: logging.Logger = None,
    **cfg,
) -> pd.DataFrame:
    """
    Calculate the image quality metrics (IQMs) for all patients and add them to the DataFrame.
    Details: 
    1. Load the FOVs for each patient
    2. Load the reconstructions and target images
    3. Calculate the IQMs for each FOV and acceleration
    4. Add the IQMs to the DataFrame

    Parameters:
    `df`: The DataFrame to which the IQMs will be added
    `pat_dirs`: The list of patient directories
    `accelerations`: The list of acceleration factors
    `fovs`: The list of field of views (FAV, CPV, TLV, PR, FR, SFR, MR)
    `iqms`: The list of image quality metrics (IQMs) to calculate
    `padding`: The padding around the lesion for the bounding box
    `max_attempts`: The maximum number of attempts to get a random patch
    `threshold`: The threshold for the label patch
    `labels`: The dictionary mapping the region names to the labels
    `patch_size`: The size of the patch to extract 
    `decimals`: The number of decimals to round the IQMs to
    `iqm_mode`: The mode of the IQM calculation (2D or 3D)
    `logger`: The logger instance

    Returns:
    `df`: The updated DataFrame with the IQMs added
    """
    FAV_DIR = Path("/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg")

    # Define a dictionary mapping FOVs to their file paths
    fov_files = {
        'FAV': FAV_DIR / '1x' / pat_dir.name / 'rss_target.nii.gz',
        'CPV': pat_dir / f"{pat_dir.name}_rss_target_dcml.mha",
        'TLV': [x for x in pat_dir.iterdir() if "roi" in x.name], # multiple roi files
        'PR': pat_dir.parent / 'segs' / f"{pat_dir.name}_mlseg_total_mr.nii.gz",
        'FR': pat_dir.parent / 'segs' / f"{pat_dir.name}_mlseg_total_mr.nii.gz",
        'SFR': pat_dir.parent / 'segs' / f"{pat_dir.name}_mlseg_tissue_types_mr.nii.gz",
        'MR': pat_dir.parent / 'segs' / f"{pat_dir.name}_mlseg_tissue_types_mr.nii.gz",
    }

    loaded_fovs = {} # Load the FOVs as NIfTI arrays and store them in a dictionary for easy access
    for fov in fovs:
        if fov in fov_files:
            if fov == 'TLV':
                # Special handling for 'TLV' as it has multiple ROI files
                loaded_fovs[fov] = [load_nifti_as_array(fp) for fp in fov_files[fov]]
            else:
                loaded_fovs[fov] = load_nifti_as_array(fov_files[fov])

    for acc in accelerations:
        logger.info(f"\tProcessing acceleration: {acc}")
        base_recon = load_nifti_as_array(pat_dir / f"{pat_dir.name}_VSharp_R{acc}_recon_dcml.mha")

        for fov in fovs:
            if fov == 'FAV':
                recon = load_nifti_as_array(FAV_DIR / f"{acc}x" / pat_dir.name / f"VSharpNet_R{acc}_recon.nii.gz")
                df = calc_iqm_and_add_to_df(df, recon, loaded_fovs['FAV'], pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)
            elif fov == 'CPV':
                df = calc_iqm_and_add_to_df(df, base_recon, loaded_fovs['CPV'], pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)
            elif fov == 'TLV':
                ref_nifti = sitk.ReadImage(str(fov_files['CPV']))      # sitk image for resampling
                for seg_idx, seg_fpath in enumerate(fov_files[fov]):
                    df, _ = process_lesion_fov(df, seg_idx, base_recon, loaded_fovs['CPV'], seg_fpath, ref_nifti, pat_dir, acc, iqms, iqm_mode, padding, decimals, logger)
            elif fov in ['SFR', 'FR', 'MR', 'PR']:
                multi_label = loaded_fovs[fov]
                df = process_ref_region(
                    df              = df,
                    recon           = base_recon,
                    target          = loaded_fovs['CPV'],
                    ml_array        = multi_label,
                    pat_dir         = pat_dir,
                    acc             = acc,
                    iqms            = iqms,
                    iqm_mode        = iqm_mode,
                    region_name     = fov,
                    label           = labels[REF_REGION_MAPPING[fov]],
                    max_attempts    = max_attempts,
                    label_threshold = label_threshold,
                    patch_size      = patch_size,
                    decimals        = decimals,
                    logger          = logger,
                )

        # Calculate an SSIM map of the reconstruction versus the target
        if DO_SSIM_MAP:
                ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
                calculate_and_save_ssim_map_3d(
                    target      = loaded_fovs['CPV'],
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


def init_empty_dataframe(iqms: List[str], logger: logging.Logger = None) -> pd.DataFrame:
    """
    Create an empty DataFrame with the specified columns and data types. 
    So that we can add the IQMs to the DataFrame later. First we add pat_id, acceleration, roi, slice columns.
    Then we add the IQMs as columns with float64 data type.

    Parameters:
    `iqms`: The list of image quality metrics (IQMs) to include in the DataFrame.
    `logger`: The logger instance for logging messages.

    Returns:
    `df`: An empty DataFrame with the specified columns and data types.
    """
    # Define columns and data types for the DataFrame
    types = {'pat_id': 'str', 'acceleration': 'float', 'roi': 'str', 'slice': 'int'}
    types.update({iqm: 'float64' for iqm in iqms})
    cols = ['pat_id', 'acceleration', 'roi', 'slice'] + iqms

    # Create DataFrame with specified columns and data types
    df = pd.DataFrame(columns=cols).astype(types)
    
    if logger:
        logger.info(f"Initialized an empty DataFrame with columns: {cols}")
    
    return df


def calc_or_load_iqms_df(
    csv_out_fpath: Path,
    force_new_csv: bool,
    iqms: List[str],
    logger: logging.Logger,
    debug: bool = False,
    **cfg,
) -> pd.DataFrame:
    '''
    Calculate the IQMs for all patients and save them to a CSV file. If the CSV file already exists, load it.

    Parameters:
    `csv_out_fpath`: The path to the CSV file where the IQMs will be saved.
    `force_new_csv`: Whether to force the calculation of the IQMs and overwrite the existing CSV file.
    `iqms`: The list of image quality metrics (IQMs) to calculate.
    `logger`: The logger instance for logging messages.
    `**cfg`: Additional keyword arguments.

    Returns:
    `df`: The DataFrame with the IQMs for all patients.
    '''
    if debug:
        csv_out_fpath = csv_out_fpath.parent / (csv_out_fpath.stem + '_debug' + csv_out_fpath.suffix)

    pat_dirs = filter_patient_dirs(cfg['patients_dir'], cfg['include_list'], logger)

    if not csv_out_fpath.exists() or force_new_csv:
        df = init_empty_dataframe(iqms, logger)
    else:
        df = pd.read_csv(csv_out_fpath, sep=';')
        logger.info(f"Loaded DataFrame from {csv_out_fpath}")

    processed_patients = set(df['pat_id'].unique()) if 'pat_id' in df.columns else set()

    for pat_dir in pat_dirs:
        if pat_dir.name in processed_patients:
            logger.info(f"Skipping IQM calculation for already processed patient: {pat_dir.name}. for file: {csv_out_fpath}")
            continue

        try:
            df = calc_iqms_on_all_patients(
                df=df,
                pat_dir=pat_dir,
                iqms=iqms,
                logger=logger,
                **cfg
            )
            df.to_csv(csv_out_fpath, index=False, sep=';')
            logger.info(f"Processed and saved data for patient: {pat_dir.name}")
        except Exception as e:
            logger.error(f"Error processing patient {pat_dir.name}: {e}")
            input(f"we have observed an error so we are not continuing for now... Press Enter to continue, then patient: {pat_dir.name} will be skipped.")
            continue

    return df


def make_iqms_plots(
    df: pd.DataFrame,
    fig_dir: Path,
    iqms: List[str],
    debug: bool,
    plot_iqms: List[str],
    logger: logging.Logger = None,
    **cfg,
) -> None:
    debug_str = "debug" if debug else ""

    plot_all_iqms_vs_accs_vs_fovs_boxplot(
        df = df,
        metrics = plot_iqms,
        save_path = fig_dir / debug_str / "all_iqms_vs_accs_vs_fovs_boxplot.png",
        legend_fig_idx = 2,
        do_also_plot_individually = False,
        logger = logger,
    )

    if False:
        plot_all_iqms_vs_accs_vs_fovs_violinplot(
            df = df,
            metrics = iqms,
            save_path = fig_dir / debug_str / "all_iqms_vs_accs_vs_fovs_violinplot.png",
            do_also_plot_individually = False,
            logger = logger,
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


def make_table_median_ci(
    df: pd.DataFrame,
    iqms: List[str],
    csv_stats_out_fpath: Path  = None,
    logger: logging.Logger     = None,
    decimals: int              = 2,
    **cfg,
) -> pd.DataFrame:
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


def calculate_or_load_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    return calc_or_load_iqms_df(logger=logger, **cfg)


def generate_plots(df: pd.DataFrame, cfg: dict, logger: logging.Logger) -> None:
    make_iqms_plots(df=df, logger=logger, **cfg)


def generate_tables(df: pd.DataFrame, cfg: dict, logger: logging.Logger) -> None:
    make_table_median_ci(df=df, logger=logger, **cfg)


def main(cfg: dict = None, logger: logging.Logger = None) -> None:
    if True:
        df = calculate_or_load_data(cfg, logger)
        generate_plots(df, cfg, logger)
    if False:
        print(summarize_dataframe(df))
        generate_tables(df, cfg, logger)
        pass


def get_configurations() -> dict:
    cfg = {
        "csv_out_fpath":      Path('data/final/iqms_vsharp_r1r3r6_with_ref_regions.csv'),                                             # Path to save the IQMs to 
        "csv_stats_out_fpath":Path('data/final/metrics_table_v1.csv'),                                                  # Path to save the statistics to as table
        "patients_dir":       Path('/scratch/hb-pca-rad/projects/03_reader_set_v2/'),                                   # Path to the directory with the patient directories data input dir
        # "patients_dir":       Path('/mnt/c/Users/Quintin/Documents/phd_local/03_datasets/03_umcg_nki_reader_set_v2'),
        "log_dir":            Path('logs'),
        "temp_dir":           Path('temp'),
        "fig_dir":            Path('figures'),
        'include_list_fpath': Path('lists/include_ids.lst'),                             # List of patient_ids to include as Path
        
        # Configuration options
        'debug':               False,              # Run in debug mode
        'seed':                42,                 # Random seed for reproducibility
        'force_new_csv':       False,              # Overwerite existing CSV file,
        'decimals':            3,                  # Number of decimals to round the IQMs to
        'do_save_lesion_segs': False,              # Save the lesion segmentations
        'do_save_ref_regions': False,              # Save the reference regions
        'do_ssim_map':         False,              # Calculate and save the SSIM map
        
        # IQM options
        'accelerations': [3, 6],                                # Accelerations included for post-processing
        'fovs':          ['FAV', 'CPV', 'TLV'],                 # FOVS options :['FAV','CPV','TLV']
        'ref_rois':      ['SFR', 'MR', 'PR', 'FR'],             # Reference regions to calculate IQMs for Options: ['SFR', 'MR', 'PR', 'FR']
        'iqms':          ['ssim', 'psnr', 'rmse', 'hfen'],      # Image quality metrics to calculate
        'iqm_mode':      '2d',                                  # The mode for calculating the IQMs. Options are: ['3d', '2d']. The iqm will either be calculated for a 2d image or 3d volume, where the 3d volume IQM is the average of the 2d IQMs for all slices.
        
        # Reference regions params and lesion FOV 
        "labels":              {'prostate': 17,
                                'femur_left': 34,
                                'subcutaneous_fat': 1,
                                'skeletal_muscle': 3
                                }, 
        'patch_size':          (45, 45),                                         # Size of the rectangle to select for the reference FOV
        "label_threshold":     0.9,                                              # Minimum percentage of the patch that must be the given label
        'padding':             20,                                               # Padding around lesion bounding box x and y direction

        # PLOTTING Options
        'plot_iqms': ['ssim', 'hfen', 'psnr'],  # Image quality metrics to plot
    }
    # Combine the FOVs and reference regions because we will process them all together
    cfg['fovs'] = cfg['fovs'] + cfg['ref_rois']
    for key, value in cfg.items():
        print(f"{key}: {value}")
    return cfg


if __name__ == "__main__":

    cfg = get_configurations()
    log_fname = 'calc_iqms_debug' if cfg['debug'] else 'calc_iqmsv2'
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname=log_fname)
    
    #### LAZY GLOBAL !!!!
    DO_SAVE_LESION_SEGS = cfg['do_save_lesion_segs']
    DO_SAVE_REF_REGIONS = cfg['do_save_ref_regions']
    DO_SSIM_MAP         = cfg['do_ssim_map']

    np.random.seed(cfg['seed'])  # Ensure reproducibility

    # Load the inclusion list if specified in the configuration
    if cfg.get('include_list_fpath'):
        try:
            with open(cfg['include_list_fpath'], 'r') as f:
                cfg['include_list'] = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f"Inclusion list file not found: {cfg['include_list_fpath']}")
            exit(1)
    
    if cfg['debug']:
        cfg['include_list'] = ['0053_ANON5517301', '0032_ANON7649583', '0120_ANON7275574']  # Random selection of patients for debugging
        cfg['include_list'] = ['0003_ANON5046358', '0053_ANON5517301', '0006_ANON2379607', '0007_ANON1586301']  # Nice examples with ROIs
        cfg['include_list'] = ['0062_ANON0319974', '0120_ANON7275574']

    main(cfg, logger)