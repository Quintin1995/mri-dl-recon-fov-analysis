import pandas as pd
import numpy as np
import SimpleITK as sitk
import logging

from pathlib import Path
from typing import Tuple, List, Dict

from assets.metrics import fastmri_ssim, fastmri_psnr, fastmri_nmse, blurriness_metric, hfen, rmse
from assets.visualization import save_slices_as_images, plot_all_iqms_vs_accs_vs_fovs_boxplot
from assets.visualization import plot_all_iqms_vs_accs_vs_fovs_violinplot
from assets.writers import write_patches_as_png
from assets.operations import calculate_bounding_box, resample_to_reference, extract_sub_volume_with_padding
from assets.operations import load_seg_from_dcm_like, load_nifti_as_array
from assets.operations import generate_ssim_map_3d_parallel
from assets.operations import select_random_nonzero_region, extract_sub_volume
from assets.util import setup_logger, summarize_dataframe


# Define the mapping between string names and actual functions
IQM_FUNCTIONS = {
    'ssim': fastmri_ssim,
    'psnr': fastmri_psnr,
    'nmse': fastmri_nmse,
    'blurriness': blurriness_metric,
    'hfen': hfen,
    'rmse': rmse,
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
    - fov: The field of view (abfov, prfov, lsfov)
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
    return df


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
    decimals: int = 3,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process the lesion FOV. This includes:
    1. Extracting the sub-volume around the lesion
    2. Saving the slices as images
    3. Calculating the IQMs on each slice
    4. dding the IQMs to the DataFrame

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
    seg_bb    = extract_sub_volume_with_padding(seg, min_coords, max_coords, padding=PADDING)
    recon_bb  = extract_sub_volume_with_padding(recon, min_coords, max_coords, padding=PADDING)
    target_bb = extract_sub_volume_with_padding(target, min_coords, max_coords, padding=PADDING)

    if DO_SAVE_LESION_SEGS:
        save_slices_as_images(
            seg_bb       = seg_bb,
            recon_bb     = recon_bb,
            target_bb    = target_bb,
            pat_dir      = pat_dir,
            output_dir   = pat_dir / "lesion_bbs",
            acceleration = acc,
            lesion_num   = seg_idx+1,
            logger       = logger,
        )

    # Each slice with a lesion IQM calculation and add to the dataframe
    for slice_idx in range(seg_bb.shape[0]):
        data = {
            'pat_id':       pat_dir.name,
            'acceleration': acc,
            'roi':          'lsfov',
        }
        
        for iqm in iqms:
            # Calculate IQMs dynamically
            iqm_func = IQM_FUNCTIONS[iqm]
            iqm_value = iqm_func(gt=target_bb[slice_idx], pred=recon_bb[slice_idx])
            data[iqm] = round(iqm_value, decimals)
        
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    return df, seg_bb


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
    data.update({'pat_id': pat_id, 'acceleration': acc, 'roi': roi})
    new_row = pd.DataFrame([data])
    return pd.concat([df, new_row], ignore_index=True)


def calculate_iqms_ref_region(recon_bb: np.ndarray, target_bb: np.ndarray, iqms: List[str], decimals: int) -> dict:
    """
    Calculate IQMs for a reference region.

    Parameters:
    `recon_bb`: The reconstruction sub-volume
    `target_bb`: The target sub-volume
    `iqms`: The list of IQMs to calculate
    `decimals`: The number of decimals to round the IQMs to
    
    Returns:
    `data`: A dictionary of calculated IQMs
    """
    data = {}
    for iqm in iqms:
        iqm_func = IQM_FUNCTIONS[iqm]
        iqm_value = iqm_func(gt=target_bb, pred=recon_bb)
        data[iqm] = round(iqm_value, decimals)
    return data


def process_ref_region(
    df: pd.DataFrame,
    recon: np.ndarray,
    target: np.ndarray,
    seg_fpath: Path,
    ref_nifti: sitk.Image,
    pat_dir: Path,
    acc: int,
    iqms: List[str],
    region_name: str,
    rectangle_size: Tuple[int, int],
    decimals: int = 3,
    seed: int = 42,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Process a reference region (e.g., fat, muscle, bone, prostate). This includes:
    1. Loading the segmentation
    2. Selecting a random region
    3. Generating a random rectangle around the region
    4. Extracting sub-volumes
    5. Calculating IQMs
    6. Adding the IQMs to the DataFrame

    Parameters:
    `df`: The DataFrame to which the IQMs will be added
    `recon`: The reconstruction volume
    `target`: The target volume
    `seg_fpath`: The path to the segmentation file
    `ref_nifti`: The reference NIfTI file
    `pat_dir`: The patient directory
    `acc`: The acceleration factor
    `iqms`: The list of IQMs to calculate
    `region_name`: The name of the reference region (e.g., fat, muscle, bone, prostate)
    `rectangle_size`: The size of the rectangle to select
    `decimals`: The number of decimals to round the IQMs to
    
    Returns:
    `df`: The updated DataFrame with the IQMs
    """
    logger.info(f"\t\t\tProcessing {region_name.upper()} FOV")

    seg = load_seg_from_dcm_like(seg_fpath=seg_fpath, ref_nifti=ref_nifti)
    
    while True:
        x_start, x_end, y_start, y_end = select_random_nonzero_region(seg, rectangle_size, seed)
        if x_end <= seg.shape[1] and y_end <= seg.shape[2]:
            break

    recon_bb = extract_sub_volume(recon, x_start, x_end, y_start, y_end)
    target_bb = extract_sub_volume(target, x_start, x_end, y_start, y_end)
    
    data = calculate_iqms_ref_region(recon_bb, target_bb, iqms, decimals)
    df = update_dataframe(df, data, pat_dir.name, acc, region_name)

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
    patients_dir: Path,
    include_list: list,
    accelerations: list,
    fovs: List[str],
    iqms: List[str],
    do_ssim_map: bool = False,
    decimals: int = 3,
    iqm_mode: str = '3d',
    rectangle_size: Tuple[int, int] = (64, 64),
    seed: int = 42,
    logger: logging.Logger = None,
    **cfg,
) -> pd.DataFrame:
    """
    Calculate the IQMs for all patients in the patients_dir. On three FOVs: abfov, prfov, and lsfov.
    We add the IQMs to the DataFrame and return it.

    Parameters:
    `df`: The DataFrame to which the IQMs will be added
    `patients_dir`: The directory where the patient directories are stored
    `include_list`: The list of strings to include in the patient directory names
    `accelerations`: The list of acceleration factors to process
    `fovs`: The list of FOVs to process
    `iqms`: The list of IQMs to calculate
    `do_ssim_map`: Whether to calculate and save the SSIM map
    `decimals`: The number of decimals to round the IQMs to
    `logger`: The logger instance
    
    Returns:
    `df`: The updated DataFrame with the IQMs
    """
    # For readability lets call abfov:fov1, prfov:fov2, lsfov:fov3, reference_fov(s):fov4

    pat_dirs = filter_patient_dirs(patients_dir, include_list, logger)
    for pat_idx, pat_dir in enumerate(pat_dirs):
        logger.info(f"Processing patient {pat_idx+1}/{len(pat_dirs)}: {pat_dir.name}")

        # Define a dictionary mapping FOVs to their file paths
        fov_files = {
            'abfov': f"/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg/1x/{pat_dir.name}/rss_target.nii.gz",   # Ground truth is in another dir.
            'prfov': pat_dir / f"{pat_dir.name}_rss_target_dcml.mha",
            'lsfov': [x for x in pat_dir.iterdir() if "roi" in x.name], # multiple roi files
            'fat': pat_dir / 'fat.nii.gz',
            'bone': pat_dir / 'bone.nii.gz',
            'muscle': pat_dir / 'muscle.nii.gz',
            'prostate': pat_dir / 'prostate.nii.gz'
        }

        loaded_fovs = {}
        for fov in fovs:
            if fov in fov_files:
                if fov == 'lsfov':
                    # Special handling for 'lsfov' as it has multiple ROI files
                    loaded_fovs[fov] = [load_nifti_as_array(fp) for fp in fov_files[fov]]
                else:
                    loaded_fovs[fov] = load_nifti_as_array(fov_files[fov])
        
        for acc in accelerations:
            logger.info(f"\tProcessing acceleration: {acc}")
            for fov in fovs:
                if fov == 'abfov':
                    acc_pat_dir = Path(f"/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg/{acc}x/{pat_dir.name}")
                    recon_fpath = [x for x in acc_pat_dir.iterdir() if f"vsharpnet_r{acc}_recon.nii.gz" in x.name.lower()][0]
                    recon       = load_nifti_as_array(recon_fpath)
                    df          = calc_iqm_and_add_to_df(df, recon, loaded_fovs['abfov'], pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)
                elif fov == 'prfov':
                    recon_fpath = [x for x in pat_dir.iterdir() if f"vsharp_r{acc}_recon_dcml" in x.name.lower()][0]
                    recon       = load_nifti_as_array(recon_fpath)
                    df          = calc_iqm_and_add_to_df(df, recon, loaded_fovs['prfov'], pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)
                elif fov == 'lsfov':
                    ref_nifti = sitk.ReadImage(str(fov_files['prfov']))  # sitk image for resampling
                    for seg_idx, seg_fpath in enumerate(fov_files[fov]):
                        df, _ = process_lesion_fov(df, seg_idx, recon, loaded_fovs['prfov'], seg_fpath, ref_nifti, pat_dir, acc, iqms, decimals, logger)
                elif fov in ['fat', 'bone', 'muscle', 'prostate']:
                    ref_nifti = sitk.ReadImage(str(fov_files['prfov']))
                    seg_fpath = fov_files[fov]
                    recon_fpath = [x for x in pat_dir.iterdir() if f"vsharp_r{acc}_recon_dcml" in x.name.lower()][0]
                    recon = load_nifti_as_array(recon_fpath)
                    target = load_nifti_as_array(seg_fpath)
                    df = process_ref_region(df, recon, target, seg_fpath, ref_nifti, pat_dir, acc, iqms, fov, rectangle_size, decimals, seed, logger)

            # Calculate an SSIM map of the reconstruction versus the target
            if do_ssim_map:
                ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
                calculate_and_save_ssim_map_3d(
                    target      = loaded_fovs['prfov'],
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
    if debug:       # lets add 'debug' reight before the file extension of the csv_out_fpath
        csv_out_fpath = csv_out_fpath.parent / (csv_out_fpath.stem + '_debug' + csv_out_fpath.suffix)

    if not csv_out_fpath.exists() or force_new_csv:
        df = init_empty_dataframe(iqms, logger)
        df = calc_iqms_on_all_patients(
            df     = df,
            iqms   = iqms,
            logger = logger,
            **cfg
        )
        df.to_csv(csv_out_fpath, index=False, sep=';')
        logger.info(f"Saved DataFrame to {csv_out_fpath}")
    else:
        df = pd.read_csv(csv_out_fpath, sep=';')
        logger.info(f"Loaded DataFrame from {csv_out_fpath}")
    return df


def make_iqms_plots(
    df: pd.DataFrame,
    fig_dir: Path,
    iqms: List[str],
    debug: bool,
    logger: logging.Logger = None,
    **cfg,
) -> None:
    
    debug_str = "debug" if debug else ""
    # for iqm in iqms:
    #     plot_iqm_vs_accs_scatter_trend(
    #         df         = df,
    #         metric     = iqm,
    #         save_path  = fig_dir / f"{iqm}_vs_accs{str_id}.png",
    #     )

    # SCATTER PLOT WITH TREND LINES
    # plot_all_iqms_vs_accs_scatter_trend(
    #     df         = df,
    #     metrics    = iqms,
    #     save_path  = fig_dir / f"all_iqms_vs_accs{str_id}.png",
    #     palette    = 'bright',
    # )
    
    # BOXPLOT - Plot all IQMs vs acceleration rates and FOVs using box plots
    plot_all_iqms_vs_accs_vs_fovs_boxplot(
        df = df,
        metrics = iqms,
        save_path = fig_dir / debug_str / "all_iqms_vs_accs_vs_fovs_boxplot.png",
        do_also_plot_individually = False,
        logger = logger,
        
    )

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
    return {
        "csv_out_fpath":      Path('data/final/iqms_vsharp_r1r3r6_v2.csv'),
        "csv_stats_out_fpath":Path('data/final/metrics_table_v1.csv'),
        "patients_dir":       Path('/scratch/hb-pca-rad/projects/03_reader_set_v2/'),
        "log_dir":            Path('logs'),
        "temp_dir":           Path('temp'),
        "fig_dir":            Path('figures'),
        'include_list_fpath': Path('lists/include_ids.lst'),        # List of patient_ids to include as Path
        'accelerations':      [3, 6],                               # Accelerations included for post-processing
        'iqms':               ['ssim', 'psnr', 'rmse', 'hfen'],     # Image quality metrics to calculate
        'iqm_mode':           '2d',                                 # The mode for calculating the IQMs. Options are: ['3d', '2d']. The iqm will either be calculated for a 2d image or 3d volume, where the 3d volume IQM is the average of the 2d IQMs for all slices.
        'decimals':           3,                                    # Number of decimals to round the IQMs to
        'do_ssim_map':        False,                                # Whether to calculate and save the SSIM map
        'fovs':               ['abfov', 'prfov', 'lsfov', 'fat'],   # FOVS options :['abfov','prfov','lsfov','fat','bone','muscle','prostate']
        'debug':              True,                                 # Run in debug mode
        'force_new_csv':      True,                                 # Overwerite existing CSV file,
        'seed':               42,                                   # Random seed for reproducibility
        'rectangle_size':     (90, 90),                           # Size of the rectangle to select for the reference FOV
    }

# Summary of lesion segmentations sizes
# (          slices           x           y
#  count  75.000000   75.000000   75.000000     # 75 lesions
#  mean    2.773333   86.133333   92.720000     # mean number of slices, x and y dimensions
#  std     1.656872   19.781464   22.912029     # standard deviation
#  min     1.000000   56.000000   53.000000
#  25%     2.000000   71.000000   78.000000
#  50%     2.000000   81.000000   91.000000     # median
#  75%     3.000000   98.000000  103.000000     # 75th percentile
#  max    11.000000  161.000000  190.000000,
#  slices     2.773333
#  x         86.133333
#  y         92.720000
#  dtype: float64,
#  slices     2.0
#  x         81.0
#  y         91.0
#  dtype: float64)
# So if we want to take approximately the same reference region size as the lesion than a 90x90 rectangle should be sufficient.


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
    
    if cfg['debug']:
        cfg['include_list'] = ['0053_ANON5517301', '0032_ANON7649583', '0120_ANON7275574']  # Random selection of patients for debugging
        cfg['include_list'] = ['0003_ANON5046358', '0006_ANON2379607', '0007_ANON1586301']  # have rois 
    
    PADDING             = 20
    DO_SAVE_LESION_SEGS = True
    main(cfg, logger)