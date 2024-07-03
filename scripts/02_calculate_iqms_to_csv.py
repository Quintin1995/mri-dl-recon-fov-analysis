import pandas as pd
import numpy as np
import SimpleITK as sitk
import logging

from pathlib import Path
from typing import Tuple, List, Dict

from metrics import fastmri_ssim, fastmri_psnr, fastmri_nmse, blurriness_metric, hfen, rmse
from visualization import save_slices_as_images, plot_iqm_vs_accs_scatter_trend, plot_all_iqms_vs_accs_scatter_trend, plot_all_iqms_vs_accs_vs_fovs_boxplot
from writers import write_patches_as_png
from operations import calculate_bounding_box, resample_to_reference, extract_sub_volume_with_padding
from operations import load_seg_from_dcm_like, load_nifti_as_array
from operations import generate_ssim_map_3d_parallel
from util import setup_logger, summarize_dataframe


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
        logger       = logger,
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
            'ssim':         round(fastmri_ssim(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'psnr':         round(fastmri_psnr(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'nmse':         round(fastmri_nmse(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'vofl':         round(blurriness_metric(image=recon_bb[slice_idx]), decimals),
            'roi':          f"lsfov" if not is_mirror else f"lsfov_mirrored",
        }
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    return df, seg_bb


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
    logger: logging.Logger = None,
    **cfg,
) -> pd.DataFrame:
    """
    Calculate the IQMs for all patients in the patients_dir. On three FOVs: abfov, prfov, and lsfov.
    We add the IQMs to the DataFrame and return it.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the IQMs will be added
    - patients_dir (Path): The directory where the patient directories are stored
    - include_list (list): The list of strings to include in the patient directory names
    - accelerations (list): The list of acceleration factors to process
    - fovs (List[str]): The list of FOVs to process
    - iqms (List[str]): The list of IQMs to calculate
    - do_ssim_map (bool): Whether to calculate and save the SSIM map
    - decimals (int): The number of decimals to round the IQMs to
    - logger (logging.Logger): The logger instance
    
    Returns:
    - df (pd.DataFrame): The updated DataFrame with the IQMs
    """

    # for readability lets call abfov:fov1, prfov:fov2, lsfov:fov3

    pat_dirs = filter_patient_dirs(patients_dir, include_list, logger)
    for pat_idx, pat_dir in enumerate(pat_dirs):
        logger.info(f"Processing patient {pat_idx+1}/{len(pat_dirs)}: {pat_dir.name}")

        # Load the target, ROIs and then the reconstructions
        # roi_fpaths = [x for x in pat_dir.iterdir() if "roi" in x.name.lower()]

        # Abdominal FOV1
        pat_dir_fov1 = Path(f"/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg/1x/{pat_dir.name}")
        target_fpath_fov1 = [x for x in pat_dir_fov1.iterdir() if "rss_target.nii.gz" in x.name.lower()][0]
        target_fov1 = load_nifti_as_array(target_fpath_fov1)
        
        # Prostate FOV2
        target_fpath_fov2 = [x for x in pat_dir.iterdir() if "rss_target_dcml" in x.name.lower()][0]
        target_fov2 = load_nifti_as_array(target_fpath_fov2)
        
        # Calc IQMs for each acceleration and FOV
        for acc in accelerations:
            logger.info(f"\tProcessing acceleration: {acc}")
            
            for fov in fovs:
                if fov == 'abfov':
                    acc_pat_dir = Path(f"/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg/{acc}x/{pat_dir.name}")
                    recon_fpath = [x for x in acc_pat_dir.iterdir() if f"vsharpnet_r{acc}_recon.nii.gz" in x.name.lower()][0]
                    recon       = load_nifti_as_array(recon_fpath)
                    df          = calc_iqm_and_add_to_df(df, recon, target_fov1, pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)

                elif fov == 'prfov':
                    recon_fpath = [x for x in pat_dir.iterdir() if f"vsharp_r{acc}_recon_dcml" in x.name.lower()][0]
                    recon       = load_nifti_as_array(recon_fpath)
                    df          = calc_iqm_and_add_to_df(df, recon, target_fov2, pat_dir, acc, iqms, fov, decimals, iqm_mode, logger)

                elif fov == 'lsfov':
                    pass
            
            # Calculate an SSIM map of the reconstruction versus the target
            if do_ssim_map:
                ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
                calculate_and_save_ssim_map_3d(
                    target      = target_fov2,
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
    types = {'pat_id': 'str', 'acceleration': 'int', 'roi': 'str', 'slice': 'int'}
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

    # VIOLIN
    # plot_all_iqms_vs_accs_violin(
    #     df        = df,
    #     metrics   = iqms,
    #     save_path = fig_dir / debug_str / "all_iqms_vs_accs_violin.png",
    #     logger    = logger,
    # )
    
    # Plot all IQMs vs acceleration rates and FOVs using box plots
    plot_all_iqms_vs_accs_vs_fovs_boxplot(
        df        = df,
        metrics   = iqms,
        save_path = fig_dir / debug_str / "all_iqms_vs_accs_vs_fovs_boxplot.png",
        logger    = logger,
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


def main(
    cfg: dict              = None,
    logger: logging.Logger = None,
) -> None:
    df = calculate_or_load_data(cfg, logger)
    print(summarize_dataframe(df))
    generate_plots(df, cfg, logger)
    # generate_tables(df, cfg, logger)


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
        'iqm_mode':           '2d',                              # The mode for calculating the IQMs. Options are: ['3d', '2d']. The iqm will either be calculated for a 2d image or 3d volume, where the 3d volume IQM is the average of the 2d IQMs for all slices.
        'decimals':           3,                                 # Number of decimals to round the IQMs to.
        'do_consider_rois':   True,                              # Whether to consider the different ROIs for the IQM calculation.
        'do_ssim_map':        False,                             # Whether to calculate and save the SSIM map.
        'fovs':               ['abfov', 'prfov'],                #['abfov', 'prfov', 'lsfov'],       # The field of views to process. Options are: ['abfov', 'prfov', 'lsfov']
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
    
    if cfg['debug']:
        cfg['include_list'] = ['0053_ANON5517301', '0032_ANON7649583', '0120_ANON7275574']
    
    main(cfg, logger)