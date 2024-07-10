import nibabel as nib
import SimpleITK as sitk
import os
import logging
from typing import List

from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

from assets.util import setup_logger
from assets.operations import clip_image_sitk
from calculate_iqms_to_csv import filter_patient_dirs


def convert_mha_to_nifti(mha_path, nifti_path):
    img = sitk.ReadImage(str(mha_path))
    sitk.WriteImage(img, str(nifti_path))


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

    if False:
        # only select patients where the x.name split first part cast as int is bigger than 103
        patients_dirs = [x for x in patients_dirs if int(x.name.split('_')[0]) > 103]
        
    if logger:
        logger.info(f"Found {len(patients_dirs)} patient directories in {rootdir}")

    return patients_dirs


def create_segs_for_patients(
    patients_dir: Path,
    include_list: List[str],
    seg_dir: Path,
    clip_percentiles: tuple,
    logger: logging.Logger,
    **cfg
):
    pat_dirs = filter_patient_dirs(patients_dir, include_list, logger)
    tasks = ['total_mr', 'tissue_types_mr']

    for pat_idx, pat_dir in enumerate(pat_dirs):
        logger.info(f"\n\n\nProcessing patient {pat_idx + 1}/{len(pat_dirs)}: {pat_dir.name}")
        
        # Convert mha to nifti because we read with nibabel instead of SimpleITK
        mha_path = pat_dir / f"{pat_dir.name}_rss_target_dcml.mha"
        nifti_path = pat_dir / f"{pat_dir.name}_rss_target_dcml.nii.gz"
        convert_mha_to_nifti(mha_path, pat_dir / nifti_path)

        # Clip the image intensities to the 1st and 99th percentiles and save the clipped image to the patient directory
        sitk_img = sitk.ReadImage(str(nifti_path))
        clipped_sitk_img = clip_image_sitk(sitk_img, clip_percentiles[0], clip_percentiles[1])
        clipped_nifti_path = pat_dir / f"{pat_dir.name}_rss_target_dcml.nii.gz"
        sitk.WriteImage(clipped_sitk_img, str(clipped_nifti_path))
        logger.info(f"Clipped image saved to: {clipped_nifti_path}")
        
        # Compute the segmentations for the clipped image and save them to the segmentation directory for each task
        for task in tasks:
            out_path = seg_dir / f"{pat_dir.name}_mlseg_{task}.nii.gz"
            multi_label_seg = totalsegmentator(input=nifti_path, output=out_path, fast=False, task=task, verbose=True, fastest=False, ml=True, license_number='aca_UZG10UFO4WQV5G')
            nib.save(multi_label_seg, out_path)
            logger.info(f"Segmentation saved to: {out_path}")


def main(cfg: dict, logger: logging.Logger):
    create_segs_for_patients(logger=logger, **cfg)


def get_configs(verbose=False):
    wd = Path(os.getcwd())
    cfg = {
        "patients_dir":       Path(r"C:\Users\Quintin\Documents\phd_local\03_datasets\03_umcg_nki_reader_set_v2"),
        "seg_dir":            Path(r"C:\Users\Quintin\Documents\phd_local\03_datasets\03_umcg_nki_reader_set_v2\segs"),
        'include_list_fpath': wd / 'lists/include_ids.lst',        # List of patient_ids to include as Path
        "log_dir":            wd / "logs",
        "debug":              False,
        "clip_percentiles":   (1, 99),     # Clip the image intensities to the 1st and 99th percentiles
    }
    if verbose:
        print("\nConfigs:")
        for key, value in cfg.items():
            print(f"{key}: {value}")
        print()
        print(f"Current working directory: {os.getcwd()}")
    if not cfg["seg_dir"].exists():
        cfg["seg_dir"].mkdir()
        print(f"Created segmentation directory: {cfg['seg_dir']}")
    
    return cfg


if __name__ == "__main__":
    print(f"\n\n\nTHIS SCRIPT I RUN LOCALLY VIA WINDOWS BECAUSE OF THE INSTALLED TotalSegmentator PACKAGE!!!\n\n\n")
    cfg = get_configs(verbose=True)
    
    log_fname = 'create_tissue_segs_debug' if cfg['debug'] else 'create_tissue_segs'
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
        cfg['include_list'] = ['0003_ANON5046358', '0006_ANON2379607', '0007_ANON1586301']  # have rois 
        cfg['include_list'] = ['0003_ANON5046358', '0004_ANON9616598']  # small segmentation test
    
    main(cfg, logger)