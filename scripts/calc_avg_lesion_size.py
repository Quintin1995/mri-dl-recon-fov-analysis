from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
from typing import Tuple, List

from calculate_iqms_to_csv import filter_patient_dirs
from assets.util import setup_logger


def get_bounding_box_2d(img_slice: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of non-zero elements in a 2D image slice.
    
    Parameters:
    img_slice (np.ndarray): 2D image slice.
    
    Returns:
    Tuple[int, int, int, int]: Bounding box coordinates (min_x, max_x, min_y, max_y).
    """
    non_zero_coords = np.argwhere(img_slice)
    min_x, min_y = non_zero_coords.min(axis=0)
    max_x, max_y = non_zero_coords.max(axis=0)
    return min_x, max_x, min_y, max_y


def save_lesion_stats(lesion_bounding_boxes: List[Tuple[int, int, int, int, int, str, str]], logger):
    """
    Save lesion statistics to a CSV file.
    
    Parameters:
    lesion_bounding_boxes (List): List of lesion bounding box data.
    logger: Logger for logging messages.
    """
    lesion_bounding_boxes = np.array(lesion_bounding_boxes, dtype=object)
    x_ranges = lesion_bounding_boxes[:, 1] - lesion_bounding_boxes[:, 0]
    y_ranges = lesion_bounding_boxes[:, 3] - lesion_bounding_boxes[:, 2]

    lesion_stats = np.column_stack((lesion_bounding_boxes, x_ranges, y_ranges))
    lesion_stats = pd.DataFrame(lesion_stats, columns=['min_x', 'max_x', 'min_y', 'max_y', 'z', 'patient_id', 'roi_name', 'x_range', 'y_range'])
    out_dir = Path('data/intermediary')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fpath = out_dir / 'lesion_bounding_boxes.csv'
    lesion_stats.to_csv(out_fpath, sep=';', index=False)
    logger.info(f"Saved lesion bounding box statistics to {out_fpath}")

    logger.info(f"Average x-range: {np.mean(x_ranges):.2f}, StdDev x-range: {np.std(x_ranges):.2f}")
    logger.info(f"Average y-range: {np.mean(y_ranges):.2f}, StdDev y-range: {np.std(y_ranges):.2f}")



def process_patient_dir(pat_dir: Path, lesion_bounding_boxes: List[Tuple[int, int, int, int, int, str, str]], logger):
    logger.info(f"Processing {pat_dir}")
    # Load all the ROI files that are not capital. The filename should have 'roi' in it ENSURE WE DONT GET CAPITAL ROI
    rois = [x for x in pat_dir.glob('**/*roi*') if not x.name.isupper()]
    logger.info(f"\tFound {len(rois)} ROIs in {pat_dir}")

    for roi in rois:
        logger.info(f"\t\tOpening {roi}")
        img = sitk.ReadImage(str(roi))
        img_array = sitk.GetArrayFromImage(img)

        # Loop over the slices and calculate the x-range and y-range of the lesion in 2D
        for z in range(img_array.shape[0]):
            img_slice = img_array[z, :, :]
            if np.any(img_slice):  # If there are non-zero values in the slice
                min_x, max_x, min_y, max_y = get_bounding_box_2d(img_slice)
                lesion_bounding_boxes.append((min_x, max_x, min_y, max_y, z, pat_dir.name, roi.name))

    return lesion_bounding_boxes


def main(cfg, logger):
    patients_dir = cfg['patients_dir']
    include_list = cfg.get('include_list', None)
    pat_dirs = filter_patient_dirs(patients_dir, include_list, logger=logger, do_sort=True)
    
    lesion_bounding_boxes = []  # Store the bounding boxes of all lesions
    for pat_dir in pat_dirs:
        process_patient_dir(pat_dir, lesion_bounding_boxes, logger)

    save_lesion_stats(lesion_bounding_boxes, logger)


def get_configurations():
    cfg = {
        'patients_dir':       Path("/scratch/hb-pca-rad/projects/03_reader_set_v2"),
        "log_dir":            Path('logs'),
        'include_list_fpath': Path('lists/include_ids.lst'),        # List of patient_ids to include as Path
        'debug':              False,
    }

    for key, value in cfg.items():
        print(f"{key}: {value}")

    return cfg


if __name__ == "__main__":
    cfg = get_configurations()

    log_fname = 'calc_avg_lesion_size_debug' if cfg['debug'] else 'calc_avg_lesion_size'
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname=log_fname)
    
    if cfg.get('include_list_fpath'):
        try:
            with open(cfg['include_list_fpath'], 'r') as f:
                cfg['include_list'] = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f"Inclusion list file not found: {cfg['include_list_fpath']}")
            exit(1)
    
    if cfg['debug']:
        cfg['include_list'] = ['0003_ANON5046358', '0006_ANON2379607', '0007_ANON1586301']  # have rois 
    
    main(cfg, logger)