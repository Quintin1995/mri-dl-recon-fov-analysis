from typing import Dict, List
from pathlib import Path
from assets.util import setup_logger


def get_configurations():
    cfg = {
        'patients_dir':       Path("/scratch/hb-pca-rad/projects/03_reader_set_v2"),
        "log_dir":            Path('logs'),
        'include_list_fpath': Path('lists/include_ids.lst'),        # List of patient_ids to include as Path
        'debug':              False,
        'pos_pats':           [
                                '0003_ANON5046358',
                                '0008_ANON8890538',
                                '0010_ANON7748752',
                                '0018_ANON9843837',
                                '0049_ANON9783006',
                                '0059_ANON7955208',
                                '0067_ANON0913099',
                                '0068_ANON7978458',
                                '0070_ANON5223499',
                                '0073_ANON5954143',
                                '0094_ANON8024204',
                                '0095_ANON4189062',
                                '0103_ANON8583296',
                                '0105_ANON9883201',
                                '0109_ANON9816976',
                                '0110_ANON8266491',
                                '0129_ANON5344332',
                                '0135_ANON9879440',
                                '0137_ANON8035619',
                                '0142_ANON7090827',
                                '0143_ANON9752849',
                                '0145_ANON0335209',
                                '0146_ANON7414571',
                                '0150_ANON5824292',
                                '0153_ANON5958718'
                                ]
    }

    for key, value in cfg.items():
        print(f"{key}: {value}")

    return cfg


def main(patients_dir: Path, include_list: List[str], pos_pats: List[str], logger, **cfg) -> None:
    logger.info(f"\n\n\nSTARTING {__file__} with {len(include_list)} patients")

    # Create dict per patient and calculate if positive and the number of lesions per patient.
    data = {}
    is_pos_counter = 0
    for patient in include_list:
        pat_dir = patients_dir / patient
        logger.info(f"Processing {pat_dir}")
        rois = list(pat_dir.glob('*roi*'))

        is_pos = 1 if len(rois) > 0 else 0
        if is_pos:
            is_pos_counter += 1

        if patient in pos_pats:
            data[patient] = {'is_positive': is_pos, 'num_lesions': len(rois)}
            for roi in rois:
                logger.info(f"\t\tFound: {roi}")
        elif len(rois) == 0:
            logger.warning(f"\tNo ROIs found for {patient}")
        
    num_pos = sum(v['is_positive'] for v in data.values())
    num_lesions = sum(v['num_lesions'] for v in data.values() if v['num_lesions'] > 0)

    logger.info(f"Number of positive patients: {num_pos}")
    logger.info(f"Number of lesions: {num_lesions}")
    
    for patient, values in data.items():
        logger.info(f"{patient}: {values}")
    logger.info(f"is_pos_counter: {is_pos_counter}")


if __name__ == '__main__':
    cfg = get_configurations()
    
    log_fname = 'calc_num_pos_and_num_lesions_debug' if cfg['debug'] else 'calc_num_pos_and_num_lesions'
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

    main(logger=logger, **cfg)
    logger.info(f"FINISHED {__file__}")
    print(f"Done")