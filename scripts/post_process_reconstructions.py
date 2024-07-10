import os
import glob
import h5py
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
import numpy as np
import pydicom
import sqlite3
import yaml
import logging

from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
# from direct.functionals.challenges import fastmri_ssim, fastmri_psnr, fastmri_nmse
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr


def parse_args(verbose: bool = False):
    """
    Parse command line arguments.
    Parameters:
    - verbose (bool): Flag for printing command line arguments to console.
    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Post process the .h5 file recontructions into dicom like reconstructions where the segmentation fits on top of. So with the correct dimension, orientation and pixel spacing as the dicom images.")
    
    # Path to the config file to run the program
    parser.add_argument("-cfg",
                        "--config_fpath",
                        type=str,
                        default="/home1/p290820/repos/direct-with-averages/projects/qvl_nki_rs/configs/post_process_config.yaml",
                        help="Path to the config file.")
    
    a = parser.parse_args()

    if verbose:
        print("Command line arguments:")
        for arg in vars(a):
            print(f"- {arg}:\t\t{getattr(a, arg)}")

    return a


def load_config():
    args = parse_args(verbose=True)
    with open(args.config_fpath, 'r', encoding='UTF-8') as file:
        cfg = yaml.safe_load(file)
    
    # Update keys that are expected to contain single path strings
    keys_to_make_a_path = ['source_dir', 'log_dir', 'db_fpath']  # Assuming these are single paths
    for key in keys_to_make_a_path:
        if key in cfg:
            cfg[key] = Path(cfg[key])
    
    # Specific handling for prediction_dirs if it contains multiple paths
    if 'prediction_dirs' in cfg:
        cfg['prediction_dirs'] = {k: Path(v) for k, v in cfg['prediction_dirs'].items()}

    for k, v in cfg.items():
        print(f"{k}:\t{v}  \tdtype: {type(v)}")
    print("")
    
    return cfg


def print_start_message():
    print("\nTakes in a directory of validation predictions and a directory of validation targets and plots them side by side for comparison.")
    print("-> Computes the SSIM, PSNR, and NMSE between the predictions and targets and saves them in the h5 files.")
    print("-> Saves the predictions and targets as nifti files for visualization in ITK-SNAP.")
    print("-> Current working directory: ", os.getcwd(), "\n\n")


def histogram_normalization(src_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    # Flatten the images
    src_img_flat = src_img.flatten()
    ref_img_flat = ref_img.flatten()

    src_img_flat = np.round(src_img_flat).astype(int)
    ref_img_flat = np.round(ref_img_flat).astype(int)

    # Compute histograms
    src_hist, src_bins = np.histogram(src_img_flat, bins=256)
    ref_hist, ref_bins = np.histogram(ref_img_flat, bins=256)

    # Compute cumulative distribution functions (CDF)
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    # Normalize CDFs so that they span from 0 to 255
    src_cdf = (src_cdf - src_cdf.min()) * 255 / (src_cdf.max() - src_cdf.min())
    ref_cdf = (ref_cdf - ref_cdf.min()) * 255 / (ref_cdf.max() - ref_cdf.min())

    # Mapping from src to ref
    lut = np.interp(src_cdf, ref_cdf, src_bins[:-1])

    # Ensure the LUT indices do not exceed the maximum index
    lut_indices = np.clip(src_img_flat, 0, len(lut) - 1)

    return lut[lut_indices].reshape(src_img.shape).astype(np.uint8)


def plot_histograms(
    array1: np.ndarray,
    array2: np.ndarray,
    directory: str = '/home1/p290820/tmp',
    outfname = "",
    logger: Optional[logging.Logger] = None
):
    # Check directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate histogram data
    hist1, bins1 = np.histogram(array1.flatten(), bins=256, range=[0,256])
    hist2, bins2 = np.histogram(array2.flatten(), bins=256, range=[0,256])

    # Create plot
    plt.figure()
    plt.hist(bins1[:-1], bins1, weights=hist1, alpha=0.5, label='Array 1')
    plt.hist(bins2[:-1], bins2, weights=hist2, alpha=0.5, label='Array 2')
    
    plt.title('Histograms')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    # Save plot
    save_path = os.path.join(directory, f"{outfname}.png")
    plt.savefig(save_path)

    logger.info(f"Saved plot at {save_path}")


def safe_as_sitk(
    img: np.ndarray,
    filename: str,
    reference_sitk: Optional[sitk.Image] = None,
    do_hist_norm: bool = False,
    do_round: bool = True,
    logger: Optional[logging.Logger] = None,
    verbose: bool  = False,
) -> None:
    """Save a numpy array as a nifti file using SimpleITK."""
    
    assert isinstance(img, np.ndarray), "Expecting a numpy array."
    assert not do_hist_norm or (do_hist_norm and reference_sitk is not None), "Expecting a reference sitk image when do_hist_norm is true."

    # round the image to the nearest integer for less memory usage
    if do_round:
        img = (img - img.min()) / (img.max() - img.min())
        img = np.round(img*1000)

    # convert the numpy array to a sitk image
    image = sitk.GetImageFromArray(img)

    # copy the information from the reference sitk image
    if reference_sitk is not None:
        logger.info(f"\t\tCopying information from reference sitk image")
        image.CopyInformation(reference_sitk)

    if do_hist_norm:
        # Assuming reference_sitk is already checked to be not None
        ref_img_np = sitk.GetArrayFromImage(reference_sitk)
        
        # Call the histogram_normalization function
        normalized_img = histogram_normalization(src_img=img, ref_img=ref_img_np)
        
        # Plot histograms before normalization
        plot_histograms(img, ref_img_np, outfname="before_hist_norm")
        
        # Plot histograms after normalization
        plot_histograms(normalized_img, ref_img_np, outfname="after_hist_norm")
        
        # Update img to the normalized image
        img = normalized_img

        # Update the sitk image
        image = sitk.GetImageFromArray(img)
        image.CopyInformation(reference_sitk)

    # save the sitk image to file
    sitk.WriteImage(image, filename)

    if verbose:
        logger.info(f"\t\tSaved as: {filename}")


def fastmri_ssim_qvl(gt: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM compatible with the FastMRI challenge."""
    assert gt.shape == target.shape, "Shape mismatch between gt and target."
    assert gt.dtype == target.dtype, "Data type mismatch between gt and target."
    assert len(gt.shape) == 3, "Expecting 3D arrays."
    
    return structural_similarity(
        gt,
        target,
        channel_axis=0,
        data_range=gt.max()
    )

def fastmri_psnr_qvl(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute PSNR compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."
    assert len(gt.shape) == 3, "Expecting 3D arrays."

    return psnr(
        image_true=gt,
        image_test=pred,
        data_range=gt.max()
    )

def fastmri_nmse_qvl(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute NMSE compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."
    assert len(gt.shape) == 3, "Expecting 3D arrays."

    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


# def get_h5_filenames_from_dir(root_dir: str, verbose = False) -> list:
#     """Get a list of h5 filenames from a directory."""
    
#     fnames = glob.glob(root_dir + '/*/*_pst_T2.h5')

#     if verbose:
#         print("Getting h5 filenames from: ", root_dir)
#         print("Number of h5 files: ", len(fnames))

#     return fnames


def process_recon(
    fpath_h5: Path,
    do_safe_as_nifti = False,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Process the reconstruction from an h5 file."""
    
    hf = h5py.File(fpath_h5, 'r')
    recon = hf['reconstruction'][()]
    acc_fac = hf.attrs["acceleration_factor"]
    modelname = hf.attrs["modelname"]
    hf.close()

    if verbose:
        logger.info(f"\tShape of the reconstruction: {recon.shape}, with dtype {recon.dtype}")
    
    if do_safe_as_nifti:
        outfname = Path(fpath_h5)
        parent_dir = outfname.parent
        # recons_dir = parent_dir / 'recons'
        # recons_dir.mkdir(exist_ok=True)  # Create 'recons' directory if it doesn't exist
        outfname = parent_dir / f"{modelname}_R{int(acc_fac)}_recon.nii.gz"
        safe_as_sitk(
            img            = recon,
            filename       = outfname,
            reference_sitk = None,
            do_hist_norm   = False,
            verbose        = True,
            do_round       = True,
            logger         = logger
        )

    return recon


def add_vis_qual_metrics_to_h5(hf: h5py, recon: np.ndarray, target: np.ndarray, logger: Optional[logging.Logger] = None):
    """Add SSIM, PSNR, and NMSE to the h5 file."""

    if 'ssim' not in hf.attrs.keys():
        hf.attrs['ssim'] = fastmri_ssim_qvl(target, recon)

    if 'psnr' not in hf.attrs.keys():
        hf.attrs['psnr'] = fastmri_psnr_qvl(target, recon)
    
    if 'nmse' not in hf.attrs.keys():
        hf.attrs['nmse'] = fastmri_nmse_qvl(target, recon)
    
    if logger:
        logger.info(f"\tSSIM: {hf.attrs['ssim']:.4f}, PSNR: {hf.attrs['psnr']:.4f}, NMSE: {hf.attrs['nmse']:.4f}")


def save_slice_as_greyscale(arr: np.ndarray, slice_idx: int = 15, verbose: bool = False, dir='/home1/p290820/tmp/006_ksp_zero_padding_sim_ksp_and_post_process_recon', title="") -> None:
    """Save a slice of an array as greyscale to file."""
    
    dim = len(arr.shape)
    assert dim in [2, 3], "Expecting 2D or 3D array."
    
    if dim == 3:
        data_slice = np.abs(arr[slice_idx, ...])
    else:
        data_slice = np.abs(arr)

    plt.imshow(data_slice, cmap='gray')
    outfname = Path(dir) / f"{title}_slice{slice_idx}.png"
    
    plt.savefig(outfname)
    
    if verbose:
        print(f"\nSaved the slice as greyscale to file: ", outfname, f"  Shape = {data_slice.shape}")


def zero_pad_in_sim_kspace(
    input_image: np.ndarray, 
    desired_shape: Tuple[int, int] = (1280, 1280), 
    logger: logging.Logger = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Zero pad a 3D image array in the simulated k-space.
    
    Parameters:
    - input_image: 3D ndarray, the image to be zero-padded
    - desired_shape: tuple, the desired shape after zero-padding
    - verbose: boolean, whether to print additional information
    
    Returns:
    - 3D ndarray, the zero-padded image
    """
    assert input_image.dtype == np.float32, "Expecting the input image to be of type float32."
    assert input_image.shape[0] < input_image.shape[1] and input_image.shape[0] < input_image.shape[2], "Expecting the first dim to be the slice dim."
    assert len(input_image.shape) == 3, "Expecting 3D array."
    
    if verbose:
        logger.info(f"\tShape of the input image before zero-padding: {input_image.shape}, with dtype {input_image.dtype}")

    n_slices, width, height = input_image.shape
    padded_image = np.zeros((n_slices, *desired_shape), dtype=input_image.dtype)

    pad_width = desired_shape[0] - width
    pad_height = desired_shape[1] - height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    for i in range(n_slices):
        img_slice            = input_image[i, ...]
        kspace_slice         = np.fft.fft2(img_slice)
        kspace_slice_shift1  = np.fft.fftshift(kspace_slice)
        kspace_slice_padded  = np.pad(kspace_slice_shift1, ((pad_left, pad_right), (pad_top, pad_bottom)), mode='constant', constant_values=0)
        kspace_slice_shift2  = np.fft.ifftshift(kspace_slice_padded)
        img_slice_padded     = np.fft.ifft2(kspace_slice_shift2)
        padded_image[i, ...] = np.abs(img_slice_padded)

        if verbose: 
            logger.info(f"\t\tSlice {i+1}/{n_slices} done.")
            logger.info(f"\t\t\tShape of the kspace_slice: {kspace_slice.shape}, with dtype {kspace_slice.dtype}")
            logger.info(f"\t\t\tShape of the kspace_slice_padded: {kspace_slice_padded.shape}, with dtype {kspace_slice_padded.dtype}")
            logger.info(f"\t\t\tShape of the img_slice_padded: {img_slice_padded.shape}, with dtype {img_slice_padded.dtype}")
            logger.info(f"\t\t\tShape of the padded_image: {padded_image.shape}, with dtype {padded_image.dtype}")

    return padded_image


def center_crop(
    arr: np.ndarray,
    crop_shape: Tuple[int, int],
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Perform a center crop on a 2D or 3D numpy array.
    
    Parameters:
    - arr (np.ndarray): The array to be cropped. Could be either 2D or 3D.
    - crop_shape (Tuple[int, int]): Desired output shape (height, width).
    
    Returns:
    - np.ndarray: Center cropped array.
    """
    
    # Validate dimensions
    if len(arr.shape) not in [2, 3]:
        raise ValueError(f"Invalid number of dimensions. Expected 2D or 3D array, got shape {arr.shape}.")
        
    if len(crop_shape) != 2:
        raise ValueError(f"Invalid crop_shape dimension. Expected a tuple of length 2, got {crop_shape}.")

    # Unpack shapes
    original_shape = arr.shape[-2:]
    crop_height, crop_width = crop_shape
    
    # Calculate padding dimensions
    pad_height = original_shape[0] - crop_height
    pad_width = original_shape[1] - crop_width
    
    if pad_height < 0 or pad_width < 0:
        raise ValueError(f"Crop shape {crop_shape} larger than the original shape {original_shape}.")
        
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Perform the crop
    if len(arr.shape) == 3:
        cropped_arr = arr[:, pad_top:-pad_bottom, pad_left:-pad_right]
    else:
        cropped_arr = arr[pad_top:-pad_bottom, pad_left:-pad_right]

    logger.info(f"\tCenter crop: Original shape: {original_shape}, crop shape: {crop_shape}, pad: {pad_top, pad_bottom, pad_left, pad_right}, cropped shape: {cropped_arr.shape}")
        
    return cropped_arr


def test_spectrum_analysis(slice_recon: np.ndarray, slice_recon_zp: np.ndarray):
    """
    Perform a spectrum analysis on a slice of the reconstruction before and after zero-padding.
    
    Parameters:
    - slice_recon: 2D ndarray, a slice from the original reconstruction
    - slice_recon_zp: 2D ndarray, a slice from the zero-padded reconstruction
    
    Returns:
    - None, but generates plots to visualize the spectrum
    """
    
    # Asserting input shapes and types for robustness
    assert slice_recon.ndim == 2, "Expecting a 2D array for slice_recon."
    assert slice_recon_zp.ndim == 2, "Expecting a 2D array for slice_recon_zp."

    # Fourier Transform of the original slice
    kspace_slice = np.fft.fftshift(np.fft.fft2(slice_recon))
    
    # Fourier Transform of the zero-padded slice
    kspace_slice_zp = np.fft.fftshift(np.fft.fft2(slice_recon_zp))

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Plot magnitude and phase for the original slice
    axs[0, 0].imshow(np.log(np.abs(kspace_slice) + 1e-6), cmap='gray')
    axs[0, 0].set_title('Original Magnitude Spectrum')
    axs[0, 1].imshow(np.angle(kspace_slice), cmap='gray')
    axs[0, 1].set_title('Original Phase Spectrum')

    # Plot magnitude and phase for the zero-padded slice
    axs[1, 0].imshow(np.log(np.abs(kspace_slice_zp) + 1e-6), cmap='gray')
    axs[1, 0].set_title('Zero-Padded Magnitude Spectrum')
    axs[1, 1].imshow(np.angle(kspace_slice_zp), cmap='gray')
    axs[1, 1].set_title('Zero-Padded Phase Spectrum')

    # save the figure to the tmp dir
    plt.savefig('/home1/p290820/tmp/006_ksp_zero_padding_sim_ksp_and_post_process_recon/spectrum_analysis.png')


def validate_input_image_space_input(input_image: np.ndarray, logger: logging.Logger = None):
    if input_image is None:
        logger.error("Input image must be provided.")
        raise ValueError("Input image must be provided.")
    if input_image.dtype != np.float32:
        logger.error("Input image dtype must be float32.")
        raise ValueError("Input image dtype must be float32.")
    if len(input_image.shape) != 3:
        logger.error("Input image must be a 3D array.")
        raise ValueError("Input image must be a 3D array.")
    if input_image.shape[0] >= input_image.shape[1] or input_image.shape[0] >= input_image.shape[2]:
        logger.error("First dimension should be the slice dimension and should be smaller than the other two.")
        raise ValueError("First dimension should be the slice dimension and should be smaller than the other two.")


def get_shapes_from_dicom(dicom_dir: str, logger=logging.Logger) -> Tuple:
    """Get the zero-pad shape and image space crop shape from the first dicom file in the directory."""

    pattern = os.path.join(dicom_dir, '*')
    logger.info(f"\tLooking for dicom files in pattern: {pattern}")

    dcm_fpaths = glob.glob(pattern)
    logger.info(f"\tNumber of detected dicom files in: {len(dcm_fpaths)}")

    first_slice_fpath = dcm_fpaths[0]

    ds = pydicom.dcmread(first_slice_fpath)

    zero_pad_shape = (ds.Rows*2, ds.Columns*2)
    image_space_crop = (ds.Rows, ds.Columns)
    logger.info(f"\tCalculated ZERO-PAD shape: {zero_pad_shape}, based on rows, cols of the DICOM: {image_space_crop}")

    return zero_pad_shape, image_space_crop


def apply_image_processing_steps(
    input_image: np.ndarray,
    zero_pad_shape: Tuple,
    image_space_crop: Tuple,
    logger: logging.Logger,
    verbose: bool = False
) -> np.ndarray:
    """
    Apply the image processing steps to make the input image look like a dicom image.
    Args:
        input_image (np.ndarray): The input image to be processed.
        zero_pad_shape (Tuple): The desired shape after zero-padding.
        image_space_crop (Tuple): The desired shape after cropping.
        logger (logging.Logger): The logger object.
        verbose (bool): Whether to print additional information.
    Returns:
        np.ndarray: The post-processed (dicom-like) reconstruction.
    """

    # Step 1: Flip both the width and height of the image, the RIM output relative to the dicom is flipped
    image = np.flip(input_image, axis=(0,1,2))
    if verbose:
        logger.info(f"\tFlipped width and height with axis=(0,1,2)")

    # Step 2: Zero-pad simulated kspace to the desired shape and get the image back from the padded kspace
    image = zero_pad_in_sim_kspace(input_image=image, desired_shape=zero_pad_shape, logger=logger, verbose=False)

    # Step 3: Take a center crop of the image space
    image = center_crop(image, crop_shape=image_space_crop, logger=logger)

    return image


def make_dicom_like(
    dicom_dir: str,
    input_image: np.ndarray = None,
    verbose: bool           = False,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Makes the given input image look like a dicom image.
    Steps: 
    1. Zero-pad the input image in the simulated kspace to the desired shape.
    2. Flip both the width and height of the image, the RIM output relative to the dicom is flipped.
    3. Take a center crop of the image space.
    Args:
        fpath_h5 (Path): Path to the h5 file.
        dicom_dir (str): Path to the dicom directory.
        input_image (np.ndarray): The input image to be processed.
        safe_as_nifti (bool): Whether to save the dicom-like image as nifti file.
        title (str): Title of the dicom-like image.
        verbose (bool): Whether to print additional information.
    Returns:
        np.ndarray: The post-processed (dicom-like) reconstruction.
    """

    # Validate the input image based on dimensionality and such
    validate_input_image_space_input(input_image, logger)

    # Obtain zero-pad shape from first dicom file
    zero_pad_shape, image_space_crop = get_shapes_from_dicom(dicom_dir, logger)

    # Apply the image processing steps
    image_dicom_like = apply_image_processing_steps(input_image, zero_pad_shape, image_space_crop, logger, verbose)
    
    return image_dicom_like


def find_nifti_t2w_file_in_pat_dir(pat_dir: Path):
    '''
    Description: Find the nifti t2w file in the patient directory.
    Args:
        pat_dir (Path): The patient directory.
    Returns:
        Path: The path to the nifti t2w file.
    '''

    t2w_files = list(pat_dir.glob('*tse2d*'))
    print(f"\tFound {len(t2w_files)} t2w files in {pat_dir}")

    if len(t2w_files) == 0:
        raise Exception("No t2w file found in the patient directory.")
    
    if len(t2w_files) > 1:
        raise Exception("Multiple t2w files found in the patient directory.")
    
    return t2w_files[0]


def add_vis_qual_met_to_h5_dicom_like(hf: h5py, recon=np.ndarray, target=np.ndarray, verbose=True):
    """
        Add SSIM, PSNR, and NMSE to the h5 file.
        The metrics are computed between the dicom-like recon and target.
        args (h5py): h5 file
        args (np.ndarray): dicom-like recon
        args (np.ndarray): target
        args (bool): verbose
        returns (None)
    """

    if 'ssim_dicom_like' not in hf.attrs.keys():
        hf.attrs['ssim_dicom_like'] = fastmri_ssim_qvl(target, recon)

    if 'psnr_dicom_like' not in hf.attrs.keys():
        hf.attrs['psnr_dicom_like'] = fastmri_psnr_qvl(target, recon)
    
    if 'nmse_dicom_like' not in hf.attrs.keys():
        hf.attrs['nmse_dicom_like'] = fastmri_nmse_qvl(target, recon)
    
    if verbose:
        print(f"\tSSIM_dicom_like: {hf.attrs['ssim_dicom_like']:.4f}, PSNR_dicom_like: {hf.attrs['psnr_dicom_like']:.4f}, NMSE_dicom_like: {hf.attrs['nmse_dicom_like']:.4f}")


def get_patient_dirs(prediction_dir: Path, inclusion_list = None) -> List[Path]:
    
    """Get a list of patient directories from a root directory."""
    
    # Get all directories in the prediction_dir directory that are not files
    pat_dirs = [d for d in Path(prediction_dir).iterdir() if d.is_dir()]

    # Filter: filter out each patient directory that does not adhere to the format: 0003_ANON5046358
    pat_dirs = [d for d in pat_dirs if len(d.name.split('_')) == 2]
    
    # Inclusion list is only the first part of the patient directory name named the sequential ID.
    if inclusion_list is not None:
        pat_dirs = [d for d in pat_dirs if d.name.split('_')[0] in inclusion_list]

    # SORTING: Sort the pat_dirs based on the sequential ID
    pat_dirs = sorted(pat_dirs, key=lambda x: int(x.name.split('_')[0]))

    # filter out the patients for which the first part of the dirname is smaller than the start_from_pat_id
    # pat_dirs = [d for d in pat_dirs if int(d.name.split('_')[0]) >= start_from_pat_id]

    return pat_dirs


def find_t2_tra_dir_in_study_dir(study_dir: Path) -> Path:
    """
    Description: Find the T2 TSE TRA directory in the study directory.
    Args:
        study_dir (Path): The study directory.
    Returns:
        Path: The T2 TSE TRA directory.
    """
    for seq_dir in study_dir.iterdir():
        if "tse2d1" in seq_dir.name.lower():  # Using lower() for case-insensitive match
            # List the files in the seq_dir and take the first and read with pydicom
            dcm_files = list(seq_dir.glob('*'))
            if dcm_files:  # Ensure there's at least one file to read
                dcm = pydicom.dcmread(dcm_files[0])
                # If ProtocolName contains T2, TSE, and TRA, case insensitive, then we return the directory
                protocol_name = dcm.ProtocolName.lower()  # Make comparison case-insensitive
                if "t2" in protocol_name and "tse" in protocol_name and "tra" in protocol_name:
                    return seq_dir  # Returning the directory, not the DICOM object
    return None  # Return None if no matching directory is found
            

def find_respective_dicom_dir(
    pat_id: str,
    source_dir: Path = None,
    db_fpath: Path = None,
) -> str:
    
    anon_id = pat_id.split('_')[-1]
    logger.info(f"Db patient ID: {anon_id}")
    
    conn = sqlite3.connect(str(db_fpath))
    cursor = conn.cursor()
    try:
        # Query to retrieve all MRI dates for the given patient ID
        query = "SELECT mri_date FROM kspace_dset_info WHERE anon_id = ? ORDER BY mri_date"
        cursor.execute(query, (anon_id,))
        results = cursor.fetchall()
        logger.info(f"\tResults from the query: {results}")
                
        if results:
            for result in results:        # loops over each study date found in the database and checks if it there is a matching dicom directory
                mri_date = str(result[0]) 
                mri_date_str = "{}-{}-{}".format(mri_date[:4], mri_date[4:6], mri_date[6:]) # Convert YYYY-MM-DD
                study_dir_path_dcms = source_dir / 'data' / pat_id / 'dicoms' / mri_date_str  # Construct expected DICOM dir path
                study_dir_path_niftis = source_dir / 'data' / pat_id / 'niftis' / mri_date_str  # Construct expected NIFTI dir path
                logger.info(f"\tChecking for study dir: {study_dir_path_dcms}")
                logger.info(f"\tChecking for study dir: {study_dir_path_niftis}")
                
                if study_dir_path_dcms.exists() and study_dir_path_niftis.exists():
                    logger.info(f"\t\tMatching dicom dir found for patient ID based on kspace acquisition date {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
                    logger.info(f"\t\tMatching nifti dir found for patient ID based on kspace acquisition date {pat_id} in {source_dir / 'data' / pat_id / 'niftis'}")
                    t2_tra_dcm_dir = find_t2_tra_dir_in_study_dir(study_dir_path_dcms)

                    # We have found the correct dicom link that immediatly to the to correct nifti and return that file too.
                    t2_tra_nif_fpath = Path(str(t2_tra_dcm_dir).replace('dicoms', 'niftis') + '.nii.gz')
                    
                    return t2_tra_dcm_dir, t2_tra_nif_fpath
                else:
                    logger.warning(f"\tNo matching study directory found for patient ID {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
                
            # If no matching directory is found after checking all dates
            logger.warning(f"\tNo matching study directory found for patient ID {pat_id}")
            raise Exception(f"No matching study directory found for patient ID {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
        else:
            logger.warning(f"\tNo MRI date found for patient ID {pat_id}")
            raise Exception(f"No MRI date found for patient ID {pat_id}")
    finally:
        conn.close()


def apply_transform_and_save_image(transform_fpath: Path, image_fname: str, output_fname: str, pat_dir: Path, logger: logging.Logger = None) -> None:
    """Applies a given transform to an image and saves the result with a specified filename."""
    if transform_fpath.exists():
        image_path = pat_dir / image_fname
        output_path = pat_dir / output_fname

        image = sitk.ReadImage(str(image_path))
        trans = sitk.ReadTransform(str(transform_fpath))
        image_transformed = sitk.Resample(image, trans)
        sitk.WriteImage(image_transformed, str(output_path))
        logger.info(f"\tTransformed image saved as: {output_path}")
        

def postprocess_all_patients(
    pat_dirs: List[Path],
    source_dir: Path         = None,
    do_make_dicom_like: bool = True,
    target_only: bool        = False,
    db_fpath: Path           = None,
    logger: logging.Logger   = None,
)-> None:
    """
    Description: Postprocess all the patients in the root dir.
    post process the recon and target from the h5 file.
    Args:
        pat_dirs (List[Path]): List of patient directories.
        source_dir (Path): The source directory where the h5 files are located.
        do_make_dicom_like (bool): Whether to make the recon and target dicom-like.
    Returns:
        None
    """
    for idx, pat_dir in enumerate(pat_dirs):
        pat_id = pat_dir.parts[-1]
        logger.info()
        logger.info(f"Loading patient {idx+1}/{len(pat_dirs)}:\nPatient ID: {pat_id}")
        hf_paths = glob.glob(str(pat_dir) + '/*.h5') # find the DL recon h5 file (vSHARP 4X recon in this case 20240207)
        
        if do_make_dicom_like:
            dicom_dir, t2_tra_nif_fpath = find_respective_dicom_dir(pat_id, source_dir, db_fpath)

        # loop over the h5 files in the patient directory we should correspond to an acquisition date with recon
        for f_idx, hf_path in enumerate(hf_paths):
            logger.info("")
            logger.info(f"Loading h5s patient {idx+1}/{len(pat_dirs)}:")
            logger.info(f"\nFile: {f_idx+1}/{len(hf_paths)}, {hf_path}")
            
            hf = h5py.File(hf_path, 'r+')
            logger.info(f"\tHeaders of the h5 file: {list(hf.keys())}")
            recon     = hf['reconstruction'][()] if not target_only else None
            target    = hf['target'][()]
            acc_fac   = int(hf.attrs["acceleration_factor"])
            modelname = hf.attrs["modelname"]

            if not target_only:
                safe_as_sitk(recon, filename=pat_dir / f"{modelname}_R{acc_fac}_recon.nii.gz", reference_sitk=None, do_hist_norm=False, do_round=True, logger=logger, verbose=True)
            safe_as_sitk(target, filename=pat_dir / f"rss_target.nii.gz", reference_sitk=None, do_hist_norm=False, do_round=True, logger=logger, verbose=True)
            hf.close()

            # Add the metrics to the h5 file
            # add_vis_qual_metrics_to_h5(hf=hf_pred, recon=recon, target=target, logger=logger)
            
            if do_make_dicom_like:
                recon_dicom_like  = make_dicom_like(dicom_dir, recon, verbose = True, logger=logger) if not target_only else None
                target_dicom_like = make_dicom_like(dicom_dir, target, verbose = True, logger=logger)
                reference_sitk = sitk.ReadImage(t2_tra_nif_fpath)
                if not target_only:
                    safe_as_sitk(
                        img            = recon_dicom_like,
                        filename       = pat_dir / f"{modelname}_R{acc_fac}_recon_dcml.nii.gz",
                        do_round       = True,
                        reference_sitk = reference_sitk,
                        do_hist_norm   = False,
                        logger         = logger,
                        verbose        = True,
                    )
                safe_as_sitk(
                    img            = target_dicom_like,
                    filename       = pat_dir / f"rss_target_dcml.nii.gz",
                    do_round       = True,
                    reference_sitk = reference_sitk,
                    do_hist_norm   = False,
                    logger         = logger,
                    verbose        = True,
                )
                
                # Apply transform if it exists and save the reconstruction image
                apply_transform_and_save_image(
                    transform_fpath = source_dir / 'data' / pat_id / 'transforms' / 'transform_recon_to_dicom.txt',
                    image_fname     = f"{modelname}_R{acc_fac}_recon_dcml.nii.gz",
                    output_fname    = f"{modelname}_R{acc_fac}_recon_dcml.nii.gz",
                    pat_dir         = pat_dir,
                    logger          = logger,
                )
                apply_transform_and_save_image(
                    transform_fpath = source_dir / 'data' / pat_id / 'transforms' / 'transform_recon_to_dicom.txt',
                    image_fname     = "rss_target_dcml.nii.gz",
                    output_fname    = "rss_target_dcml.nii.gz",
                    pat_dir         = pat_dir,
                    logger          = logger,
                )
            # if add_vis_qual_met_to_h5:
                # add_vis_qual_met_to_h5_dicom_like(hf=hf_pred, recon=recon_dicom_like, target=target_dicom_like, verbose=True)
                    
            logger.info(f"\tShape of the recon: {recon.shape}, with dtype {recon.dtype}") if not target_only else None
            logger.info(f"\tShape of the target: {target.shape}, with dtype {target.dtype}")
            logger.info(f"\tShape of the recon_dicom_like: {recon_dicom_like.shape}, with dtype {recon_dicom_like.dtype}") if not target_only else None
            logger.info(f"\tShape of the target_dicom_like: {target_dicom_like.shape}, with dtype {target_dicom_like.dtype}")


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


####################################################################################################
# Description:
# Post-process the reconstructions from the vSHARP model.
# This code is used to post-process the reconstructions from the vSHARP model.
# The reconstructions are stored in h5 files and are post-processed in the following way:
# 1. The reconstructions are saved as nifti files.
# 2. The reconstructions are made dicom-like.
# 3. The reconstructions are saved as dicom-like nifti files.
# 4. The reconstructions are evaluated using the SSIM, PSNR, and NMSE metrics.
# 5. The reconstructions are evaluated using the SSIM, PSNR, and NMSE metrics on the dicom-like reconstructions.
if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname='post_process_inference')
    
    for acceleration, pred_dir in cfg['prediction_dirs'].items():
        print("\n\n\n")
        logger.info(f"Processing {acceleration} acceleration data at {pred_dir}")
        postprocess_all_patients(
            pat_dirs           = get_patient_dirs(pred_dir, cfg['inclusion_list']),
            source_dir         = cfg['source_dir'],
            do_make_dicom_like = cfg['do_make_dicom_like'],
            target_only        = cfg['target_only'],
            db_fpath           = cfg['db_fpath'],
            logger             = logger,
        )