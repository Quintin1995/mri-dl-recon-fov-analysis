from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_laplace
from numpy.linalg import norm
import numpy as np
from typing import List, Union


def hfen(gt: np.ndarray, pred: np.ndarray, sigma: float = 1.5, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Calculate the High Frequency Error Norm (HFEN) between two images.
    
    Parameters:
    `gt`: The ground truth (GT) image.
    `pred`: The reconstructed image.
    `sigma`: The standard deviation of the Gaussian filter (default 1.5).
    `iqm_mode`: The mode of calculation ('2d' or '3d').
    
    Returns:
    `hfen_value`: The calculated HFEN value. Returns a list of HFEN values if iqm_mode is '2d' and input is 3D.
    """
    assert gt.shape == pred.shape, "Ground truth and predicted images must have the same shape"
    assert iqm_mode in ['2d', '3d'], "iqm_mode must be '2d' or '3d'"

    if gt.ndim == 2:
        # Directly compute HFEN for 2D input
        log_gt = gaussian_laplace(gt, sigma=sigma)
        log_pred = gaussian_laplace(pred, sigma=sigma)
        numerator = norm(log_gt - log_pred)
        denominator = norm(log_gt)
        return numerator / denominator if denominator != 0 else float('inf')

    elif gt.ndim == 3:
        if iqm_mode == '2d':
            # Compute HFEN for each slice in 3D input
            hfen_values = []
            for i in range(gt.shape[0]):
                log_gt_slice = gaussian_laplace(gt[i], sigma=sigma)
                log_pred_slice = gaussian_laplace(pred[i], sigma=sigma)
                numerator = norm(log_gt_slice - log_pred_slice)
                denominator = norm(log_gt_slice)
                hfen_value = numerator / denominator if denominator != 0 else float('inf')
                hfen_values.append(hfen_value)
            return hfen_values
        elif iqm_mode == '3d':
            # Compute HFEN over the entire 3D volume
            log_gt = gaussian_laplace(gt, sigma=sigma)
            log_pred = gaussian_laplace(pred, sigma=sigma)
            numerator = norm(log_gt.ravel() - log_pred.ravel())
            denominator = norm(log_gt.ravel())
            return numerator / denominator if denominator != 0 else float('inf')
    else:
        raise ValueError("Unsupported dimensionality of input data")


def rmse(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Calculate the Root Mean Squared Error (RMSE) between two images.
    
    Parameters:
    gt (np.ndarray): The ground truth (GT) image.
    pred (np.ndarray): The reconstructed image.
    iqm_mode (str): The mode of calculation ('2d' or '3d'). Defaults to '3d'.
    
    Returns:
    Union[float, List[float]]: The calculated RMSE value. Returns a list of RMSE values if iqm_mode is '2d' and input is 3D.
    """
    assert gt.shape == pred.shape, "Ground truth and predicted images must have the same shape"
    assert iqm_mode in ['2d', '3d'], "iqm_mode must be '2d' or '3d'"

    if gt.ndim == 2:
        # Directly compute RMSE for 2D input
        return np.sqrt(np.mean((gt - pred) ** 2))
    elif gt.ndim == 3:
        if iqm_mode == '2d':
            # Compute RMSE for each slice in 3D input
            return [np.sqrt(np.mean((gt[i] - pred[i]) ** 2)) for i in range(gt.shape[0])]
        elif iqm_mode == '3d':
            # Compute RMSE over the entire 3D volume
            return np.sqrt(np.mean((gt - pred) ** 2))
    else:
        raise ValueError("Unsupported dimensionality of input data")


def fastmri_ssim(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute SSIM compatible with the FastMRI challenge, supporting 2D and 3D modes.

    Parameters:
    - gt (np.ndarray): The ground truth (GT) image.
    - pred (np.ndarray): The reconstructed image.
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - Union[float, List[float]]: The calculated SSIM value. Returns a list of SSIM values if iqm_mode is '2d' and input is 3D.
    """
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2 and pred.ndim == 2:
        # Directly compute SSIM for 2D input
        return ssim(gt, pred, data_range=gt.max() - gt.min())

    assert gt.ndim == 3 and pred.ndim == 3, "Expecting 3D arrays."

    if iqm_mode == '2d':
        # Compute SSIM for each 2D slice in 3D input using list comprehension
        ssim_values = [
            ssim(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())
            for gt_slice, pred_slice in zip(gt, pred)
        ]
        return ssim_values
    elif iqm_mode == '3d':
        # Compute SSIM over the entire 3D volume
        return ssim(gt, pred, channel_axis=0, data_range=gt.max() - gt.min())
    else:
        raise ValueError("Invalid iqm_mode. Choose '2d' or '3d'.")


def fastmri_psnr(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute PSNR compatible with the FastMRI challenge, supporting 2D and 3D modes.
    
    Parameters:
    - gt (np.ndarray): The ground truth (GT) image.
    - pred (np.ndarray): The reconstructed image.
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - Union[float, List[float]]: The calculated PSNR value. Returns a list of PSNR values if iqm_mode is '2d' and input is 3D.
    """
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2 and pred.ndim == 2:
        # Directly compute PSNR for 2D input
        return psnr(gt, pred, data_range=gt.max() - gt.min())

    assert gt.ndim == 3 and pred.ndim == 3, "Expecting 3D arrays."

    if iqm_mode == '2d':
        # Compute PSNR for each 2D slice in 3D input using list comprehension
        psnr_values = [
            psnr(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())
            for gt_slice, pred_slice in zip(gt, pred)
        ]
        return psnr_values
    elif iqm_mode == '3d':
        # Compute PSNR over the entire 3D volume
        return psnr(gt, pred, data_range=gt.max() - gt.min())
    else:
        raise ValueError("Invalid iqm_mode. Choose '2d' or '3d'.")
    

def fastmri_nmse(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute NMSE compatible with the FastMRI challenge, supporting 2D and 3D modes.
    
    Parameters:
    - gt (np.ndarray): The ground truth (GT) image.
    - pred (np.ndarray): The reconstructed image.
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - Union[float, List[float]]: The calculated NMSE value. Returns a list of NMSE values if iqm_mode is '2d' and input is 3D.
    """
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2 and pred.ndim == 2:
        # Directly compute NMSE for 2D input
        return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2

    assert gt.ndim == 3 and pred.ndim == 3, "Expecting 3D arrays."

    if iqm_mode == '2d':
        # Compute NMSE for each 2D slice in 3D input using list comprehension
        nmse_values = [
            np.linalg.norm(gt_slice - pred_slice)**2 / np.linalg.norm(gt_slice)**2
            for gt_slice, pred_slice in zip(gt, pred)
        ]
        return nmse_values
    elif iqm_mode == '3d':
        # Compute NMSE over the entire 3D volume
        return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2
    else:
        raise ValueError("Invalid iqm_mode. Choose '2d' or '3d'.")


def blurriness_metric(image: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute a blurriness metric based on the Laplacian. This is the variance of the Laplacian (VoFL).
    
    Parameters:
    - image (np.ndarray): The input image (2D or 3D).
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - float or List[float]: The computed VoFL value. Returns a list of VoFL values if iqm_mode is '2d'.
    """
    if image.ndim == 2:
        # Directly compute VoFL for 2D input
        laplacian = gaussian_laplace(image, sigma=1)
        return np.var(laplacian)

    assert image.ndim == 3, "Expecting 3D arrays for '2d' or '3d' mode."

    if iqm_mode == '2d':
        # Compute VoFL for each 2D slice in 3D input using list comprehension
        vofl_values = [
            np.var(gaussian_laplace(slice_, sigma=1))
            for slice_ in image
        ]
        return vofl_values
    elif iqm_mode == '3d':
        # Compute VoFL over the entire 3D volume
        laplacian = gaussian_laplace(image, sigma=1)
        return np.var(laplacian)
    else:
        raise ValueError("Invalid iqm_mode. Choose '2d' or '3d'.")
