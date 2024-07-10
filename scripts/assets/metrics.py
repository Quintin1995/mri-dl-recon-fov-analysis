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
    - gt (np.ndarray): The ground truth (GT) image (3D).
    - pred (np.ndarray): The reconstructed image (3D).
    - sigma (float): The standard deviation of the Gaussian filter (default 1.5).
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - hfen_value (float or List[float]): The calculated HFEN value. Returns a list of HFEN values if iqm_mode is '2d'.
    """
    if iqm_mode == '2d':
        hfen_values = []
        for i in range(gt.shape[0]):
            log_gt_slice = gaussian_laplace(gt[:, :, i], sigma=sigma)
            log_pred_slice = gaussian_laplace(pred[:, :, i], sigma=sigma)
            
            numerator = norm(log_gt_slice.ravel() - log_pred_slice.ravel())
            denominator = norm(log_gt_slice.ravel())
            
            hfen_value = numerator / denominator if denominator != 0 else float('inf')
            hfen_values.append(hfen_value)
        return hfen_values
    else:
        log_gt = gaussian_laplace(gt, sigma=sigma)
        log_pred = gaussian_laplace(pred, sigma=sigma)
        
        log_gt_flat = log_gt.ravel()
        log_pred_flat = log_pred.ravel()
        
        numerator = norm(log_gt_flat - log_pred_flat)
        denominator = norm(log_gt_flat)
        
        hfen_value = numerator / denominator if denominator != 0 else float('inf')
        return hfen_value


def rmse(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Calculate the Root Mean Squared Error (RMSE) between two images.
    
    Parameters:
    - gt (np.ndarray): The ground truth (GT) image.
    - pred (np.ndarray): The reconstructed image.
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - rmse_value (float or List[float]): The calculated RMSE value. Returns a list of RMSE values if iqm_mode is '2d'.
    """
    if iqm_mode == '2d':
        rmse_values = []
        for i in range(gt.shape[0]):
            rmse_value = np.sqrt(np.mean((gt[i, :, :] - pred[i, :, :]) ** 2))
            rmse_values.append(rmse_value)
        return rmse_values
    else:
        return np.sqrt(np.mean((gt - pred) ** 2))


def fastmri_ssim(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute SSIM compatible with the FastMRI challenge, supporting 2D and 3D modes.

    Parameters:
    - gt (np.ndarray): The ground truth (GT) image (3D).
    - pred (np.ndarray): The reconstructed image (3D).
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    """
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."

    if iqm_mode == '2d':
        ssim_values = []
        for i in range(gt.shape[0]):
            ssim_value = ssim(
                gt[i, :, :],
                pred[i, :, :],
                data_range=gt[i, :, :].max() - gt[i, :, :].min()
            )
            ssim_values.append(ssim_value)
        return ssim_values
    else:
        return ssim(
            gt,
            pred,
            channel_axis=0,
            data_range=gt.max() - gt.min()
        )


def fastmri_psnr(gt: np.ndarray, pred: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """Compute PSNR compatible with the FastMRI challenge, supporting 2D and 3D modes."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt   = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."

    if iqm_mode == '2d':
        psnr_values = []
        for i in range(gt.shape[0]):
            psnr_value = psnr(
                image_true=gt[i, :, :],
                image_test=pred[i, :, :],
                data_range=gt[i, :, :].max() - gt[i, :, :].min()
            )
            psnr_values.append(psnr_value)
        return psnr_values
    else:
        return psnr(
            image_true=gt,
            image_test=pred,
            data_range=gt.max() - gt.min()
        )
    

def fastmri_nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute NMSE compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."

    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


def blurriness_metric(image: np.ndarray, iqm_mode: str = '3d') -> Union[float, List[float]]:
    """
    Compute a blurriness metric based on the Laplacian. We call this the variance of the Laplacian (VoFL).
    
    Parameters:
    - image (np.ndarray): The input image (2D or 3D).
    - iqm_mode (str): The mode of calculation ('2d' or '3d').
    
    Returns:
    - float or List[float]: The computed VoFL value. Returns a list of VoFL values if iqm_mode is '2d'.
    """
    if iqm_mode == '2d':
        vofl_values = []
        for i in range(image.shape[0]):
            laplacian = gaussian_laplace(image[i, :, :], sigma=1)
            vofl_value = np.var(laplacian)
            vofl_values.append(vofl_value)
        return vofl_values
    else:
        laplacian = gaussian_laplace(image, sigma=1)
        return np.var(laplacian)