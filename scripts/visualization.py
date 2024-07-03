import numpy as np
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


def save_slices_as_images(
    seg_bb: np.ndarray,
    recon_bb: np.ndarray,
    target_bb: np.ndarray,
    pat_dir: Path,
    output_dir: str,
    acceleration: int,
    lesion_num: int,
    is_mirror: bool = False,
    logger: logging.Logger = None
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


def plot_iqm_vs_accs_scatter_trend(
    df: pd.DataFrame,
    metric    = 'ssim',
    save_path = None,
    palette   = 'bright',
    logger: logging.Logger = None,
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

def plot_all_iqms_vs_accs_vs_fovs_boxplot(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: Path,
    logger: logging.Logger = None,
) -> None:
    sns.set(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 12,
    })
    
    # Generate individual plots
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='acceleration', y=metric, hue='roi')
        plt.title(f'{metric.upper()} by Acceleration and FOV (Box Plot)', fontsize=16, fontweight='bold')
        plt.xlabel('Acceleration', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.legend(title='FOV', loc='upper right')
        plt.grid(True, linestyle='--', linewidth=0.7)
        individual_save_path = save_path.parent / f"{metric}_vs_accs_vs_fovs_boxplot.png"
        plt.savefig(individual_save_path, bbox_inches='tight')
        plt.close()
        if logger:
            logger.info(f"Saved box plot for {metric} by acceleration and FOV at {individual_save_path}")
    
    # Create a 2x2 grid for the combined boxplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x='acceleration', y=metric, hue='roi', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Acceleration and FOV', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Acceleration', fontsize=12)
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].legend(title='FOV', loc='upper right')
        axes[i].grid(True, linestyle='--', linewidth=0.7)

    # Remove any unused subplots
    if len(metrics) < 4:
        for j in range(len(metrics), 4):
            fig.delaxes(axes[j])

    plt.tight_layout()
    combined_save_path = save_path.parent / "all_iqms_vs_accs_vs_fovs_boxplot.png"
    plt.savefig(combined_save_path, bbox_inches='tight')
    plt.close()
    if logger:
        logger.info(f"Saved combined box plot for all IQMs by acceleration and FOV at {combined_save_path}")