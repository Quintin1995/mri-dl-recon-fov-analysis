import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from pathlib import Path
from assets.operations import compute_error_map


def save_slice_metrics_image(
    recon_bb: np.ndarray,
    target_bb: np.ndarray,
    iqm_values: dict,
    x_coords: tuple,
    y_coords: tuple,
    output_dir: Path,
    acceleration: int,
    iqms: List[str],
    slice_idx: int,
    scaling: float = 1.0,
    lesion_num: int = None,  # Optional parameter
    seg_bb: np.ndarray = None,  # Optional parameter
    region_name: str = None,  # Optional parameter
    logger: logging.Logger = None
):
    """
    Save a single slice from recon_bb, target_bb, and error map as an image.
    Optionally include segmentation bounding box if provided.
    
    Parameters:
        seg_bb (np.ndarray, optional): The slice from the segmentation bounding box.
        recon_bb (np.ndarray): The slice from the reconstruction bounding box.
        target_bb (np.ndarray): The slice from the target bounding box.
        iqm_values (dict): The IQM values for this slice.
        x_coords (tuple): The x-coordinates of the bounding box.
        y_coords (tuple): The y-coordinates of the bounding box.
        output_dir (Path): The directory where the images will be saved.
        acceleration (int): The acceleration factor.
        iqms (List[str]): List of IQMs to display.
        slice_idx (int): The slice index.
        scaling (float): Scaling factor for the error map.
        lesion_num (int, optional): The lesion number if processing lesions.
        logger (logging.Logger, optional): Logger instance for logging information.
    
    Returns:
        None
    """
    num_plots = 4 if seg_bb is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))
    
    titles = ['Segmentation', 'Reconstruction', 'Target', f"Error Map ({scaling:.1f}x)"] if seg_bb is not None else ['Reconstruction', 'Target', f"Error Map ({scaling:.1f}x)"]
    slices = [seg_bb, recon_bb, target_bb, compute_error_map(target_bb, recon_bb, scaling)] if seg_bb is not None else [recon_bb, target_bb, compute_error_map(target_bb, recon_bb, scaling)]
    
    for ax, title, slice_data in zip(axes, titles, slices):
        ax.imshow(slice_data, cmap='gray')
        ax.axis('off')
        ax.set_title(f'{title} Slice {slice_idx}')

    # lets frame the min and max coords as x

    caption = "\n".join([f"{iqm}: {iqm_values[iqm]:.3f}" for iqm in iqms])
    caption += f"\nYs: {y_coords}, ydiff: {y_coords[1] - y_coords[0]}    Xs: {x_coords}, xdiff: {x_coords[1] - x_coords[0]}"
    caption += f"\nAcceleration: {acceleration}x"
    fig.text(0.5, 0.01, caption, ha='center', va='top', fontsize=12)
    
    fpath = output_dir / f"vsharp_{'lesion' if lesion_num else region_name}{lesion_num if lesion_num else ''}_slice{slice_idx}_R{acceleration}.png"
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"\t\t\tSaved slice {slice_idx} to {fpath}")


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


def get_full_name(abbr: str) -> str:
    full_name_map = {
        'FAV': 'Full Abdominal View (FAV)',
        'CPV': 'Clinical Prostate View (CPV)',
        'TLV': 'Targeted Lesion View (TLV)',
        'SFR': 'Sub. Fat Region (SFR)',
        'MR':  'Muscle Region (MR)',
        'PR':  'Prostate Region (PR)',
        'FR':  'Femur Region (FR)',
    }
    return full_name_map.get(abbr, abbr)  # Default to abbreviation if not found


def plot_all_iqms_vs_accs_vs_fovs_boxplot(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: Path,
    legend_fig_idx: int = 2,
    do_also_plot_individually=False,
    logger: logging.Logger = None
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

    df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    if logger:
        logger.info(f"Data types: {df.dtypes}")  # Debugging: Print data types

    # Generate individual plots
    if do_also_plot_individually:
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(data=df, x='acceleration', y=metric, hue='roi')
            plt.title(f'{metric.upper()} by Acceleration and FOV (Box Plot)', fontsize=16, fontweight='bold')
            plt.xlabel('Acceleration', fontsize=14)
            plt.ylabel(metric.upper(), fontsize=14)

            # Add sample size to legend
            handles, labels = ax.get_legend_handles_labels()
            sample_counts = df.groupby(['acceleration', 'roi']).size().unstack().fillna(0)

            # Construct new labels with sample sizes
            new_labels = [
                f'{get_full_name(label)} (n={sample_counts[label].sum():.0f})'
                for label in labels
            ]

            ax.legend(handles, new_labels, title='FOV', loc='upper right')

            plt.grid(True, linestyle='--', linewidth=0.7)
            individual_save_path = save_path.parent / f"{metric}_vs_accs_vs_fovs_boxplot.png"
            plt.savefig(individual_save_path, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"Saved box plot for {metric} by acceleration and FOV at {individual_save_path}")

    # Determine layout
    num_metrics = len(metrics)
    if num_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    elif num_metrics == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    elif num_metrics == 3:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    elif num_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    else:
        raise ValueError("Number of metrics must be 1, 2, 3, or 4")

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = sns.boxplot(data=df, x='acceleration', y=metric, hue='roi', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Acceleration and FOV', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Acceleration', fontsize=12)
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].grid(True, linestyle='--', linewidth=0.7)

        if i == legend_fig_idx:  # Place legend in the specified plot index
            handles, labels = ax.get_legend_handles_labels()
            sample_counts = df.groupby(['acceleration', 'roi']).size().unstack().fillna(0)

            # Construct new labels with sample sizes
            new_labels = [
                f'{get_full_name(label)} (n={sample_counts[label].sum():.0f})'
                for label in labels
            ]
            ax.legend(handles, new_labels, title='FOV', loc='upper right')
        else:
            ax.legend_.remove()  # Remove the legend from other plots

    # Remove any unused subplots
    if num_metrics < len(axes):
        for j in range(num_metrics, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    combined_save_path = save_path.parent / "all_iqms_vs_accs_vs_fovs_boxplot.png"
    plt.savefig(combined_save_path, bbox_inches='tight')
    plt.close()
    if logger:
        logger.info(f"Saved combined box plot for all IQMs by acceleration and FOV at {combined_save_path}")


def plot_all_iqms_vs_accs_vs_fovs_violinplot(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: Path,
    do_also_plot_individually=False,
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

    # Ensure the 'acceleration' and metric columns are numeric
    df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Debugging: Print the DataFrame head and data types
    print(df.head())
    print(df.dtypes)
    print(df.describe())  # Check the statistics of the data
    print(df['acceleration'].unique())
    print(df['roi'].unique())

    # Map FOV names
    fov_map = {
        'abfov': 'Abdominal FOV',
        'prfov': 'Prostate FOV',
        'lsfov': 'Lesion FOV'
    }
    df['roi'] = df['roi'].map(fov_map)
    
    # Generate individual plots
    if do_also_plot_individually:
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            ax = sns.violinplot(data=df, x='acceleration', y=metric, hue='roi', split=True)
            plt.title(f'{metric.upper()} by Acceleration and FOV (Violin Plot)', fontsize=16, fontweight='bold')
            plt.xlabel('Acceleration', fontsize=14)
            plt.ylabel(metric.upper(), fontsize=14)

            # Add sample size to legend
            handles, labels = ax.get_legend_handles_labels()
            sample_counts = df.groupby(['acceleration', 'roi']).size().unstack().fillna(0)
            new_labels = [f'{label} (n={int(sample_counts[col][acc])})' 
                          for col, label in zip(sample_counts.columns, labels) 
                          for acc in sample_counts.index]
            ax.legend(handles, new_labels, title='FOV', loc='upper right')

            plt.grid(True, linestyle='--', linewidth=0.7)
            individual_save_path = save_path.parent / f"{metric}_vs_accs_vs_fovs_violinplot.png"
            plt.savefig(individual_save_path, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"Saved violin plot for {metric} by acceleration and FOV at {individual_save_path}")

    # Create a 2x2 grid for the combined violin plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = sns.violinplot(data=df, x='acceleration', y=metric, hue='roi', split=True, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Acceleration and FOV', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Acceleration', fontsize=12)
        axes[i].set_ylabel(metric.upper(), fontsize=12)

        # Add sample size to legend
        handles, labels = ax.get_legend_handles_labels()
        sample_counts = df.groupby(['acceleration', 'roi']).size().unstack().fillna(0)
        new_labels = [f'{label} (n={int(sample_counts[col][acc])})' 
                      for col, label in zip(sample_counts.columns, labels) 
                      for acc in sample_counts.index]
        if ax.legend_ is not None:
            ax.legend_.remove()  # Remove the legend from individual plots

    # Create a single legend for the entire figure
    fig.legend(handles, new_labels, title='FOV', loc='upper right')

    # Remove any unused subplots
    if len(metrics) < 4:
        for j in range(len(metrics), 4):
            fig.delaxes(axes[j])

    plt.tight_layout()
    combined_save_path = save_path.parent / "all_iqms_vs_accs_vs_fovs_violinplot.png"
    plt.savefig(combined_save_path, bbox_inches='tight')
    plt.close()
    if logger:
        logger.info(f"Saved combined violin plot for all IQMs by acceleration and FOV at {combined_save_path}")