# MRI DLRecon FOV Analysis

![Violin Plot](path/to/your/violin_plot_image.png)

## Overview
This repository contains the code and data for analyzing a **Deep Learning Reconstruction (DLRecon)** model on accelerated k-space MRI data. The analysis focuses on different **Fields of View (FOVs)** and assesses visual quality metrics in various regions of the image.

## Project Description
We aim to evaluate the performance of a DLRecon model by computing visual quality metrics across different acceleration factors (R3 and R6) for 120 UCMG patients. The metrics include **SSIM**, **PSNR**, **RMSE**, and **HFEN**. The analysis is conducted both on the entire 3D volume and on lesion-specific 2D slices.

## Visual Quality Metrics
- **SSIM (Structural Similarity Index Measure)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **RMSE (Root Mean Squared Error)**
- **HFEN (High-Frequency Error Norm)**

## Acceleration Factors
- **R1**: Ground truth (3 averages with GRAPPA 2) Created with Root Sum of Squares (RSS)
- **R3**: 1 average out of 3
- **R6**: Half an average with GRAPPA 4

## Analysis
### 3D Volume Analysis
- **Description:** Compute visual quality metrics across the entire 3D volume.
- **Pros:** Provides an overall assessment of image quality.
- **Cons:** Averages out variations, potentially masking important slice-specific details.

### 2D Slice Analysis
- **Description:** Compute visual quality metrics on a slice-by-slice basis, focusing particularly on lesion-specific slices.
- **Pros:** Captures slice-specific variations, important for assessing lesion visibility and quality.
- **Cons:** May require more computational resources.

**Placeholder for further explanation and results on 3D vs. 2D analysis.**

## Results
### Violin Plot
- The violin plot below illustrates the distribution of SSIM, PSNR, RMSE, and HFEN for the 120 patients across acceleration factors R3 and R6.

![Violin Plot](path/to/your/violin_plot_image.png)

**Placeholder for detailed results and interpretation.**

## Installation
**Placeholder for installation instructions.**

## Usage
**Placeholder for usage instructions.**

## Data
**Placeholder for data description and how to access it.**

## Contributing
**Placeholder for contributing guidelines.**

## License
**Placeholder for license information.**

## Contact
**Placeholder for contact information.**

---

### Notes on 2D vs. 3D Analysis

It may make more sense to assess the visual quality in 2D rather than 3D for the following reasons:
- **Detailed Assessment**: 2D analysis allows for a more detailed assessment of each slice, which is crucial for identifying and evaluating lesions.
- **Variability**: 3D analysis averages the metrics across the entire volume, potentially masking significant variations that could be critical for diagnosis and treatment.
- **Computational Efficiency**: While 2D analysis may require more computational resources per slice, it provides a clearer picture of the image quality variations across different slices, especially in clinically significant regions.
