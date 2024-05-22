# mri-dl-recon-fov-analysis
Analyze a DLRecon model on accelerated k-space MRI data across different FOVs, assessing visual quality metrics in various image regions.



# Introduction
## Purpose
To analyse the different visual quality metrics on different fields of view on varying image regions on prostate MRI.
To shed light on the correlation of lesion visual quality retention versus non-lesion visual quality retention in terms of SSIM, PSNR, VIF and HFEN.


# Methodology
## Dataset
We have the NYU publically available dataset N=312 k-spaces.

## The AI Reconstruction model
We have a trained vSHARP reconstruction model. That iteratively updates the missing data in k-space to reduce error in kspace and optimize the visual quality in image space.

## Evalutation Metrics
For this study we will evaluate the computed visual quality with metrics such as SSIM, PSNR, VIF and HFEN. These are common metrics in the field of visual quality analysis and are also used to optimize the reconstruction models.

## Fields of view.
Results will be analyzed on multiple fields of view (FOVs). We will consider de full FOV (including air around the body), the prostate region and the lesion FOV.
1. full         (full transversal view of the entire abdomen, including air around the body)
2. prostatea    (zoomed in around the prostate, bladder and surrounding structures. This fov is identical to a clinical DICOM scan)
3. lesion       (10 pixel boundary FOV around a lesion.)


# Results
![Description of the image](figures/all_iqms_vs_accs_violin_v2.png)


# Discussion


# Conclusion