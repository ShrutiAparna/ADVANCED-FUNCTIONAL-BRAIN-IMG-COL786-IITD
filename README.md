# ADVANCED-FUNCTIONAL-BRAIN-IMG-COL786-IITD
This repository is all about COL786 assignments

## Assignments

### Assignment 1 — Brain Anatomy, Atlas-Based Localization & fMRI Time Series

Explored structural and functional neuroimaging fundamentals using **FSL/FSLeyes**. Identified the major brain lobes and anatomical landmarks on **T1- and T2-weighted MRI scans**, and used the **Harvard–Oxford and Juelich atlases** to localize key functional regions including Broca’s area, Wernicke’s area, fusiform face area, amygdala, hippocampus, and thalamus. Also analyzed voxel-wise **BOLD time series** from a checkerboard fMRI experiment to understand temporal activation patterns.

### Assignment 2 — fMRI Preprocessing, Registration & Subject-Level Functional Analysis

Implemented a complete **fMRI preprocessing pipeline in FSL**, including brain extraction, motion correction, spatial smoothing, temporal filtering, and linear registration to **MNI standard space**. Performed subject-level GLM analysis for motor-task contrasts involving left- and right-hand movement under audio and visual stimuli. Conjunction analysis highlighted activation in the **precentral gyrus**, consistent with expected motor cortex involvement.

### Assignment 3 — Group-Level fMRI Analysis Across Subjects

Extended the analysis to **higher-level FEAT group analysis** across 30 subjects. Studied average activation patterns using motor contrasts from Assignment 2 and performed conjunction analysis to identify regions consistently activated across participants. Results showed strong group-level activation in the **precentral/postcentral gyri**, cerebellum, and associated motor regions for both left- and right-hand movement.

### Assignment 4 — Custom GLM and Group Analysis Implementation

Developed custom implementations for **single-subject and group-level fMRI GLM analysis** and compared the outputs with FSL. Evaluated parameter estimates, contrast estimates, and statistical maps through correlation analysis and visual comparison. The custom implementation showed good agreement with FSL while also highlighting differences in scaling and statistical estimation methods.

## Course Project — Image Reconstruction Using fMRI Data

Worked on a course project focused on **reconstructing visual images from fMRI brain activity** using deep learning–based decoding pipelines. Studied and replicated existing methods including **Shen et al. (2019)** and **Kamitani Lab’s Deep Image Reconstruction**, then explored improvements through **semantic-guided reconstruction** inspired by recent CLIP + VQGAN based approaches.

A major contribution of the project was improving reconstructed image quality using **post-processing denoising techniques** such as **Total Variation filtering, Richardson–Lucy deconvolution, Non-Local Means, and BM3D**. The reconstructed outputs were evaluated using **SSIM, PCC, LPIPS, and MSE**, where **Total Variation filtering consistently gave the best balance between structural preservation and noise reduction**.

The project provided practical experience with **fMRI-based brain decoding, feature extraction using VGG/CLIP, generative image reconstruction using VQGAN, and evaluation of perceptual image quality**, while also highlighting challenges such as noisy BOLD signals, limited subject data, and computational constraints in large-scale neural decoding pipelines.
