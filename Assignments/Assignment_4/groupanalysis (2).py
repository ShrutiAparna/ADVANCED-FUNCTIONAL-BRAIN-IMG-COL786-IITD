#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import os
import sys
from scipy import stats
import time

def load_files(file_list_path):
    """
    Load all files specified in the file list.
    
    Parameters:
    file_list_path (str): Path to text file containing list of file paths
    
    Returns:
    list: List of NIfTI image objects
    """
    with open(file_list_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    
    if not file_paths:
        raise ValueError("File list is empty or could not be read.")
    
    print(f"Loading {len(file_paths)} files...")
    
    images = []
    for path in file_paths:
        try:
            img = nib.load(path)
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            sys.exit(1)
    
    # Verify all images have the same dimensions
    shapes = [img.shape for img in images]
    if len(set(shapes)) > 1:
        print("Error: Images have different dimensions:")
        for i, path in enumerate(file_paths):
            print(f"  {path}: {shapes[i]}")
        sys.exit(1)
    
    return images

def perform_group_analysis(images):
    """
    Perform group analysis using one-sample t-test at each voxel.
    
    Parameters:
    images (list): List of NIfTI image objects
    
    Returns:
    tuple: (t_stat_data, z_stat_data, degrees_of_freedom)
    """
    # Get data from all images
    data_list = [img.get_fdata() for img in images]
    
    # Stack data for analysis
    data_stack = np.stack(data_list, axis=-1)  # Last dimension is now subjects
    
    # Get dimensions
    shape = data_stack.shape[:-1]  # Shape without subjects dimension
    num_subjects = data_stack.shape[-1]
    
    print(f"Performing group analysis on {num_subjects} subjects...")
    print(f"Data dimensions: {shape}")
    
    # Pre-allocate arrays
    t_stat_data = np.zeros(shape)
    p_value_data = np.zeros(shape)
    
    # Calculate degrees of freedom
    df = num_subjects - 1
    
    # Track progress
    total_voxels = np.prod(shape)
    start_time = time.time()
    last_update = start_time
    voxels_processed = 0
    
    # Perform t-test at each voxel
    # Reshape data to work with 1D arrays for efficiency
    reshaped_data = data_stack.reshape(-1, num_subjects)
    
    # Optimize by vectorizing the t-test calculation
    # Calculate mean across subjects
    means = np.mean(reshaped_data, axis=1)
    
    # Calculate standard deviation across subjects
    stds = np.std(reshaped_data, axis=1, ddof=1)  # ddof=1 for sample std
    
    # Calculate standard error
    se = stds / np.sqrt(num_subjects)
    
    # Calculate t-statistic
    with np.errstate(divide='ignore', invalid='ignore'):
        t_values = means / se
    
    # Handle NaN and Inf values
    t_values = np.nan_to_num(t_values, nan=0, posinf=0, neginf=0)
    
    # Reshape back to original dimensions
    t_stat_data = t_values.reshape(shape)
    
    # Convert t-statistics to z-statistics
    # We use the survival function (1 - cdf) of the t-distribution
    # to get the p-value, then use the inverse of the standard normal
    # to convert to z-score
    #p_values = stats.t.sf(np.abs(t_values), df) * 2  # Two-tailed p-value
    #z_values = stats.norm.isf(p_values)
    # A more FSL-like t-to-z conversion
    
    # Handle NaN and Inf values in z_values
    #z_values = np.nan_to_num(z_values, nan=0, posinf=0, neginf=0)
    
    # Reshape z-values to original dimensions
    #z_stat_data = z_values.reshape(shape)
    # Two-tailed p-values
    p_values = stats.t.sf(np.abs(t_values), df) * 2

    # Convert to z-values
    z_values = stats.norm.isf(p_values)

    # FSL-like: Restore the original sign
    z_values_signed = np.sign(t_values) * z_values

    # Handle NaN and Inf values in z_values
    z_values_signed = np.nan_to_num(z_values_signed, nan=0, posinf=0, neginf=0)

    # Reshape z-values to original dimensions
    z_stat_data = z_values_signed.reshape(shape)

    
    print(f"Group analysis completed in {time.time() - start_time:.2f} seconds.")
    
    return t_stat_data, z_stat_data, df

def save_results(t_stat_data, z_stat_data, output_prefix, reference_img):
    """
    Save t-statistics and z-statistics as NIfTI files.
    
    Parameters:
    t_stat_data (array): t-statistic values
    z_stat_data (array): z-statistic values
    output_prefix (str): Prefix for output files
    reference_img (NIfTI): Reference image for header information
    """
    # Create t-statistic image
    t_stat_img = nib.Nifti1Image(t_stat_data, reference_img.affine, reference_img.header)
    t_stat_path = f"{output_prefix}.tstat.nii.gz"
    nib.save(t_stat_img, t_stat_path)
    print(f"Saved t-statistics to {t_stat_path}")
    
    # Create z-statistic image
    z_stat_img = nib.Nifti1Image(z_stat_data, reference_img.affine, reference_img.header)
    z_stat_path = f"{output_prefix}.zstat.nii.gz"
    nib.save(z_stat_img, z_stat_path)
    print(f"Saved z-statistics to {z_stat_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python group_analysis.py <file_list.txt> <output_prefix>")
        sys.exit(1)
    
    file_list_path = sys.argv[1]
    output_prefix = sys.argv[2]
    
    print(f"Group analysis tool")
    print(f"File list: {file_list_path}")
    print(f"Output prefix: {output_prefix}")
    
    # Load files
    images = load_files(file_list_path)
    
    # Perform group analysis
    t_stat_data, z_stat_data, df = perform_group_analysis(images)
    
    # Save results using the first image as reference
    save_results(t_stat_data, z_stat_data, output_prefix, images[0])
    
    print(f"Analysis complete with {len(images)} subjects and {df} degrees of freedom.")

if __name__ == "__main__":
    main()
