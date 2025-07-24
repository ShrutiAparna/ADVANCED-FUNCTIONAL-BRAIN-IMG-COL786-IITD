#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import argparse

def register_to_standard(input_dir, output_dir, standard_brain, contrast_type='cope'):
    """
    Register contrast files to standard space using FLIRT with existing transformation matrices.
    
    Parameters:
    input_dir (str): Directory containing first-level analysis results
    output_dir (str): Directory to save registered files
    standard_brain (str): Path to standard brain template
    contrast_type (str): Type of contrast files to register (cope, zstat, etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define subject numbers (1-32, omitting 15 and 26)
    subject_numbers = [n for n in range(1, 33) if n not in [15, 26]]
    
    # Track successful registrations
    registered_files = []
    
    for subject_num in subject_numbers:
        subject_id = f"sub-{subject_num:02d}"
        print(f"Processing {subject_id}...")
        
        # Find contrast files for this subject
        contrast_pattern = os.path.join(input_dir, subject_id, f"*{contrast_type}*.nii.gz")
        contrast_files = glob.glob(contrast_pattern)
        
        if not contrast_files:
            print(f"  No {contrast_type} files found for {subject_id}")
            continue
        
        # Find transformation matrix for this subject
        transform_pattern = os.path.join(input_dir, subject_id, "*.mat")
        transform_files = glob.glob(transform_pattern)
        
        if not transform_files:
            print(f"  No transformation matrix found for {subject_id}")
            continue
        
        # Use the first transformation matrix found
        transform_file = transform_files[0]
        
        # Register each contrast file
        for contrast_file in contrast_files:
            contrast_name = os.path.basename(contrast_file)
            output_file = os.path.join(output_dir, f"{subject_id}_{contrast_name}")
            
            # Skip if output file already exists
            if os.path.exists(output_file):
                print(f"  {output_file} already exists, skipping.")
                registered_files.append(output_file)
                continue
            
            # Run FLIRT command
            cmd = [
                "flirt",
                "-in", contrast_file,
                "-ref", standard_brain,
                "-applyxfm",
                "-init", transform_file,
                "-out", output_file
            ]
            
            print(f"  Registering {contrast_name}...")
            try:
                subprocess.run(cmd, check=True)
                registered_files.append(output_file)
                print(f"  Registered to {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"  Error registering {contrast_name}: {e}")
    
    return registered_files

def create_file_list(registered_files, output_file, contrast_num=None):
    """
    Create a text file listing all registered files for a specific contrast.
    
    Parameters:
    registered_files (list): List of all registered files
    output_file (str): Path to output text file
    contrast_num (int, optional): Contrast number to filter by
    """
    # Filter by contrast number if specified
    if contrast_num is not None:
        filtered_files = [f for f in registered_files if f".cope{contrast_num}" in f or f".zstat{contrast_num}" in f]
    else:
        filtered_files = registered_files
    
    # Write file list
    with open(output_file, 'w') as f:
        for file_path in filtered_files:
            f.write(f"{file_path}\n")
    
    print(f"Created file list with {len(filtered_files)} files: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare files for fMRI group analysis')
    parser.add_argument('--input-dir', required=True, help='Directory containing first-level results')
    parser.add_argument('--output-dir', required=True, help='Directory to save registered files')
    parser.add_argument('--standard', required=True, help='Path to standard brain template')
    parser.add_argument('--contrast-type', default='cope', help='Type of contrast files (cope, zstat, etc.)')
    parser.add_argument('--contrasts', type=int, nargs='+', help='Contrast numbers to create file lists for')
    
    args = parser.parse_args()
    
    # Register files to standard space
    registered_files = register_to_standard(
        args.input_dir, 
        args.output_dir,
        args.standard,
        args.contrast_type
    )
    
    # Create overall file list
    all_list_path = os.path.join(args.output_dir, f"all_{args.contrast_type}_files.txt")
    create_file_list(registered_files, all_list_path)
    
    # Create contrast-specific file lists if requested
    if args.contrasts:
        for contrast_num in args.contrasts:
            contrast_list_path = os.path.join(args.output_dir, f"{args.contrast_type}{contrast_num}_files.txt")
            create_file_list(registered_files, contrast_list_path, contrast_num)

if __name__ == "__main__":
    main()
