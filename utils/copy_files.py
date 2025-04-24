import os
import shutil

def copy_matching_mat_files(source_dir, mat_dir, target_dir):
    """
    Copies .mat files from mat_dir to target_dir if their base names match
    the base names of files (before '_Analysis') in the source_dir.
    
    Parameters:
    - source_dir: Directory containing the filenames (e.g., "*.png") to match.
    - mat_dir: Directory containing the .mat files.
    - target_dir: Directory to copy the matching .mat files to.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Extract base names from source directory
    file_bases = [
        f.split("_Analysis")[0] 
        for f in os.listdir(source_dir) 
        if f.endswith(".png")
    ]

    # Loop through the .mat files and copy if they match
    for mat_file in os.listdir(mat_dir):
        if mat_file.endswith(".mat"):
            mat_name = os.path.splitext(mat_file)[0]  # Base name without extension
            if mat_name in file_bases:
                src_path = os.path.join(mat_dir, mat_file)
                dest_path = os.path.join(target_dir, mat_file)
                shutil.copy(src_path, dest_path)
                print(f"Copied: {mat_file}")

    print("All matching .mat files have been copied.")

# Example Usage
if __name__ == "__main__":
    # Define your directories
    source_dir = "../neurons/amy_error_plots/"
    mat_dir = "../mat_files/amy_ce_bp_files/"
    target_dir = "../test/"

    # Call the function
    copy_matching_mat_files(source_dir, mat_dir, target_dir)
