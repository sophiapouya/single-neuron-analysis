import os
import shutil

def match_and_copy_files(plot_dir, data_dir, target_dir):
    """
    Finds files in the plot directory with the format "neuronName_Analysis.png" 
    that match files in the data directory in the format "neuronName.mat". 
    Copies matching files to the target directory.
    
    Parameters:
    - plot_dir: Directory containing the plot files (e.g., "neuronName_Analysis.png").
    - data_dir: Directory containing the data files (e.g., "neuronName.mat").
    - target_dir: Directory where matching plot files will be copied.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Extract base names (without extensions) of all .mat files in the data directory
    neuron_names = {
        os.path.splitext(file)[0] for file in os.listdir(data_dir) if file.endswith('.mat')
    }

    # Iterate through files in the plot directory
    for file in os.listdir(plot_dir):
        if file.endswith('_Analysis.png'):
            # Extract the base neuron name from the plot file
            neuron_name = file.split('_Analysis.png')[0]
            
            # Check if the neuron name matches any in the data directory
            if neuron_name in neuron_names:
                # Copy the file to the target directory
                src = os.path.join(plot_dir, file)
                dst = os.path.join(target_dir, file)
                shutil.copy(src, dst)
                print(f"Copied: {file} to {target_dir}")

# Example Usage
plot_directory = '../plots/amy_plots'
data_directory = '../neurons/amy_error_neurons'
target_directory = '../neurons/amy_error_plots'

match_and_copy_files(plot_directory, data_directory, target_directory)
