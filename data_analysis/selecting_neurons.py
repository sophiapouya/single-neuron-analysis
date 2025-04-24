from data_loading import * 
import os
import shutil
from stats import t_test

# Define folders to process and target directories
    # Folder with mat files
folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY/', 
'/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP/',
'/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC/', 
'/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA/']

#Folder for all neurons relevant to error analysis
target_dirs_error_all = {
    'amy': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/amy_all_neurons/',
    'hip': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/hip_all_neurons/',
    'acc': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/acc_all_neurons/',
    'sma': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/sma_all_neurons/'
}
# Folder for neurons identified as error neurons
target_dirs_error = {
    'amy': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/amy_error_neurons/',
    'hip': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/hip_error_neurons/',
    'acc': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/acc_error_neurons/',
    'sma': '/Users/sophiapouya/workspace/utsw/research_project/ERROR/sma_error_neurons/'
}
# Folder for all neurons relevant to conflict analysis
target_dirs_conflict_all = {
    'amy': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/amy_all_neurons/',
    'hip': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/hip_all_neurons/',
    'acc': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/acc_all_neurons/',
    'sma': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/sma_all_neurons/'
}
# Folder for neurons identified as conflict neurons
target_dirs_conflict = {
    'amy': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/amy_conflict_neurons/',
    'hip': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/hip_conflict_neurons/',
    'acc': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/acc_conflict_neurons/',
    'sma': '/Users/sophiapouya/workspace/utsw/research_project/CONFLICT/sma_conflict_neurons/'
}

# Loop through the specified folders and process files
for folder in folders:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # Ensure it's a .mat file and not a directory
        if os.path.isfile(file_path) and file.endswith('.mat'):
            data = load_data(file_path)  # Load the .mat file
            data, error, correctCongruent, correctIncongruent = bin_data(data)  # Bin the data
            average_bp = np.mean(np.array(data['spikeRatesBp']))
            average_stim = np.mean(np.array(data['spikeRatesStim']))
            
            # Skip if spikeRatesBp or spikeRatesStim is less than 0.5hz
            if average_bp <= 0.5 or average_stim <= 0.5:
                continue

            # Perform the t-test for the bp window
            _, ce_pv, _, ic_pv = t_test(
                correctCongruent['bpRasterSpikes'], 
                correctIncongruent['bpRasterSpikes'], 
                error['bpRasterSpikes']
            )
            # Perform the t-test for the stim window
            _, ce_pv2, _, ic_pv2 = t_test(
                correctCongruent['stimulusRasterSpikes'], 
                correctIncongruent['stimulusRasterSpikes'], 
                error['stimulusRasterSpikes']
            )
                
            # Error Neurons
            if len(error['trial'])>= 5:
                if 'AMY' in folder:
                    shutil.copy(file_path, target_dirs_error_all['amy'])
                    if ce_pv < 0.05 and ce_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_error['amy'])
                elif 'HIP' in folder:
                    shutil.copy(file_path, target_dirs_error_all['hip'])
                    if ce_pv < 0.05 and ce_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_error['hip'])
                elif 'ACC' in folder:
                    shutil.copy(file_path, target_dirs_error_all['acc'])
                    if ce_pv < 0.05 and ce_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_error['acc'])
                elif 'SMA' in folder:
                    shutil.copy(file_path, target_dirs_error_all['sma'])
                    if ce_pv < 0.05 and ce_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_error['sma'])

            # Conflict Neurons
            if len(correctCongruent['trial']) >= 5 and len(correctIncongruent['trial']) >= 5:
                if 'AMY' in folder:
                    shutil.copy(file_path, target_dirs_conflict_all['amy'])
                    if ic_pv < 0.05 and ic_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_conflict['amy'])
                elif 'HIP' in folder:
                    shutil.copy(file_path, target_dirs_conflict_all['hip'])    
                    if ic_pv < 0.05 and ic_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_conflict['hip'])            
                elif 'ACC' in folder:
                    shutil.copy(file_path, target_dirs_conflict_all['acc'])    
                    if ic_pv < 0.05 and ic_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_conflict['acc']) 
                elif 'SMA' in folder:
                    shutil.copy(file_path, target_dirs_conflict_all['sma'])    
                    if ic_pv < 0.05 and ic_pv2 < 0.05:
                        shutil.copy(file_path, target_dirs_conflict['sma']) 
