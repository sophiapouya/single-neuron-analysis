import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from data_loading import load_data, bin_data
from stats import t_test, roc_analysis
from plotting import plot_data
from permutation import bin_spikes
from scipy.stats import levene

##################################################################################
# Main section for testing
##################################################################################
def main():

#Below is code for outputting a single plot for a neuron
######################################################
    #define a specific mat file
    test_file='s61ch19ba1c1.mat'
    
    data = load_data(test_file)
    #bin the data
    data, error, correctCongruent, correctIncongruent = bin_data(data)
    # plot all the data and save to file in same directory 
    plot_data(data, error, correctCongruent, correctIncongruent, test_file, output_folder=None)

    # # Divide each spike count array by 2.0
    # correct_congruent_spike_counts = np.array(correctCongruent['spikeRatesBp'])
    # correct_incongruent_spike_counts = np.array(correctIncongruent['spikeRatesBp'])
    # error_spike_counts = np.array(error['spikeRatesBp'])

    # # Print the values to the screen
    # print("Correct Congruent Spike Counts (divided by 2):", correct_congruent_spike_counts)
    # print("Correct Incongruent Spike Counts (divided by 2):", correct_incongruent_spike_counts)
    # print("Error Spike Counts (divided by 2):", error_spike_counts)


#Below is code for outputting plots of all the neurons
######################################################
    # folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC/','/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA/']
    # outputFolders= ['/Users/sophiapouya/workspace/utsw/research_project/neuron_plots/acc_plots/','/Users/sophiapouya/workspace/utsw/research_project/neuron_plots/sma_plots/']
    # count = 0
    # for folder in folders:
    #     for file in os.listdir(folder):
    #         file_path = os.path.join(folder,file)
        
    #         # Process only if it's a file and a .mat file
    #         if os.path.isfile(file_path) and file.endswith('.mat'):
    #             data=load_data(file_path)
    #             data, error, correctCongruent, correctIncongruent = bin_data(data)
    #             plot_data(data, error, correctCongruent, correctIncongruent, file, outputFolders[count])
    #     count = count + 1

#Below is code for plotting AUC values for error/correct and congruent/incongruent neurons
######################################################

    # ce_data = ['/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/amy_error_neurons/','/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/hip_error_neurons/']
    # ci_data = ['/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_conflict_neurons/','/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_conflict_neurons/']
    # amy_ce_bp_auc, hip_ce_bp_auc, amy_ci_bp_auc, hip_ci_bp_auc = [], [], [], [] 
    # Correct/Error ROC AUC analysis
    # for data_folder in ce_data:        
    #     for file in os.listdir(data_folder):
    #         file_path = os.path.join(data_folder,file)
    #         raw_data = load_data(file_path)
    #         data = bin_spikes(raw_data)
    #         fpr, tpr, roc_auc = roc_analysis(data['bpSpikeCount'],data['labels'], time_window=1.0)
    #         count = count + 1
    #         print(count)
    #         if data_folder.__contains__('amy'):
    #             amy_ce_bp_auc.append(roc_auc)
    #         else:
    #             hip_ce_bp_auc.append(roc_auc)
    # #create figure and axis
    # fig, ax = plt.subplots()
    # # Plot the histograms
    # counts_amy, bins_amy, _ = ax.hist(amy_ce_bp_auc, bins=50, alpha=0.8, color='teal', label='AMY')
    # # Invert the HIP histogram
    # counts_hip, bins_hip = np.histogram(hip_ce_bp_auc, bins=150)
    # ax.bar(bins_hip[:-1], -counts_hip, width=bins_hip[1] - bins_hip[0], alpha=0.8, color='red', label='HIP')
    # ax.axhline(0, color='black', linewidth=1)  # Horizontal line at y=0
    # ax.set_xlabel('AUC of Error Neurons')
    # ax.set_ylabel('Number of Neurons') 
    # ax.legend()
    # plt.savefig('AUC_error_neurons.png')
    # # Display the plot
    # plt.show()

    
# Below is code for downselecting BP error and conflict neurons relevant for further analysis
# Conflict and error neurons selected with p-value < 0.05  
######################################################

    # # Define folders to process and target directories
    # # Folder with mat files
    # folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY/', '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP/']
    # # Folder for all neurons relevant to error analysis
    # target_dirs_error_all = {
    #     'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/amy_all_neurons/',
    #     'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/hip_all_neurons/'
    # }
    # # Folder for neurons identified as error neurons
    # target_dirs_error = {
    #     'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/amy_error_neurons/',
    #     'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/hip_error_neurons/'
    # }
    # # Folder for all neurons relevant to conflict analysis
    # target_dirs_conflict_all = {
    #     'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_all_neurons_bp/',
    #     'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_all_neurons_bp/'
    # }
    # # Folder for neurons identified as conflict neurons
    # target_dirs_conflict = {
    #     'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_conflict_neurons_bp/',
    #     'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_conflict_neurons_bp/'
    # }
    # folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC/', '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA/']

    # target_dirs_error_all = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/error_neurons/acc_all_neurons/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/error_neurons/sma_all_neurons/'
    # }
    # # Folder for neurons identified as error neurons
    # target_dirs_error = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/error_neurons/acc_error_neurons/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/error_neurons/sma_error_neurons/'
    # }
    # # Folder for all neurons relevant to conflict analysis
    # target_dirs_conflict_all = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/acc_all_neurons_bp/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/sma_all_neurons_bp/'
    # }
    # # Folder for neurons identified as conflict neurons
    # target_dirs_conflict = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/acc_conflict_neurons_bp/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/sma_conflict_neurons_bp/'
    # }

    # # Loop through the specified folders and process files
    # for folder in folders:
    #     for file in os.listdir(folder):
    #         file_path = os.path.join(folder, file)
            
    #         # Ensure it's a .mat file and not a directory
    #         if os.path.isfile(file_path) and file.endswith('.mat'):
    #             data = load_data(file_path)  # Load the .mat file
    #             data, error, correctCongruent, correctIncongruent = bin_data(data)  # Bin the data
    #             average_bp = np.mean(np.array(data['bpSpikeCount']))
                
    #             # Skip if spikeRatesBp is less than 0.5
    #             if average_bp <= 0.5:
    #                 continue
                
    #             # Perform the t-test for the bp window
    #             _, ce_pv, _, ic_pv = t_test(
    #                 correctCongruent['bpSpikeCount'], 
    #                 correctIncongruent['bpSpikeCount'], 
    #                 error['bpSpikeCount']
    #             )
                    
    #             # Error Neurons
    #             if len(error['trial'])>= 5:
    #                 if 'ACC' in folder:
    #                     shutil.copy(file_path, target_dirs_error_all['acc'])
    #                     if (ce_pv is not None) and ce_pv < 0.05:
    #                         shutil.copy(file_path, target_dirs_error['acc'])
    #                 elif 'SMA' in folder:
    #                     shutil.copy(file_path, target_dirs_error_all['sma'])
    #                     if ce_pv < 0.05:
    #                         shutil.copy(file_path, target_dirs_error['sma'])

    #             # Conflict Neurons
    #             if len(correctCongruent['trial']) >= 5 and len(correctIncongruent['trial']) >= 5:
    #                 if 'ACC' in folder:
    #                     shutil.copy(file_path, target_dirs_conflict_all['acc'])
    #                     if ic_pv < 0.05:
    #                         shutil.copy(file_path, target_dirs_conflict['acc'])
    #                 elif 'SMA' in folder:
    #                     shutil.copy(file_path, target_dirs_conflict_all['sma'])    
    #                     if ic_pv < 0.05:
    #                         shutil.copy(file_path, target_dirs_conflict['sma'])            

#Below is code for downselecting Stimulus Conflict neurons relevant for further analysis
# Conflict neurons selected with p-value < 0.05  
######################################################

    # # Define folders to process and target directories
    # # Folder with mat files
    
    # folders = ['/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_all_neurons/',
    #            '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_all_neurons/']
    # folders = ['/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/acc_all_neurons/',
    #            '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/sma_all_neurons/']


    # # Folder for neurons identified as conflict neurons
    # target_dirs_conflict = {
    #     'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_conflict_neurons_stim/',
    #     'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_conflict_neurons_stim/'
    # }
    # # Folder for neurons identified as conflict neurons
    # target_dirs_conflict = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/acc_conflict_neurons_stim/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/conflict_neurons/sma_conflict_neurons_stim/'
    # }

    # # Loop through the specified folders and process files
    # for folder in folders:
    #     for file in os.listdir(folder):
    #         file_path = os.path.join(folder, file)
            
    #         # Ensure it's a .mat file and not a directory
    #         if os.path.isfile(file_path) and file.endswith('.mat'):
    #             data = load_data(file_path)  # Load the .mat file
    #             data, error, correctCongruent, correctIncongruent = bin_data(data)  # Bin the data
    #             average_stim= np.mean(np.array(data['stimSpikeCount']))
                
    #             # Skip if spikeRatesStim is less than 0.5
    #             if average_stim <= 0.5:
    #                 continue
                
    #             if len(correctIncongruent['trial'])<5 or len(correctCongruent['trial'])<5:
    #                 continue

    #             # Perform the t-test for the stim window
    #             _, ce_pv, _, ic_pv = t_test(
    #                 correctCongruent['stimSpikeCount'], 
    #                 correctIncongruent['stimSpikeCount'], 
    #                 error['stimSpikeCount']
    #             )
                    
    #             # Conflict Neurons
    #             if 'acc' in folder:
    #                 if ic_pv < 0.05:
    #                     shutil.copy(file_path, target_dirs_conflict['acc'])
    #             elif 'sma' in folder:   
    #                 if ic_pv < 0.05:
    #                     shutil.copy(file_path, target_dirs_conflict['sma'])            

# Below is code for copying and identifying neurons for use in SVM region specific analysis
############################################################################################
    # # Define folders to process and target directories
    # # Folder with mat files
    # folders_error = ['../error_neurons/amy_all_neurons/', '../error_neurons/hip_all_neurons/']
    # folders_conflict = ['../conflict_neurons/amy_all_neurons/', '../conflict_neurons/hip_all_neurons/']
    # # Folder for all error neurons relevant to svm analysis
    # target_dirs_error = {
    #    'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/svm_error_neurons/amy/',
    #    'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/svm_error_neurons/hip/'
    #    }
    # # Folder for all conflict neurons relevant to svm analysis
    # target_dirs_conflict = {
    #    'amy': '/home/sophiapouya/workspace/utsw/sophia_research_project/svm_conflict_neurons/amy/',
    #    'hip': '/home/sophiapouya/workspace/utsw/sophia_research_project/svm_conflict_neurons/hip/'
    #    }

    # folders_error = ['../error_neurons/acc_all_neurons/', '../error_neurons/sma_all_neurons/']
    # folders_conflict = ['../conflict_neurons/acc_all_neurons/', '../conflict_neurons/sma_all_neurons/']
    # # Folder for all error neurons relevant to svm analysis
    # target_dirs_error = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/svm_error_neurons/acc/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/svm_error_neurons/sma/'
    # }
    # # Folder for all conflict neurons relevant to svm analysis
    # target_dirs_conflict = {
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/svm_conflict_neurons/acc/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/svm_conflict_neurons/sma/'
    # }

    # # Loop through the specified folders and process files
    # for folder in folders_conflict:
    #     for file in os.listdir(folder):
    #         file_path = os.path.join(folder, file)
    #         # Ensure it's a .mat file and not a directory
    #         if os.path.isfile(file_path) and file.endswith('.mat'):
    #             data = load_data(file_path)  # Load the .mat file
    #             data, error, correctCongruent, correctIncongruent = bin_data(data)  # Bin the data
                
    #             #Error trials-> make sure using correct folder (folders_error)
    #             if len(error['trial']) >= 10:
    #                 if 'acc' in folder:
    #                     shutil.copy(file_path, target_dirs_error['acc'])
    #                 elif 'sma' in folder:
    #                     shutil.copy(file_path, target_dirs_error['sma'])
                
    #             #Conflict Trials -> make sure using correct folder (folders_conflict)
    #             if len(correctCongruent['trial']) >= 10 and len(correctIncongruent['trial']) >= 10:
    #                 if 'acc' in folder:
    #                     shutil.copy(file_path, target_dirs_conflict['acc'])
    #                 elif 'sma' in folder:
    #                     shutil.copy(file_path, target_dirs_conflict['sma'])           

# # Below is code for copying and identifying neurons for use in SVM global analysis
# ############################################################################################
#     # Define folders to process and target directories
#     # Folder with mat files
#     folder = '/User/sophiapouya/workspace/utsw/channel_data/hip/'

#     # Folder for all neurons relevant to svm global analysis
#     target_dir_global_svm = '/Users/workspace/utsw/sophia_research_project/svm_global/hip_neurons/'

#     # Minimum required trials per trial type
#     min_trials_per_type = 10

#     # Loop through the specified folder and process files
#     for file in os.listdir(folder):
#         file_path = os.path.join(folder, file)

#         # Ensure it's a .mat file
#         if os.path.isfile(file_path) and file.endswith('.mat'):
#             print(f"Processing file: {file}")

#             # Load and process data
#             data = load_data(file_path)  # Load the .mat file
#             data, error, correctCongruent, correctIncongruent = bin_data(data)  # Bin the data

#             # Check trial type counts
#             trial_type_counts = [data['trialType'].count(tt) for tt in range(1, 10)]  # Count trials for each type
#             print(f"Trial type counts for {file}: {trial_type_counts}")

#             # Validate minimum trials for all trial types
#             if all(count >= min_trials_per_type for count in trial_type_counts):
#                 print(f"File {file} meets the minimum trial type requirements. Copying to SVM folder.")
#                 shutil.copy(file_path, target_dir_global_svm)  # Copy file to target folder


if __name__ == "__main__":
    main()