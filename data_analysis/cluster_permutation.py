from personal_plots import get_trial_data, plot
from scipy.io import loadmat
#perform t test at every time point across the train -1s to 2.8s
from scipy.stats import ttest_ind
import numpy as np
import os
from joblib import Parallel, delayed

def cluster_permutation_test(error_trials, correct_trials, n_permutations=1000, threshold=2.0, n_jobs=-1):

    # Get basic shapes
    n_timepoints = error_trials.shape[1]
    n_error = error_trials.shape[0]
    n_correct = correct_trials.shape[0]
    total_trials = n_error + n_correct
    
    # Compute observed t-values vectorized along time (axis=0)
    observed_t, _ = ttest_ind(error_trials, correct_trials, axis=0)
    
    # Identify clusters in the observed data
    clusters = []
    in_cluster = False
    for t in range(n_timepoints):
        if observed_t[t] > threshold:
            if not in_cluster:
                cluster_start = t
                in_cluster = True
        else:
            if in_cluster:
                cluster_end = t
                cluster_stat = np.sum(observed_t[cluster_start:cluster_end])
                clusters.append((cluster_start, cluster_end, cluster_stat))
                in_cluster = False
    if in_cluster:
        cluster_stat = np.sum(observed_t[cluster_start:])
        clusters.append((cluster_start, n_timepoints, cluster_stat))
    
    # Prepare combined data for permutation testing
    combined_data = np.concatenate([error_trials, correct_trials], axis=0)
    
    # Define a helper function for one permutation iteration
    def permutation_iteration(_):
        permuted_indices = np.random.permutation(total_trials)
        perm_error = combined_data[permuted_indices[:n_error], :]
        perm_correct = combined_data[permuted_indices[n_error:], :]
        # Compute t-values vectorized along time axis
        perm_t, _ = ttest_ind(perm_error, perm_correct, axis=0)
        perm_cluster_stats = []
        in_cluster_perm = False
        for t in range(n_timepoints):
            if perm_t[t] > threshold:
                if not in_cluster_perm:
                    cluster_start_perm = t
                    in_cluster_perm = True
            else:
                if in_cluster_perm:
                    cluster_end_perm = t
                    cluster_stat_perm = np.sum(perm_t[cluster_start_perm:cluster_end_perm])
                    perm_cluster_stats.append(cluster_stat_perm)
                    in_cluster_perm = False
        if in_cluster_perm:
            cluster_stat_perm = np.sum(perm_t[cluster_start_perm:])
            perm_cluster_stats.append(cluster_stat_perm)
        return max(perm_cluster_stats) if perm_cluster_stats else 0

    # Run permutations in parallel
    max_cluster_stats = Parallel(n_jobs=n_jobs)(delayed(permutation_iteration)(i) for i in range(n_permutations))
    max_cluster_stats = np.array(max_cluster_stats)
    
    # Compute p-values for each observed cluster
    p_values = [np.mean(max_cluster_stats >= cluster_stat) for (_, _, cluster_stat) in clusters]
    
    return observed_t, clusters, p_values

def main():
    #single plot example
    # test_file='s61ch19ba1c1.mat'
    # data = loadmat(test_file)
    # trial_data = get_trial_data(data=data, start_offset=-1.5*1e6, end_offset=3.5*1e6)
    # error_trials = np.array([trial["moving_avg"] for trial in trial_data["error"]])
    # correct_trials = np.array([trial["moving_avg"] for trial in trial_data["correct"]])
    # observed_t, clusters, p_values= cluster_permutation_test(error_trials=error_trials, correct_trials=correct_trials)
    # plot(trial_data, file_name=test_file, plot_mode="error_correct", output_dir='/Users/sophiapouya/workspace/utsw/research_project/', clusters=clusters, cluster_p_values=p_values)
    
    # Cluster for error correct analysis pngs

    folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY/', 
    '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP/',
    '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC/', 
    '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA/',
    '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC/']
    # output_dir = '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/OFC_stats/'
    # neuron_folder = folders[0]
    target_output_dirs = {
        'amy': '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/AMY_stats/',
        'hip': '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/HIP_stats/',
        'acc': '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/ACC_stats/',
        'sma': '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/SMA_stats/',
        'ofc': '/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/OFC_stats/'
    }

    label=''
    for folder in folders: 
        # Counters to track neurons
        total_neurons_relevant = 0
        neurons_with_significant_clusters = 0
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            # Ensure it's a .mat file and not a directory
            if os.path.isfile(file_path) and file.endswith('.mat'):
                data = loadmat(file_path)
                trial_data = get_trial_data(data=data, start_offset=-1.5*1e6, end_offset=3.5*1e6)
                
                # Only perform analysis if there are at least 10 error trials
                if len(trial_data['incongruent']) >= 10 and len(trial_data['congruent'])>=10:
                    total_neurons_relevant += 1
                    
                    error_trials = np.array([trial["moving_avg"] for trial in trial_data["incongruent"]])
                    correct_trials = np.array([trial["moving_avg"] for trial in trial_data["congruent"]])
                    
                    observed_t, clusters, p_values = cluster_permutation_test(
                        error_trials=error_trials, 
                        correct_trials=correct_trials
                    )
                    
                    # Check if any cluster in this neuron is significant (p < 0.05)
                    if any(p < 0.05 for p in p_values):
                        neurons_with_significant_clusters += 1
                    
                    # Use the current file name as identifier when plotting
                    if 'AMY' in folder: 
                        output_dir = target_output_dirs['amy']
                        label = "AMY"
                    elif 'HIP' in folder: 
                        output_dir = target_output_dirs['hip']
                        label="HIP"
                    elif 'ACC' in folder: 
                        output_dir = target_output_dirs['acc']
                        label="ACC"
                    elif 'SMA' in folder: 
                        output_dir = target_output_dirs['sma']
                        label="SMA"
                    elif 'OFC' in folder:
                        output_dir = target_output_dirs['ofc']
                        label="OFC"

                    plot(trial_data, file_name=file, plot_mode="congruent_incongruent", output_dir=output_dir, 
                        clusters=clusters, cluster_p_values=p_values)
                else:
                    continue

        print("Region: ", label)
        print("Total neurons relevant for analysis:", total_neurons_relevant)
        print("Neurons with at least one significant cluster (p < 0.05):", neurons_with_significant_clusters)

if __name__ == "__main__":
    main()