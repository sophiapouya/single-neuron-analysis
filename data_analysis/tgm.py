import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from scipy.stats import percentileofscore
from scipy.ndimage import label
from svm import process_data  # Ensure this is the correct import

def temporal_generalization(all_neurons_data, midpoints, iterations, analysis_type='error_correct', 
                            validation_method="None", n_folds=5, n_permutations=50):
    """
    Compute Temporal Generalization Matrix (TGM) with Cluster-Based Permutation Testing.
    """
    num_bins = len(midpoints)
    labels = np.array([0] * 10 + [1] * 10)  # 10 Correct, 10 Error trials
    tgm = np.zeros((num_bins, num_bins))  # Store average accuracy across iterations
    shuffled_tgm_distribution = np.zeros((n_permutations, num_bins, num_bins))  # Store shuffled results

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    # **Iterate over 50 iterations to get stable accuracy estimates**
    for _ in range(iterations):
        feature_matrices = []  # Reset feature matrices each iteration

        # Construct new feature matrices for each time bin
        for time_idx in range(num_bins):
            feature_matrix = []
            for neuron, spike_data in all_neurons_data.items():
                if analysis_type == 'error_correct':
                    group1_trials = spike_data['correct'][:, time_idx]
                    group2_trials = spike_data['error'][:, time_idx]
                elif analysis_type == 'congruence':
                    group1_trials = spike_data['correct_congruent'][:, time_idx]
                    group2_trials = spike_data['correct_incongruent'][:, time_idx]

                # Sample 10 trials per group
                sampled_group1 = np.random.choice(group1_trials, 10, replace=False)
                sampled_group2 = np.random.choice(group2_trials, 10, replace=False)

                trials_for_neuron = np.concatenate([sampled_group1, sampled_group2])
                feature_matrix.append(trials_for_neuron)

            feature_matrices.append(np.array(feature_matrix).T)  # Shape: (20 trials, num_neurons)

        # **Compute TGM with Stratified K-Fold Cross-Validation**
        for train_idx in range(num_bins):
            fold_accuracies = np.zeros(num_bins)  # Store fold accuracies for averaging

            for train_indices, test_indices in skf.split(feature_matrices[train_idx], labels):
                svm_model = SVC(kernel='linear')
                svm_model.fit(feature_matrices[train_idx][train_indices], labels[train_indices])

                for test_idx in range(num_bins):
                    predictions = svm_model.predict(feature_matrices[test_idx][test_indices])
                    fold_accuracy = np.mean(predictions == labels[test_indices])
                    fold_accuracies[test_idx] += fold_accuracy

            tgm[train_idx, :] += fold_accuracies / n_folds  # Accumulate across iterations

    tgm /= iterations  # Average over iterations

    # **Cluster-Based Permutation Testing**
    if validation_method == "shuffled_labels":
        for perm in range(n_permutations):
            shuffled_labels = shuffle(labels, random_state=perm)  # Shuffle labels once per permutation

            for _ in range(iterations):  # Permutation should be iterated over the same number of times
                feature_matrices = []  # Reset for permutation iteration

                # Construct feature matrices again
                for time_idx in range(num_bins):
                    feature_matrix = []
                    for neuron, spike_data in all_neurons_data.items():
                        if analysis_type == 'error_correct':
                            group1_trials = spike_data['correct'][:, time_idx]
                            group2_trials = spike_data['error'][:, time_idx]
                        elif analysis_type == 'congruence':
                            group1_trials = spike_data['correct_congruent'][:, time_idx]
                            group2_trials = spike_data['correct_incongruent'][:, time_idx]

                        sampled_group1 = np.random.choice(group1_trials, 10, replace=False)
                        sampled_group2 = np.random.choice(group2_trials, 10, replace=False)

                        trials_for_neuron = np.concatenate([sampled_group1, sampled_group2])
                        feature_matrix.append(trials_for_neuron)

                    feature_matrices.append(np.array(feature_matrix).T)  # Shape: (20 trials, num_neurons)

                # Compute shuffled TGM
                for train_idx in range(num_bins):
                    fold_accuracies = np.zeros(num_bins)

                    for train_indices, test_indices in skf.split(feature_matrices[train_idx], shuffled_labels):
                        svm_model = SVC(kernel='linear')
                        svm_model.fit(feature_matrices[train_idx][train_indices], shuffled_labels[train_indices])

                        for test_idx in range(num_bins):
                            predictions = svm_model.predict(feature_matrices[test_idx][test_indices])
                            fold_accuracy = np.mean(predictions == shuffled_labels[test_indices])
                            fold_accuracies[test_idx] += fold_accuracy

                    shuffled_tgm_distribution[perm, train_idx, :] += fold_accuracies / n_folds

        # Compute significance threshold (95th percentile of shuffled data)
        threshold_tgm = np.percentile(shuffled_tgm_distribution, 95, axis=0)
        significant_mask = tgm > threshold_tgm

        return tgm, significant_mask

    return tgm, None


def plot_tgm(tgm, midpoints, output_file, significant_mask=None):
    """
    Plot the temporal generalization matrix with a custom color gradient and capped accuracy.
    """
    time_ms = midpoints / 1_000  # Convert microseconds to milliseconds

    # Define the custom colormap from the new gradient
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "cyan", "yellow", "red"]
    )

    # Define the bounds and normalization to cap at 0.8
    bounds = np.linspace(0.45, 0.8, 100)  # Gradient from 0.45 to 0.8
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Plot the TGM
    plt.figure(figsize=(10, 8))
    plt.imshow(
        tgm, aspect='auto', origin='lower', cmap=custom_cmap, norm=norm,
        extent=[time_ms[0], time_ms[-1], time_ms[0], time_ms[-1]]
    )
    cbar = plt.colorbar(label='Accuracy', ticks=[0.45, 0.5, 0.6, 0.7, 0.8])
    cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in [0.45, 0.5, 0.6, 0.7, 0.8]])  # Format tick labels

    # Overlay significance contours if available
    if significant_mask is not None:
        contour_levels = [0.5]  # Contour where mask is 1 (significant)
        plt.contour(significant_mask, levels=contour_levels, colors='black', linestyles='dashed', 
                    extent=[time_ms[0], time_ms[-1], time_ms[0], time_ms[-1]])

    # Add labels and title
    plt.xlabel('Testing Time (ms)')
    plt.ylabel('Training Time (ms)')
    plt.title('Temporal Generalization Matrix')

    # Save and show the figure
    plt.savefig(output_file)
    plt.show()

# Calling TGM
########################################################################################################
directory_path_error_hip = "/Users/sophiapouya/workspace/utsw/research_project/svm_error_neurons/hip"
bin_size = 400  # ms bins
step_size = 20  # ms steps
iterations = 50  # Number of iterations for averaging
event_type = 'button_press'  # 'button_press' or 'stimulus'
analysis_type = 'error_correct' # 'error_correct' or 'congruence'
output_file = 'hip_error_tgm_bp.png'
output_file_permuted = 'hip_error_tgm_bp_cluster_permutation.png'


# Process the data
all_neurons_data_hip_error, midpoints_hip_error = process_data(directory_path_error_hip, bin_size, step_size, event_type)

# Generate Temporal Generalization Matrix
# tgm_hip_error = temporal_generalization(all_neurons_data_hip_error, midpoints_hip_error, iterations, analysis_type)

# Without permutation testing
tgm, _ = temporal_generalization(all_neurons_data_hip_error, midpoints_hip_error, iterations, analysis_type='error_correct')

# Plot results (handle significance if available)
plot_tgm(tgm, midpoints_hip_error, output_file=output_file)

# # With permutation testing
# tgm, significant_mask = temporal_generalization(all_neurons_data_hip_error, midpoints_hip_error, iterations, analysis_type='error_correct', validation_method="shuffled_labels")

# # Plot results (handle significance if available)
# plot_tgm(tgm, midpoints_hip_error, output_file=output_file_permuted, significant_mask=significant_mask)

