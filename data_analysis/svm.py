import numpy as np
import os
import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


def process_data(directory_path, bin_size, step_size, event_type='button_press'):
    """
    Processes the data and bins spikes for each trial.
    
    Parameters:
        directory_path (str): Path to the data directory.
        bin_size (int): Size of the bin in ms.
        step_size (int): Step size between bins in ms.
        event_type (str): Event to align to ('button_press' or 'stimulus').
    
    Returns:
        dict: Binned spike data for all neurons.
        np.array: Midpoints for bins.
    """
    midpoints = np.arange(-500 * 1_000, (1500 * 1_000) + step_size * 1_000, step_size * 1_000)
    midpoints_baseline= np.arange(-1000 * 1_000, step_size * 1_000, step_size * 1_000)
    all_neurons_data = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")
            data = scipy.io.loadmat(file_path)

            # Extract data
            answers = data['answers']
            colors_presented = data['colorsPresented']
            texts_presented = data['textsPresented']
            events = data['events']
            timestamps = data['timestampsOfCell']

            # Choose the event times based on event_type
            if event_type == 'button_press':
                event_times = events[:, 1]
            elif event_type == 'stimulus':
                event_times = events[:, 0]
            else:
                raise ValueError("Invalid event_type. Choose 'button_press' or 'stimulus'.")
            
            # Define for baseline window
            stimulus_times = events[:,0]

            outcomes = np.where(answers == colors_presented, 0, 1)
            congruences = [
                1 if answers[i] == colors_presented[i] and colors_presented[i] == texts_presented[i] else
                0 if answers[i] == colors_presented[i] else None
                for i in range(len(answers))
            ]

            correct_congruent_trials = []
            correct_incongruent_trials = []
            error_trials = []
            correct_trials = []
            all_baseline_spike_counts = []
            all_spike_counts = []

            for i in range(len(answers)):
                spike_counts = []
                baseline_spike_counts = []
                for midpoint in midpoints:
                    bin_start = event_times[i] + midpoint - 0.5 * bin_size * 1_000
                    bin_end = event_times[i] + midpoint + 0.5 * bin_size * 1_000
                    spike_count = np.sum((timestamps >= bin_start) & (timestamps < bin_end))
                    spike_counts.append(spike_count)

                for midpoint in midpoints_baseline:
                    bin_start = stimulus_times[i] + midpoint - 0.5 * bin_size * 1_000
                    bin_end = stimulus_times[i] + midpoint + 0.5 * bin_size * 1_000
                    baseline_spike_count = np.sum((timestamps >= bin_start) & (timestamps < bin_end))
                    baseline_spike_counts.append(baseline_spike_count)

                all_baseline_spike_counts.append(baseline_spike_counts)
                all_spike_counts.append(spike_counts)

                if outcomes[i] == 0:
                    correct_trials.append(spike_counts)
                    if congruences[i] == 1:
                        correct_congruent_trials.append(spike_counts)
                    else:
                        correct_incongruent_trials.append(spike_counts)
                else:
                    error_trials.append(spike_counts)

            all_baseline_spike_counts = np.array(all_baseline_spike_counts)
            all_spike_counts = np.array(all_spike_counts)

            baseline_mean = np.mean(all_baseline_spike_counts)
            baseline_std = np.std(all_baseline_spike_counts, ddof=1)  # Single value

            # Avoid division by zero
            if baseline_std == 0:
                baseline_std = 1

            z_scored_spike_counts = (all_spike_counts - baseline_mean) / baseline_std

            all_neurons_data[filename.replace('.mat', '')] = {
                'correct_congruent': np.array(correct_congruent_trials),
                'correct_incongruent': np.array(correct_incongruent_trials),
                'error': np.array(error_trials),
                'correct': np.array(correct_trials)
            }
            
            print("Shape of all_spike_counts:", all_spike_counts.shape)  # (n_trials, 101)
            print("Baseline Mean:", baseline_mean)  # Should be a single number
            print("Baseline Std Dev:", baseline_std)  # Should be a single number
            print("Shape of z_scored_spike_counts:", z_scored_spike_counts.shape)  # Should match (n_trials, 101)

    return all_neurons_data, midpoints

def svm(all_neurons_data, midpoints, iterations, permutation_test=False, analysis_type='error_correct'):
    accuracy_vs_time = []

    # Loop over each bin defined by midpoints
    for bin_idx, midpoint in enumerate(midpoints):
        bin_accuracies = []

        for _ in range(iterations):
            feature_matrix = []
            labels = [0] * 10 + [1] * 10

            # Loop through all neurons and randomly select trials
            for neuron, spike_data in all_neurons_data.items():
                if analysis_type == 'error_correct':
                    group1_trials = spike_data['correct'][:, bin_idx]  # Correct trials
                    group2_trials = spike_data['error'][:, bin_idx]  # Error trials
                elif analysis_type == 'congruence':
                    group1_trials = spike_data['correct_congruent'][:, bin_idx]  # Congruent trials
                    group2_trials = spike_data['correct_incongruent'][:, bin_idx]  # Incongruent trials

                # Randomly sample 10 trials for each group
                sampled_group1 = shuffle(group1_trials)[:10]
                sampled_group2 = shuffle(group2_trials)[:10]

                # Combine the selected trials into a single list
                trials_for_neuron = np.concatenate([sampled_group1, sampled_group2])

                # Append the trials for this neuron to the feature matrix
                feature_matrix.append(trials_for_neuron)

            # Convert feature matrix to numpy array and check shape
            feature_matrix = np.array(feature_matrix).T  # Shape (20, # neurons)
            labels = np.array(labels)  # Shape (20,)

            # Shuffle labels for permutation test
            if permutation_test:
                labels = shuffle(labels)

            # Perform cross-validation on this feature matrix
            svm_model = SVC(kernel='linear')
            cv_scores = cross_val_score(svm_model, feature_matrix, labels, cv=5)
            bin_accuracies.append(np.mean(cv_scores))

        # Compute average accuracy for this bin
        accuracy_vs_time.append(np.mean(bin_accuracies))

    return accuracy_vs_time


def plot_accuracy_vs_time(midpoints, accuracy_vs_time, output_file):
    # Convert midpoints to ms
    midpoints = midpoints / 1_000  # Convert from us to ms

    # Plot the accuracy over time
    plt.figure(figsize=(10, 6))
    plt.plot(midpoints, accuracy_vs_time, marker='o', label='Accuracy')

    # Add labels, title, and grid
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance Level (50%)')
    plt.xlabel('Time (ms) relative to button press onset (at t=0)')
    plt.ylabel('Classification Accuracy')
    plt.title('Classifier Accuracy Over Time (True Moving Average)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)


# Calling SVM
##############################################################################

# directory_path= "/Users/sophiapouya/workspace/utsw/research_project/svm_error_neurons/hip"

# bin_size = 400  # ms bins
# step_size = 20  # ms steps
# iterations = 50  # Number of iterations for averaging
# event_type = 'button_press'  # 'button_press' or 'stimulus'

# # Process data
# all_neurons_data, midpoints = process_data(directory_path, bin_size, step_size, event_type)

# # Run SVM analysis
# accuracy_vs_time = svm(all_neurons_data, midpoints, iterations, analysis_type='error_correct')

# # Plot results
# plot_accuracy_vs_time(midpoints, accuracy_vs_time, output_file='accuracy_vs_time.png')