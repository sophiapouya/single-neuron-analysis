import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from data_loading import load_data, bin_data

def svm_global_specificity(all_neurons_data, midpoints, iterations):
    """
    Perform SVM analysis for global specificity (trial type classification).
    Ensures feature_matrix has shape (n_samples, n_neurons) and labels are aligned.
    """
    accuracy_vs_time = []

    # Loop over time bins (midpoints)
    for bin_idx, midpoint in enumerate(midpoints):
        bin_accuracies = []

        # Perform bootstrapping over iterations
        for _ in range(iterations):
            feature_matrix = []
            labels = []

            # Collect data for all trials
            for trial_type in range(1, 10):
                trial_type_labels = []
                trial_type_data = []

                for neuron, spike_data in all_neurons_data.items():
                    trials = np.where(spike_data['trialType'] == trial_type)[0]

                    sampled_trials = np.random.choice(trials, size=10, replace=False)

                    # Append this neuron's data for the sampled trials
                    trial_type_data.append(spike_data['spike_counts'][sampled_trials, bin_idx])

                # Only extend labels once per trial type
                trial_type_labels.extend([trial_type] * len(sampled_trials))

                # Append the trial data and labels for this trial type
                feature_matrix.append(np.array(trial_type_data).T)  # Trials x Neurons
                labels.extend(trial_type_labels)

            # Convert feature_matrix to NumPy array
            feature_matrix = np.vstack(feature_matrix)  # Shape: (n_samples, n_neurons)
            labels = np.array(labels)

            # Perform SVM classification
            svm_model = SVC(kernel='linear', decision_function_shape='ovr')
            cv_scores = cross_val_score(svm_model, feature_matrix, labels, cv=5)  # 5-fold cross-validation
            bin_accuracies.append(np.mean(cv_scores))

        # Average accuracy for the current bin
        accuracy_vs_time.append(np.mean(bin_accuracies) if bin_accuracies else 0)

    return accuracy_vs_time


def plot_accuracy_vs_time(midpoints, accuracy_vs_time, output_file):
    """
    Plot classification accuracy over time for global specificity analysis.
    """
    midpoints = midpoints / 1_000  # Convert from us to ms

    plt.figure(figsize=(10, 6))
    plt.plot(midpoints, accuracy_vs_time, marker='o', label='Trial Type Classification')
    plt.axhline(y=1/9, color='r', linestyle='--', label='Chance Level (11.1%)')  # Chance for 9 trial types
    plt.xlabel('Time (ms)')
    plt.ylabel('Classification Accuracy')
    plt.title('Global Specificity: Trial Type Classification Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

def main():
    """
    Main function to perform global specificity analysis for trial types.
    """
    # Define directories and parameters
    hip_neurons = '/home/sophiapouya/workspace/utsw/sophia_research_project/svm_global/hip_neurons/'
    bin_size = 400  # ms bins
    step_size = 20  # ms steps
    iterations = 50  # Number of iterations
    midpoints = np.arange(-500 * 1_000, (1500 * 1_000) + step_size * 1_000, step_size * 1_000)

    # Initialize data structure
    all_neurons_data = {}

    # Process each .mat file in the directory
    for file in os.listdir(hip_neurons):
        file_path = os.path.join(hip_neurons, file)
        if os.path.isfile(file_path) and file.endswith('.mat'):
            print(f"Processing file: {file}")

            # Load and bin data
            data = load_data(file_path)
            # Define the stimulus points and button press points
            data, _, _, _ = bin_data(data)  # Ignore unused categories (error, congruent, incongruent)

            # Define stimulus and button press times
            num_trials = len(data['trialType'])  # Total number of trials
            stimulus_times = data['events'][:num_trials]  # First half: stimulus times
            button_press_times = data['events'][num_trials:]  # Second half: button press times

            # Binning logic
            spike_counts = []
            for trial_idx in range(len(data['trialType'])):
                trial_spike_counts = []
                for midpoint in midpoints:
                    # Define bin window
                    bin_start = stimulus_times[trial_idx] + midpoint - 0.5 * bin_size * 1_000
                    bin_end = stimulus_times[trial_idx] + midpoint + 0.5 * bin_size * 1_000

                    # Count spikes in the bin
                    spike_count = np.sum(
                        (data['timestampsOfCell'] >= bin_start) &
                        (data['timestampsOfCell'] < bin_end)
                    )
                    trial_spike_counts.append(spike_count)

                spike_counts.append(trial_spike_counts)

            # Store trialType and spike counts
            all_neurons_data[file] = {
                'spike_counts': np.array(spike_counts),  # Shape: trials × bins
                'trialType': np.array(data['trialType'])
            }

    # Run SVM analysis
    print("Starting SVM global specificity analysis...")
    accuracy_vs_time = svm_global_specificity(all_neurons_data, midpoints, iterations)

    # Plot results
    output_file = 'global_specificity_accuracy.png'
    plot_accuracy_vs_time(midpoints, accuracy_vs_time, output_file)
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
