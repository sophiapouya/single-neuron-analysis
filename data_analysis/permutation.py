import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, levene
from data_loading import load_data  

def bin_spikes(data):
    """
    Bins spike data around stimulus and button press events and assigns labels for correct/incorrect and congruent/incongruent trials.
    """
    bpIndex = int(len(data['events']) / 2)

    for i in range(len(data['answers'])):
        # Stimulus window
        stimulus_start, stimulus_end = data['events'][i] - 0.5e6, data['events'][i] + 1.5e6
        stim_spike_count = len(data['timestampsOfCell'][
            (data['timestampsOfCell'] >= stimulus_start) & (data['timestampsOfCell'] <= stimulus_end)
        ])
        data['stimSpikeCount'].append(stim_spike_count)

        # Button press window
        bp_start, bp_end = data['events'][i + bpIndex] - 0.5e6, data['events'][i + bpIndex] + 1.5e6
        bp_spike_count = len(data['timestampsOfCell'][
            (data['timestampsOfCell'] >= bp_start) & (data['timestampsOfCell'] <= bp_end)
        ])
        data['bpSpikeCount'].append(bp_spike_count)

        # Label error/correct trials: 1 for correct, 0 for incorrect
        data['labels_ce'].append(1 if data['answers'][i] == data['colorsPresented'][i] else 0)

        # Congruent/incongruent applies only to correct trials
        if (data['answers'][i] == data['colorsPresented'][i]):
            # Label congruent/incongruent trials: 1 for correct, 0 for incorrect
            data['labels_ic'].append(1 if data['colorsPresented'][i]==data['textsPresented'][i] else 0)

    return data

def permutate(spike_counts, labels, num_trials=1000):
    """
    Performs permutation testing on spike counts to determine statistical significance.
    """
    results = np.zeros(num_trials, dtype=int)

    for i in range(num_trials):
        shuffled_labels = random.sample(labels, len(labels))  # Shuffle labels
        group1 = [spike_counts[j] for j in range(len(labels)) if shuffled_labels[j] == 1]
        group2 = [spike_counts[j] for j in range(len(labels)) if shuffled_labels[j] == 0]

        if len(group1) < 5 or len(group2) < 5:
            continue

        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        results[i] = 1 if p_value < 0.05 else 0

    return results

def analyze_data(data, spike_counts, labels, window_name, num_trials=1000):
    """
    Performs t-tests and permutation tests for the given labels and spike counts.
    """
    group1 = [spike_counts[i] for i in range(len(labels)) if labels[i] == 1]
    group2 = [spike_counts[i] for i in range(len(labels)) if labels[i] == 0]

    if len(group1) < 5 or len(group2) < 5:
        print(f"Not enough data for t-test in {window_name} window.")
        return None, None, np.zeros(num_trials, dtype=int)

    # Perform Levene's test for variance equality
    _, p_levene = levene(group1, group2)
    equal_var = p_levene >= 0.05

    # Perform t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)

    # Perform permutation test
    results = permutate(spike_counts, labels, num_trials)

    return t_stat, p_value, results

def plot_results(results, analysis_type, window_name, output_file):
    """
    Plots the results of permutation tests or percentages of significant neurons.
    """
    results_array = np.array(list(results.values()))
    percent_significant = np.mean(results_array == 1, axis=0) * 100

    plt.figure(figsize=(10, 5))
    plt.hist(percent_significant, bins=30, alpha=0.7, color='blue')
    plt.xlabel(f"Percentage of Significant Neurons (%) - {analysis_type}")
    plt.ylabel("Count")
    plt.title(f"Distribution of Significant Neurons - {window_name.capitalize()} Window")
    plt.grid(visible=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def process_file(file_path, results, analysis_type, window_results, num_trials=1000):
    """
    Processes a single file for error/correct and congruent/incongruent analysis.
    """
    raw_data = load_data(file_path)
    data = bin_spikes(raw_data)

    # Analyze button press and stimulus windows
    for window_name, spike_counts in [('stimulus', data['stimSpikeCount']), ('button press', data['bpSpikeCount'])]:
        print(f"\nAnalyzing {window_name} window for file: {file_path}")

        if analysis_type == "error_correct":
            labels = data['labels_ce']
        elif analysis_type == "congruence":
            labels = data['labels_ic']

        t_stat, p_value, permutation_results = analyze_data(data, spike_counts, labels, window_name, num_trials)

        if t_stat is not None and p_value is not None:
            print(f"T-test ({window_name} - {analysis_type}): t={t_stat:.3f}, p={p_value:.3g}")

        # Store results for the current window
        if window_name not in window_results:
            window_results[window_name] = {}
        window_results[window_name][file_path] = permutation_results


def process_directory(directory, analysis_type, output_files, num_trials=1000):
    """
    Processes all .mat files in a directory for error/correct and congruent/incongruent analysis.
    Generates separate plots for stimulus and button press windows.
    """
    window_results = {'stimulus': {}, 'button press': {}}

    for file in os.listdir(directory):
        if file.endswith('.mat'):
            file_path = os.path.join(directory, file)
            print(f"Processing file: {file_path}")
            process_file(file_path, {}, analysis_type, window_results, num_trials)

    # Generate plots for each window
    for window_name, results in window_results.items():
        plot_results(results, analysis_type, window_name, output_files[window_name])


def main():
    """
    Main function to process data and generate plots.
    """

if __name__ == "__main__":
    

    directories = {
        "AMY Conflict Neurons": "/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/amy_all_neurons/",
        "HIP Conflict Neurons": "/home/sophiapouya/workspace/utsw/sophia_research_project/conflict_neurons/hip_all_neurons/"
        #"AMY Error Neurons": "/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/amy_all_neurons/",
        #"HIP Error Neurons": "/home/sophiapouya/workspace/utsw/sophia_research_project/error_neurons/hip_all_neurons/",

    }

    for name, path in directories.items():
        print(f"\nAnalyzing {name}: {path}")

        # # Error/Correct Analysis
        # analysis_type = "error_correct"  # Define analysis type
        # output_files = {
        #     "stimulus": f"{name}_stimulus_error_correct_plot.png",
        #     "button press": f"{name}_button_press_error_correct_plot.png",
        # }
        # process_directory(path, analysis_type, output_files, num_trials=1000)

        # Congruence Analysis
        analysis_type = "congruence"  # Define analysis type
        output_files = {
            "stimulus": f"{name}_stimulus_congruence_plot.png",
            "button press": f"{name}_button_press_congruence_plot.png",
        }
        process_directory(path, analysis_type, output_files, num_trials=1000)

