import numpy as np
import scipy.io
import statsmodels.api as sm
import os
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Function to fit GLMs and perform likelihood ratio test
def fit_glm_and_test(spike_counts, outcomes, rt):
    if np.any(np.isnan(spike_counts)) or np.any(np.isinf(spike_counts)):
        print("Invalid spike counts detected.")
        return np.nan, np.nan

    if np.all(spike_counts == spike_counts[0]):
        print("No variation in spike counts.")
        return np.nan, np.nan

    try:
        X_full = sm.add_constant(np.column_stack((outcomes, rt)))
        glm_full = sm.GLM(spike_counts, X_full, family=sm.families.Poisson()).fit()

        X_null = sm.add_constant(rt)
        glm_null = sm.GLM(spike_counts, X_null, family=sm.families.Poisson()).fit()

        lr_stat = 2 * (glm_full.llf - glm_null.llf)
        p_value = chi2.sf(lr_stat, df=1)
        return lr_stat, p_value
    except Exception as e:
        print(f"Error during GLM fitting: {e}")
        return np.nan, np.nan

# Function to perform sliding-window GLM analysis
def sliding_window_glm(data, bin_size, step_size):
    answers = data['answers']
    colors_presented = data['colorsPresented']
    events = data['events']
    button_press_times = events[:, 1]
    rt = np.array(data['RTs'].flatten() * 1_000_000)
    timestamps = data['timestampsOfCell']

    outcomes = np.where(answers == colors_presented, 0, 1)

    stimulus_window = (-500_000, 1500_000)
    total_bins = int((stimulus_window[1] - stimulus_window[0]) / step_size) + 1

    lr_stats = []
    p_values = []

    for bin_idx in range(total_bins):
        spike_counts = []
        for trial_idx, button_press in enumerate(button_press_times):
            window_start = stimulus_window[0] + bin_idx * step_size - bin_size / 2
            window_end = stimulus_window[0] + bin_idx * step_size + bin_size / 2

            start = button_press + window_start
            end = button_press + window_end
            counts = np.sum((timestamps >= start) & (timestamps <= end))
            spike_counts.append(counts)

        spike_counts = np.array(spike_counts)
        lr_stat, p_value = fit_glm_and_test(spike_counts, outcomes, rt)
        lr_stats.append(lr_stat)
        p_values.append(p_value)

    significant = np.array(p_values) < 0.05
    differential_latency = None
    for i in range(len(significant) - 14):
        if np.all(significant[i:i + 15]):
            differential_latency = stimulus_window[0] + i * step_size
            break

    return lr_stats, p_values, differential_latency

# Function to process a directory of .mat files
def process_directory(directory_path, bin_size, step_size):
    differential_latencies = []
    lr_stats_all = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            data = scipy.io.loadmat(file_path)
            lr_stats, _, differential_latency = sliding_window_glm(data, bin_size, step_size)

            if differential_latency is not None:
                differential_latencies.append(differential_latency / 1_000)
                lr_stats_all.append(lr_stats)

    return lr_stats_all, differential_latencies

# Function to plot differential latencies
def plot_differential_latencies(differential_latencies):
    if len(differential_latencies) == 0:
        print("No differential latencies found.")
        return

    sorted_latencies = sorted(differential_latencies, reverse=True)
    median_differential_latency = np.median(sorted_latencies)
    print(f"Median Differential Latency (ms): {median_differential_latency}")

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_latencies, range(1, len(sorted_latencies) + 1), color='b', label="Differential Latency")
    plt.axvline(x=median_differential_latency, color='r', linestyle='--', label=f"Median: {median_differential_latency:.2f} ms")

    plt.xlabel("Differential Latency (ms)")
    plt.ylabel("Files (sorted by latency)")
    plt.title("Differential Latency Across Files")
    plt.legend()
    plt.grid(True)
    plt.savefig('hip_diff_latencies.png')
    plt.show()

# Function to plot a heatmap
def plot_heatmap_with_sorted_neurons(lr_stats, differential_latencies):
    sorted_indices = np.argsort(differential_latencies)
    sorted_lr_stats = np.array(lr_stats)[sorted_indices]

    # Diagnostics to understand the data range
    max_lr = np.nanmax(sorted_lr_stats)

    # Define color scale limits using percentiles to avoid outliers
    vmax = max_lr*.35
    vmin = 0   # 1st percentile as lower limit

    # Create time bins from -500 ms to 1500 ms
    time_bins = np.linspace(-500, 1500, len(sorted_lr_stats[0]))
    
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(sorted_lr_stats, cmap="YlOrRd", 
                cbar_kws={'label': 'Likelihood Ratio'}, vmin=vmin, vmax=vmax)

    # Set x-axis labels
    plt.xticks(ticks=np.linspace(0, len(time_bins) - 1, 5).astype(int), 
               labels=[f"{int(t)} ms" for t in np.linspace(-500, 1500, 5)], rotation=45)

    plt.title("Heatmap of Likelihood Ratios Across Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron/File")
    plt.savefig('hip_heatmap.png')
    plt.show()

# Load and process the data-> single file example
########################################################
# data = scipy.io.loadmat('s61ch19ba1c1.mat')

# bin_size = 400_000  # Bin size in microseconds
# step_size = 10_000  # Step size in microseconds

# lr_stats, p_values, differential_latencies = sliding_window_glm(data, bin_size, step_size)
# print(lr_stats)
# print("Differential Latency (ms):", differential_latencies / 1_000 if differential_latencies else "None")  # Convert to ms for readability

# #Plot the results
# plot_differential_latencies(differential_latencies)


# Plotting the whole directory and a Heatmap
########################################################
directory_path = '../neurons/hip_error_neurons/'
bin_size = 400_000  # Bin size in microseconds
step_size = 10_000  # Step size in microseconds

lr_stats_all, differential_latencies = process_directory(directory_path, bin_size, step_size)
#plot_differential_latencies(differential_latencies)
plot_heatmap_with_sorted_neurons(lr_stats_all, differential_latencies)


