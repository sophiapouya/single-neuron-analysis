import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

### 🔹 **Load MATLAB Data**
mat_file_path = "/Users/sophiapouya/workspace/utsw/research_project/s61ch19ba1c1.mat"
data = loadmat(mat_file_path)

# **Extract spikes, events, and response times**
spikes = data["timestampsOfCell"].flatten()  # Spike timestamps (µs)
stimulus_times = data["events"][:, 0]  # Stimulus onset times (µs)
response_times = data["RTs"].flatten()  # RTs in s

# **Define fixed analysis window**
pre_stimulus_window = -1_000_000  # -1.0s before stimulus onset
post_stimulus_window = 3_500_000  # +1.0s after median response time

# Compute median response time
median_response_time = 0.8

print(f"📌 Stretching each trial from stimulus onset (0s) to its own response time.")

### **🔹 Extract Trial Categories**
def extract_trial_categories(data):
    """Extract trial indices for correct congruent, correct incongruent, and error trials."""
    correct_congruent, correct_incongruent, error_trials = [], [], []

    for i in range(len(data["answers"])):
        if data['answers'][i] == data['colorsPresented'][i]:  
            if data['colorsPresented'][i] == data['textsPresented'][i]:  
                correct_congruent.append(i)  
            else:
                correct_incongruent.append(i)  
        else:
            error_trials.append(i)  

    return correct_congruent, correct_incongruent, error_trials

# **Get trial indices**
congruent_indices, incongruent_indices, error_indices = extract_trial_categories(data)

### **🔹 Extract and Stretch Spikes for Each Trial**
trial_spikes = []
trial_rts = []
trial_labels = []  # 1 = congruent, 2 = incongruent, 3 = error

for trial_idx, stim_time in enumerate(stimulus_times):
    # **Pre-Stimulus Window (Untouched)**
    trial_start = stim_time + pre_stimulus_window

    # **Post-Stimulus Window (Stretched)**
    trial_end_response = stim_time + post_stimulus_window 

    # **Extract spikes within the trial window**
    trial_spike_times = spikes[(spikes >= trial_start) & (spikes <= trial_end_response)]
    
    # **Align spike times relative to stimulus onset (Stimulus at 0s)**
    trial_spike_times = trial_spike_times - stim_time

    # **Separate pre-stimulus and post-stimulus spikes**
    pre_stim_spikes = (trial_spike_times[trial_spike_times <= 0])/1e6  # Before stimulus (Untouched) in seconds
    post_stim_spikes = (trial_spike_times[trial_spike_times > 0])/1e6  # After stimulus (Stretchable) in seconds

    # **Compute per-trial stretching factor**
    scale_factor = (median_response_time) / (response_times[trial_idx]) if response_times[trial_idx] > 0 else 1.0
    post_stim_spikes = post_stim_spikes * scale_factor  # Stretch spikes

    # **Combine all spike times for the trial**
    trial_spike_times = np.concatenate([pre_stim_spikes, post_stim_spikes])

    # **Store spike data, RTs, and labels**
    trial_spikes.append(trial_spike_times)
    trial_rts.append(response_times[trial_idx])

    if trial_idx in congruent_indices:
        trial_labels.append(1)  # Congruent
    elif trial_idx in incongruent_indices:
        trial_labels.append(2)  # Incongruent
    elif trial_idx in error_indices:
        trial_labels.append(3)  # Error

### **🔹 Sort Trials by Response Time**
# Convert to NumPy array for sorting
trial_rts = np.array(trial_rts)
trial_labels = np.array(trial_labels)
trial_spikes = np.array(trial_spikes, dtype=object)  # Preserve different-length arrays

# **Sort Congruent, Incongruent, and Error trials separately**
sorted_indices_congruent = np.argsort(-trial_rts[trial_labels == 1])  # Descending RT
sorted_indices_incongruent = np.argsort(-trial_rts[trial_labels == 2])  # Descending RT
sorted_indices_error = np.argsort(-trial_rts[trial_labels == 3])  # Descending RT

# **Reorder trials**
congruent_sorted = trial_spikes[trial_labels == 1][sorted_indices_congruent]
incongruent_sorted = trial_spikes[trial_labels == 2][sorted_indices_incongruent]
error_sorted = trial_spikes[trial_labels == 3][sorted_indices_error]  # Errors at the bottom


### **🔹 Raster Plot Function**
def plot_raster(ax, spike_data, color, start_index):
    for trial_idx, trial_spikes in enumerate(spike_data, start=start_index):
        ax.scatter(trial_spikes, [trial_idx] * len(trial_spikes), color=color, s=2)

# **Generate Raster Plot**
fig, ax = plt.subplots(figsize=(10, 6))

# **Plot trials (Errors go at the bottom)**
plot_raster(ax, error_sorted, "red",0)  # Errors first
plot_raster(ax, congruent_sorted, "blue", len(error_sorted))
plot_raster(ax, incongruent_sorted, "orange", len(congruent_sorted)+len(error_sorted))

# **Markers and labels**
ax.axvline(0, linestyle="--", color="red", label="Stimulus Onset")
ax.axvline(median_response_time, linestyle="--", color="blue", label="Median RT")
ax.set_xlim(pre_stimulus_window/1e6, 2.8)  # Adjust x-axis limit
ax.set_xlabel("Time (s)")
ax.set_ylabel("Trial (Sorted by RT)")
ax.set_title("Time-Stretched Raster Plot (Sorted by RT, Errors at Bottom)")
ax.legend()
plt.show()