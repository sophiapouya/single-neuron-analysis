
import numpy as np
import matplotlib.pyplot as plt
from neuronal_rate_morphing import (
    compute_k_b_to_transform_spikes,
    smart_transform_spikes,
    rolling_spikes_to_rate,
    assign_spikes_to_event,
    Spike_Rate_Morphor
)
from data_loading import bin_data, load_data

# 📌 Load Data
mat_file_path = "/Users/sophiapouya/workspace/utsw/research_project/s61ch19ba1c1.mat"
data = load_data(mat_file_path)
data, error, correctCongruent, correctIncongruent = bin_data(data)

# ✅ Convert max RT to µs
max_response_time_s = np.max(data['RTs'])  # RTs are stored in seconds
max_response_time_us = max_response_time_s * 1e6  # Convert to µs

print(f"📌 Max Response Time (µs): {max_response_time_us}")

# ✅ Convert Data to Match Example Format
spikes = np.array(data['timestampsOfCell'])  # Convert spike times to numpy array
list_epo_t0 = list(data['events'])[:len(data['events']) // 2]  # Extract trial start times
array_original_edges = np.array(data['trialType'])[:, None]  # Convert trial types to a NumPy array
list_events_to_monitor = []  # Placeholder, adjust if needed
edge_names = ['start', 'SO', 'FB', 'end']  # Adjust if needed

# Package into the expected format
formatted_data = [spikes, list_epo_t0, array_original_edges, list_events_to_monitor, edge_names]

# ✅ Initialize Morphing
ws = 200  # Window size in ms (adjust as needed)
hop = 20  # Step size in ms (adjust as needed)

morphor = Spike_Rate_Morphor(ws, hop)
morphor.make_original_windows_from_edges(array_original_edges)

# ✅ Compute Transformation Factors
morphor.make_listof_transforming_factors()  # This should work now

print("✅ Transformation factors computed successfully.")

# ✅ Example usage of morphing
rates_full, monitored_event_times = morphor.compute_rate_of_multitrial_aligned_stages(
    spikes,
    list_epo_t0,
    array_original_edges,
    epoch_safe_margin=(-1, 1),
    nest_events_to_monitor=list_events_to_monitor,
    return_df=True
)

print("✅ Rate computation completed.")
