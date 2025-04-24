import numpy as np
from data_loading import bin_data, load_data

# 📌 Load Data
mat_file_path = "/Users/sophiapouya/workspace/utsw/research_project/s61ch19ba1c1.mat"
data = load_data(mat_file_path)
data, error, correctCongruent, correctIncongruent = bin_data(data)

# ✅ Convert spikes to seconds (from µs)
spikes = data['timestampsOfCell'] / 1e6  

# ✅ Define event times in seconds
stimulus_times = np.array(data['events'][:len(data['events']) // 2]) / 1e6
button_press_times = np.array(data['events'][len(data['events']) // 2:]) / 1e6
response_times = button_press_times - stimulus_times  # Trial-specific RTs

# ✅ Format list_epo_t0 (start of trials)
list_epo_t0 = stimulus_times.tolist()  # Convert to list

# ✅ Convert array_original_edges to **relative** times (same as pickle file)
array_original_edges = np.column_stack([
    np.full_like(stimulus_times, -1.0),  # Pre-stimulus (-1s)
    np.zeros_like(stimulus_times),  # Stimulus onset (0s)
    response_times,  # Response time per trial
    response_times + 2.0  # Post-response (+2s)
])

# ✅ Convert target_edges to list of dicts (matching pickle file format)
target_edges = [{1: np.array([rt])} for rt in response_times]  # Convert each RT to dict

# ✅ Debugging Output
print("\n📌 MATLAB Data (Reformatted to Match Pickle)")
print("Spikes Type:", type(spikes))  
print("Spikes Shape:", spikes.shape)  
print("First 10 Spikes (s):", spikes[:10])  

print("\nlist_epo_t0 Type:", type(list_epo_t0))  
print("list_epo_t0 Length:", len(list_epo_t0))  
print("First 5 list_epo_t0 values (s):", list_epo_t0[:5])  

print("\narray_original_edges Type:", type(array_original_edges))  
print("array_original_edges Shape:", array_original_edges.shape)  
print("First 3 Rows of array_original_edges (s):\n", array_original_edges[:3])  

print("\ntarget_edges Type:", type(target_edges))  
print("target_edges Length:", len(target_edges))  
print("First 3 Elements of target_edges:", target_edges[:3])  