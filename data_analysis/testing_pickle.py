import pickle
import numpy as np

# 📌 Load the pickle file
pickle_file_path = "example1.pickle"

with open(pickle_file_path, 'rb') as handle:
    pickle_data = pickle.load(handle)
# ✅ Unpack the Pickle Data
spikes, list_epo_t0, array_original_edges, list_events_to_monitor, edge_names = pickle_data

# ✅ Debugging Spikes
print("📌 Pickle File Data")
print("Spikes Type:", type(spikes))  
print("Spikes Shape:", spikes.shape)  
print("First 10 Spikes (s):", spikes[:10])  # Should be in seconds

# ✅ Debugging list_epo_t0
print("\nlist_epo_t0 Type:", type(list_epo_t0))  
print("list_epo_t0 Length:", len(list_epo_t0))  
print("First 5 list_epo_t0 values (s):", list_epo_t0[:5])  

# ✅ Debugging array_original_edges
print("\narray_original_edges Type:", type(array_original_edges))  
print("array_original_edges Shape:", array_original_edges.shape)  
print("First 3 Rows of array_original_edges (s):\n", array_original_edges[:3])  

# ✅ Debugging target_edges (Converted list_events_to_monitor)
print("\ntarget_edges Type:", type(list_events_to_monitor))  
print("target_edges Length:", len(list_events_to_monitor))  
print("First 3 Elements of target_edges:", list_events_to_monitor[:3])  

# ✅ Debugging edge_names
print("\nedge_names Type:", type(edge_names))  
print("edge_names Length:", len(edge_names))  
print("Edge Names:", edge_names)  