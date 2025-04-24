from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

#define spike counter function
def get_spikes(data, start_time, end_time):
    spikes = data["timestampsOfCell"][(data["timestampsOfCell"]>=start_time) & (data["timestampsOfCell"]<=end_time)]       
    return spikes

def compress_stretch(spikes, trial_rt, rt_scale = 0.8):
    #spikes in microseconds
    rt_scale=rt_scale*1e6
    scale_factor = rt_scale/(trial_rt*1e6) if trial_rt > 0 else 1.0
    return np.array(spikes*scale_factor)

def moving_avg(start_time, end_time, spikes, bin_size=0.4*1e6, step_size=.02*1e6):
    # Precompute bin edges for efficiency
    bin_centers = np.arange(start_time, end_time, step_size)
    bin_starts = bin_centers - bin_size / 2
    bin_ends = bin_centers + bin_size / 2
    # Compute spike counts for each bin (without incorrect filtering)
    trial_avg = [
        np.sum((spikes >= bin_starts[j]) & (spikes <= bin_ends[j]))
        for j in range(len(bin_centers))
    ]
    # Ensure trial_avg has the expected length
    trial_avg = np.array(trial_avg)/(bin_size/1e6)
    return trial_avg

def calc_avg(moving_avg_list, target_shape):
    #Compute the average across multiple trials
    if not moving_avg_list:
        return np.zeros(target_shape)
    return np.mean(np.array(moving_avg_list), axis=0)


def get_trial_data(data, start_offset, end_offset):
    trial_data = {
        "correct": [],
        "incongruent": [],
        "congruent": [],
        "error": [],
        "colors": {"1": [], "2": [], "3": []}, #1 -> red, 2-> green, 3-> blue
        "text": {"1": [], "2": [], "3": []} #1 -> red, 2-> green, 3-> blue
    }
    for trial in range(len(data["answers"])):
        stimulus_time = data["events"][trial,0]
        start_time = stimulus_time + start_offset*1e6
        end_time = stimulus_time + end_offset*1e6
        trial_rt = data["RTs"][trial]
        #Add extra time to both vectors to allow for edge artifact correction in moving avg calc
        #spikes altered -> stimulus, 3.5 seconds
        #spikes unaltered -> -1.0, stimulus
        trial_spikes_altered = get_spikes(data, start_time=stimulus_time, end_time=end_time)
        trial_spikes_altered_relative = trial_spikes_altered - stimulus_time
        trial_spikes_unaltered= get_spikes(data, start_time=start_time, end_time=stimulus_time)
        trial_spikes_unaltered_relative = trial_spikes_unaltered - stimulus_time
        trial_spikes_stretched = compress_stretch(spikes=trial_spikes_altered_relative, trial_rt=trial_rt)
        trial_spikes = np.concatenate((trial_spikes_unaltered_relative, trial_spikes_stretched))
        trial_moving_avg = moving_avg(start_time=-1.0*1e6, end_time=2.8*1e6, spikes=trial_spikes)
        error_flag = 0 if data["answers"][trial] == data["colorsPresented"][trial] else 1
        
        # Create trial dictionary
        trial_info = {"trial_spikes": trial_spikes, "moving_avg": trial_moving_avg, "trial_rt":trial_rt, "error_flag":error_flag}

        #now add data based on trial type to create correct Raster and PSTH plots
        if data["answers"][trial] == data["colorsPresented"][trial]:
            #correct
            trial_data["correct"].append(trial_info)
            if data["colorsPresented"][trial] == data["textsPresented"][trial]:
                #congruent
                trial_data["congruent"].append(trial_info)
            else:
                #incongruent
                trial_data["incongruent"].append(trial_info)
        else:
            #error
            trial_data["error"].append(trial_info)
            continue

        color = str(int(data["colorsPresented"][trial].item()))  # Convert NumPy array to integer, then to string
        text = str(int(data["textsPresented"][trial].item()))  # Convert NumPy array to integer, then to string
        trial_data["colors"][color].append(trial_info)
        trial_data["text"][text].append(trial_info)

    return trial_data

def plot_rasters(ax, data, color, start_index, time_range=(-0.5, 1.5)):
    for trialIndex, timesRaster in enumerate(data, start=start_index):
        filtered_spikes = timesRaster[(timesRaster >= time_range[0] * 1_000_000) & (timesRaster <= time_range[1] * 1_000_000)]
        ax.scatter(filtered_spikes / 1_000_000, [trialIndex] * len(filtered_spikes), color=color, s=2)

def plot(trial_data, time_range=(-1.0, 2.8), file_name=None, plot_mode="error_correct", output_dir=None, clusters=None, cluster_p_values=None):

    # Check for None values as in production_plots
    if file_name is None:
        file_name = "Untitled"
    
    # Create a single figure with constrained_layout
    fig = plt.figure(constrained_layout=True, figsize=(6, 10), dpi=300)
    # Create a GridSpec with 3 rows: a tiny row for the title, then for raster and PSTH plots
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[0.001, 3, 1])
    
    # Create an axis for the title and hide its frame
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')
    clean_file_name = os.path.splitext(file_name)[0]
    # Append the plot_mode to the title so you know which category is being plotted
    ax_title.set_title(clean_file_name, fontsize=12)
    
    # Create the raster and PSTH axes
    ax_raster = fig.add_subplot(gs[1, 0])
    ax_psth = fig.add_subplot(gs[2, 0])
    
    # Define the categories and colors depending on the selected mode
    if plot_mode == "error_correct":
        # Combine correct trials into one group vs errors
        categories = {"error": 'red', "correct": 'green'}
        sorted_trials = {
            cat: sorted(trial_data.get(cat, []), key=lambda t: t["trial_rt"])
            for cat in categories
        }
    elif plot_mode == "error_correct_subtypes":
        # Split correct trials into congruent vs incongruent, along with errors
        categories = {"error": 'red', "congruent": 'blue', "incongruent": 'orange'}
        sorted_trials = {
            cat: sorted(trial_data.get(cat, []), key=lambda t: t["trial_rt"])
            for cat in categories
        }
    elif plot_mode == "congruent_incongruent":
        # Split correct trials into congruent vs incongruent
        categories = {"congruent": 'blue', "incongruent": 'orange'}
        sorted_trials = {
            cat: sorted(trial_data.get(cat, []), key=lambda t: t["trial_rt"])
            for cat in categories
        }
    elif plot_mode == "color":
        # Assume trial_data["colors"] is a dict with keys "1", "2", "3"
        # Map these keys to red, blue, and green
        categories = {"1": 'red', "2": 'green', "3": 'blue'}
        sorted_trials = {
            cat: sorted(trial_data.get("colors", {}).get(cat, []), key=lambda t: t["trial_rt"])
            for cat in categories
        }
    elif plot_mode == "text":
        # Assume trial_data["text"] is a dict with keys "1", "2", "3"
        categories = {"1": 'red', "2": 'green', "3": 'blue'}
        sorted_trials = {
            cat: sorted(trial_data.get("text", {}).get(cat, []), key=lambda t: t["trial_rt"])
            for cat in categories
        }
    else:
        raise ValueError("Invalid plot_mode. Please choose 'error_correct', 'error_correct_subtypes', 'color', or 'text'.")
    
    # --- Raster Plot ---
    start_index = 0
    for cat, color in categories.items():
        # Retrieve spike arrays for each trial in this category
        trials_list = [trial["trial_spikes"] for trial in sorted_trials[cat]]
        plot_rasters(ax_raster, trials_list, color, start_index, time_range)
        start_index += len(trials_list)
    
    # Add stimulus onset markers
    ax_raster.axvline(x=0, color='black', linewidth=2)
    ax_raster.axvline(x=0.8, color='black', linewidth=3, linestyle="dashed")
    ax_raster.axvline(x=1.8, color='black', linewidth=3, linestyle="dashed")
    ax_raster.set_ylabel('Trial Number (Sorted)')
    ax_raster.set_xlim(time_range)
    ax_raster.set_ylim([0, start_index])
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['top'].set_visible(False)
    
    # Add category labels along the side of the raster plot
    x_label_pos = time_range[1] + 0.05  # adjust as needed
    
    if plot_mode == "error_correct":
        error_count = len(sorted_trials.get("error", []))
        correct_count = len(sorted_trials.get("correct", []))
        ax_raster.text(x_label_pos, error_count / 2, "wrong", color=categories["error"],
                       fontsize=10, ha="left", va="center", rotation=90)
        ax_raster.text(x_label_pos, error_count + correct_count / 2, "correct", color=categories["correct"],
                       fontsize=10, ha="left", va="center", rotation=90)
    elif plot_mode == "error_correct_subtypes":
        error_count = len(sorted_trials.get("error", []))
        congruent_count = len(sorted_trials.get("congruent", []))
        incongruent_count = len(sorted_trials.get("incongruent", []))
        ax_raster.text(x_label_pos, error_count / 2, "wrong", color=categories["error"],
                       fontsize=10, ha="left", va="center", rotation=90)
        ax_raster.text(x_label_pos, error_count + congruent_count / 2, "correct_congruent", color=categories["congruent"],
                       fontsize=10, ha="left", va="center", rotation=90)
        ax_raster.text(x_label_pos, error_count + congruent_count + incongruent_count / 2, "correct_incongruent",
                       color=categories["incongruent"], fontsize=10, ha="left", va="center", rotation=90)
    elif plot_mode == "congruent_incongruent":
        congruent_count = len(sorted_trials.get("congruent", []))
        incongruent_count = len(sorted_trials.get("incongruent", []))
        ax_raster.text(x_label_pos, congruent_count / 2, "correct_congruent", color=categories["congruent"],
                       fontsize=10, ha="left", va="center", rotation=90)
        ax_raster.text(x_label_pos, congruent_count + incongruent_count / 2, "correct_incongruent",
                       color=categories["incongruent"], fontsize=10, ha="left", va="center", rotation=90)
    elif plot_mode in ["color", "text"]:
        total = 0
        for cat, c in categories.items():
            count = len(sorted_trials.get(cat, []))
            if count > 0:
                if plot_mode == "color":
                    label = f"{c} color"
                else:
                    label = f"{c} text"
                ax_raster.text(x_label_pos, total + count / 2, label,
                            color=c, fontsize=10, ha="left", va="center", rotation=90)
            total += count
    
    # --- PSTH Plot ---
    # Define PSTH times and calculate the average PSTH per category
    psth_times = np.arange(time_range[0] * 1e6, time_range[1] * 1e6, 0.02 * 1e6) / 1e6
    avg_psth = {}
    for cat in categories:
        trials = sorted_trials[cat]
        avg_psth[cat] = calc_avg([trial["moving_avg"] for trial in trials], len(psth_times))
    
    for cat, color in categories.items():
        ax_psth.plot(psth_times, avg_psth[cat], color=color, label=f"{plot_mode} {cat}")
    
    ax_psth.axvline(x=0, color='black', linewidth=2)
    ax_psth.axvline(x=0.8, color='black', linewidth=3, linestyle="dashed")
    ax_psth.axvline(x=1.8, color='black', linewidth=3, linestyle="dashed")
    ax_psth.spines['right'].set_visible(False)
    ax_psth.spines['top'].set_visible(False)
    ax_psth.set_xlim(time_range)
    ax_psth.set_ylabel('Spike Rate (Hz)')
    ax_psth.set_xlabel('Time to SO (s)')

    # --- Shade Significant Clusters on PSTH ---
    # If clusters and cluster p-values are provided, shade those with p < 0.05.
    if clusters is not None and cluster_p_values is not None:
        for (start, end, cluster_stat), p in zip(clusters, cluster_p_values):
            if p < 0.05:
                # Map indices to time using psth_times. Since 'end' is exclusive,
                # we can use psth_times[end-1] or psth_times[end] if available.
                t_start = psth_times[start]
                t_end = psth_times[end - 1]
                ax_psth.axvspan(t_start, t_end, color='grey', alpha=0.3)
    
    outfile_name= clean_file_name+f"_{plot_mode}.png"
    output = os.path.join(output_dir,outfile_name)
    plt.savefig(output)
    plt.close(fig)

def main():
    #single plot example
    test_file='s61ch19ba1c1.mat'
    data = loadmat(test_file)
    trial_data = get_trial_data(data=data, start_offset=-1.5*1e6, end_offset=3.5*1e6)
    print(len(trial_data["error"]))
    plot(trial_data=trial_data, file_name=test_file, plot_mode="error_correct_subtypes", output_dir='/Users/sophiapouya/workspace/utsw/research_project/data_analysis/')


    # # Define folders to process and target directories
    #     # Folder with mat files
    # folders = ['/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY/', 
    # '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP/',
    # '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC/', 
    # '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA/',
    # '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC/']
    # #Folder for all neurons relevant to error analysis
    # target_dirs_error_correct = {
    #     'amy': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY_plots_ec/',
    #     'hip': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP_plots_ec/',
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC_plots_ec/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA_plots_ec/',
    #     'ofc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC_plots_ec/'
    # }
    # # Folder for neurons identified as error neurons
    # target_dirs_error_congruent_incongruent = {
    #     'amy': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY_plots_eci/',
    #     'hip': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP_plots_eci/',
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC_plots_eci/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA_plots_eci/',
    #     'ofc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC_plots_eci/'
    # }
    # # Folder for all neurons relevant to conflict analysis
    # target_dirs_colors = {
    #     'amy': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY_plots_colors/',
    #     'hip': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP_plots_colors/',
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC_plots_colors/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA_plots_colors/',
    #     'ofc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC_plots_colors/'
    # }
    # # Folder for all neurons relevant to conflict analysis
    # target_dirs_texts = {
    #     'amy': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/AMY_plots_texts/',
    #     'hip': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP_plots_texts/',
    #     'acc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/ACC_plots_texts/',
    #     'sma': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/SMA_plots_texts/',
    #     'ofc': '/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC_plots_texts/'
    # }
    # # Loop through the specified folders and process files
    # for folder in folders:
    #     for file in os.listdir(folder):
    #         file_path = os.path.join(folder, file)
            
    #         # Ensure it's a .mat file and not a directory
    #         if os.path.isfile(file_path) and file.endswith('.mat'):
    #             data = loadmat(file_path)
    #             trial_data = get_trial_data(data=data, start_offset=-1.5*1e6, end_offset=3.5*1e6)
    #             if 'AMY' in folder:
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct", output_dir=target_dirs_error_correct['amy'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct_subtypes", output_dir=target_dirs_error_congruent_incongruent['amy'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="color", output_dir=target_dirs_colors['amy'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="text", output_dir=target_dirs_texts['amy'])
    #             elif 'HIP' in folder:
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct", output_dir=target_dirs_error_correct['hip'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct_subtypes", output_dir=target_dirs_error_congruent_incongruent['hip'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="color", output_dir=target_dirs_colors['hip'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="text", output_dir=target_dirs_texts['hip'])
    #             elif 'ACC' in folder:
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct", output_dir=target_dirs_error_correct['acc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct_subtypes", output_dir=target_dirs_error_congruent_incongruent['acc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="color", output_dir=target_dirs_colors['acc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="text", output_dir=target_dirs_texts['acc'])
    #             elif 'SMA' in folder:
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct", output_dir=target_dirs_error_correct['sma'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct_subtypes", output_dir=target_dirs_error_congruent_incongruent['sma'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="color", output_dir=target_dirs_colors['sma'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="text", output_dir=target_dirs_texts['sma'])
    #             elif 'OFC' in folder:
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct", output_dir=target_dirs_error_correct['ofc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="error_correct_subtypes", output_dir=target_dirs_error_congruent_incongruent['ofc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="color", output_dir=target_dirs_colors['ofc'])
    #                 plot(trial_data=trial_data, file_name=file, plot_mode="text", output_dir=target_dirs_texts['ofc'])


if __name__ == "__main__":
    main()
