import matplotlib.pyplot as plt
import os
import numpy as np
from plotting import plot_rasters
from stats import t_test, calc_avg
from data_loading import load_data, bin_data

def plot_data(data, error, correctCongruent, correctIncongruent, file_name, output_folder=None):
    """
    Plots raster and PSTH for a single neuron with correct formatting.
    """

    # **Make Raster Tall, Keep PSTH Square**
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(6, 10),  # **Tall figure for correct proportions**
        gridspec_kw={'height_ratios': [3, 1]},
        constrained_layout = True  # **Raster 5× taller than PSTH**
    )

    ax_raster_stim, ax_raster_bp, ax_psth_stim, ax_psth_bp = axes.flat
    # Define colors
    errorColor = 'red'
    congruentColor = 'blue'
    incongruentColor = 'orange'

    ### **Sorting Trials by Reaction Time (RT)**
    def sort_by_RT(trials, rt_data):
        """
        Sort trials by reaction time in ascending order.
        Shortest RTs will appear at the top, longest RTs at the bottom.
        """
        return [x for _, x in sorted(zip([rt_data[i] for i in trials], trials), reverse=False)]

    sorted_error_trials = sort_by_RT(error['trial'], data['RTs'])
    sorted_cc_trials = sort_by_RT(correctCongruent['trial'], data['RTs'])
    sorted_ci_trials = sort_by_RT(correctIncongruent['trial'], data['RTs'])

    # Reorder raster data based on sorted trials
    sorted_error_raster = [error['stimulusTimesRaster'][error['trial'].index(t)] for t in sorted_error_trials]
    sorted_cc_raster = [correctCongruent['stimulusTimesRaster'][correctCongruent['trial'].index(t)] for t in sorted_cc_trials]
    sorted_ci_raster = [correctIncongruent['stimulusTimesRaster'][correctIncongruent['trial'].index(t)] for t in sorted_ci_trials]

    ### **Raster for Stimulus**
    start_index = 0
    for trial_set, color in zip(
        [sorted_error_raster, sorted_cc_raster, sorted_ci_raster],
        [errorColor, congruentColor, incongruentColor]
    ):
        plot_rasters(ax_raster_stim, trial_set, color, start_index)
        start_index += len(trial_set)

    ax_raster_stim.plot([0, 0], [0, start_index], color='grey', linewidth=2)    
    ax_raster_stim.set_ylabel('Trial Number (Sorted)')
    ax_raster_stim.spines['right'].set_visible(False)
    ax_raster_stim.spines['left'].set_visible(False)
    ax_raster_stim.spines['top'].set_visible(False)


    # **Plot Black Dot for RT on Stimulus Raster**
    all_sorted_trials = sorted_error_trials + sorted_cc_trials + sorted_ci_trials
    for trial_index, trial in enumerate(all_sorted_trials):
        rt_time = data['RTs'][trial]  # RTs are already in seconds
        ax_raster_stim.scatter(rt_time, trial_index, color='black', s=4, marker='o')


    ### **Sorting & Raster for Button Press**
    sorted_error_raster_bp = [error['bpTimesRaster'][error['trial'].index(t)] for t in sorted_error_trials]
    sorted_cc_raster_bp = [correctCongruent['bpTimesRaster'][correctCongruent['trial'].index(t)] for t in sorted_cc_trials]
    sorted_ci_raster_bp = [correctIncongruent['bpTimesRaster'][correctIncongruent['trial'].index(t)] for t in sorted_ci_trials]

    start_index = 0
    for trial_set, color in zip(
        [sorted_error_raster_bp, sorted_cc_raster_bp, sorted_ci_raster_bp],
        [errorColor, congruentColor, incongruentColor]
    ):
        plot_rasters(ax_raster_bp, trial_set, color, start_index)
        start_index += len(trial_set)

    ax_raster_bp.plot([0, 0], [0, start_index], color='grey', linewidth=2)   
    ax_raster_bp.plot([1, 1], [0, start_index], color='grey', linewidth=2, linestyle="--")     
    ax_raster_bp.spines['right'].set_visible(False)
    ax_raster_bp.spines['left'].set_visible(False)
    ax_raster_bp.spines['top'].set_visible(False)

    ax_raster_stim.set_xlim([-0.5, 1.5])  # Ensure x-axis is properly bounded
    ax_raster_stim.set_ylim([0, start_index])  # Make y-axis fit the trials

    ax_raster_bp.set_xlim([-0.5, 1.5])  
    ax_raster_bp.set_ylim([0, start_index])  

    # Add vertical side labels for Stimulus Raster
    x_label_pos = 1.6
    ax_raster_stim.text(x_label_pos, len(error['trial']) / 2, "wrong", color=errorColor, fontsize=10,
                        ha="left", va="center",  rotation=90)

    ax_raster_stim.text(x_label_pos, len(error['trial']) + len(correctCongruent['trial']) / 2, "crc&gru",
                        color=congruentColor, fontsize=10, ha="left", va="center", rotation=90)

    ax_raster_stim.text(x_label_pos, len(error['trial']) + len(correctCongruent['trial']) + len(correctIncongruent['trial']) / 2, "crc&ingru",
                        color=incongruentColor, fontsize=10, ha="left", va="center", rotation=90)

    # Add vertical side labels for Button Press Raster
    ax_raster_bp.text(x_label_pos, len(error['trial']) / 2, "wrong", color=errorColor, fontsize=10,
                    ha="left", va="center", rotation=90)

    ax_raster_bp.text(x_label_pos, len(error['trial']) + len(correctCongruent['trial']) / 2, "crc&gru",
                    color=congruentColor, fontsize=10, ha="left", va="center", rotation=90)

    ax_raster_bp.text(x_label_pos, len(error['trial']) + len(correctCongruent['trial']) + len(correctIncongruent['trial']) / 2, "crc&ingru",
                    color=incongruentColor, fontsize=10, ha="left", va="center", rotation=90)

    ### **PSTH for Stimulus**
    psthTimes = np.arange(-0.5, 1.5, 0.02)
    ccMovingAvgStimulus = calc_avg(correctCongruent['movingAvgStimulus'], len(psthTimes))
    ciMovingAvgStimulus = calc_avg(correctIncongruent['movingAvgStimulus'], len(psthTimes))
    errMovingAvgStimulus = calc_avg(error['movingAvgStimulus'], len(psthTimes))

    ax_psth_stim.plot(psthTimes, ccMovingAvgStimulus / 0.2, color=congruentColor)
    ax_psth_stim.plot(psthTimes, ciMovingAvgStimulus / 0.2, color=incongruentColor)
    ax_psth_stim.plot(psthTimes, errMovingAvgStimulus / 0.2, color=errorColor)
    ax_psth_stim.axvline(x=0, color='grey', linewidth=2)
    ax_psth_stim.spines['right'].set_visible(False)
    ax_psth_stim.spines['top'].set_visible(False)
    ax_psth_stim.set_ylabel('Spike Rate (Hz)')
    ax_psth_stim.set_xlabel('Time to SO (s)')

    ### **PSTH for Button Press**
    ccMovingAvgBp = calc_avg(correctCongruent['movingAvgBp'], len(psthTimes))
    ciMovingAvgBp = calc_avg(correctIncongruent['movingAvgBp'], len(psthTimes))
    errMovingAvgBp = calc_avg(error['movingAvgBp'], len(psthTimes))

    ax_psth_bp.plot(psthTimes, ccMovingAvgBp / 0.2, color=congruentColor)
    ax_psth_bp.plot(psthTimes, ciMovingAvgBp / 0.2, color=incongruentColor)
    ax_psth_bp.plot(psthTimes, errMovingAvgBp / 0.2, color=errorColor)
    ax_psth_bp.axvline(x=0, color='grey', linewidth=2)
    ax_psth_bp.axvline(x=1, color='grey', linewidth=2, linestyle="--")
    ax_psth_bp.spines['right'].set_visible(False)
    ax_psth_bp.spines['top'].set_visible(False)


    ax_psth_bp.set_xlabel('Time to BP (s)')

    # Remove '.mat' extension from file name
    clean_file_name = os.path.splitext(file_name)[0]
    fig.suptitle(clean_file_name, fontsize=12, y=.99)

    ### **Final Layout Adjustments**
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=0.5)
    if output_folder is None:
        plt.savefig(f"{file_name}_Analysis.png", format='png', dpi=100, bbox_inches='tight')
    else:
        file_path = os.path.join(output_folder, f"{file_name}_Analysis.png")
        plt.savefig(file_path, format='png', dpi=100, bbox_inches='tight')

    plt.close(fig)

output_folder = "/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC_plots/"
mat_files = "/Users/sophiapouya/workspace/utsw/research_project/channel_data/OFC/"
for file in os.listdir(mat_files):
    if file.endswith('.mat'):
        file_path = os.path.join(mat_files, file)
        data = load_data(file_path)
        data, err, correctCongruent, correctIncongruent = bin_data(data)
        plot_data(data, err, correctCongruent, correctIncongruent, file, output_folder=output_folder)  # Saves PNGs



# for folder in folders:
#     for file in os.listdir(folder):
#         file_path = os.path.join(folder, file)
    
#         # Process only .mat files
#         if os.path.isfile(file_path) and file.endswith('.mat'):
#             data = load_data(file_path)
#             data, error, correctCongruent, correctIncongruent = bin_data(data)
#             neuron_name = os.path.splitext(file)[0]
#             neuron_data_list.append((data, error, correctCongruent, correctIncongruent, neuron_name))

# file = "s61ch19ba1c1.mat"
# data = load_data(file)
# data, err, correctCongruent, correctIncongruent, = bin_data(data)
# plot_data(data, err, correctCongruent, correctIncongruent, file)