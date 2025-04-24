import os
import numpy as np
import matplotlib.pyplot as plt
from stats import calc_avg, roc_analysis, t_test

def plot_rasters(ax, data, color, start_index):
    for trialIndex, timesRaster in enumerate(data, start=start_index):
        ax.scatter(timesRaster / 1_000_000, [trialIndex] * len(timesRaster), color=color, s=2)

def plot_data(data, error, correctCongruent, correctIncongruent, file_name, output_folder):

    # Create a figure and axis for the raster and PSTH plot
    fig, axes = plt.subplots(
        nrows=3,ncols=2,figsize=(12, 16),
        sharex=False,
        gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1, 1]}
    )
    ax, ax2, psth, psth2, roc, roc2 = axes.flat
    
    #share axis between psth and ax and psth2 and ax2
    psth.sharex(ax)
    psth2.sharex(ax2)

    # Define colors for each trial type
    errorColor = 'red'
    congruentColor = 'blue'
    incongruentColor = 'orange'

    #Raster for Stimulus ###################################################
    plot_rasters(ax, error['stimulusTimesRaster'], errorColor, 0)
    plot_rasters(ax, correctCongruent['stimulusTimesRaster'], congruentColor, len(error['stimulusTimesRaster']))
    plot_rasters(ax, correctIncongruent['stimulusTimesRaster'], incongruentColor, len(error['stimulusTimesRaster']) + len(correctCongruent['stimulusTimesRaster']))

          
    #perform ttest and set plot title to results
    rce_ttest, rce_pvalue, ric_ttest, ric_pvalue = t_test(
        np.array(correctCongruent['spikeRatesBp']),
        np.array(correctIncongruent['spikeRatesStim']),
        np.array(error['spikeRatesStim']))
    
    if len(error['trial']) >=5:
        if rce_ttest is not None and rce_pvalue is not None:
            ce_text = f'T-test (correct/error): t={rce_ttest:.2f}, p={rce_pvalue:.3f}\n'
        else:
            ce_text = 'Insufficient data for Correct/Error T-tests'
    else:
        ce_text = 'Insufficient data for Correct/Error T-tests' 
    if ric_ttest is not None and ric_pvalue is not None: 
        ci_text = f'T-test (incongruent/congruent): t={ric_ttest:.2f}, p={ric_pvalue:.3f}'
    else: 
        ci_text = 'Insufficient data for Incongruent/Congruent T-tests'
    
    ax.set_title(ce_text+'\n'+ci_text)
    
    # Add vertical grey line at x=0 (stimulus onset)
    ax.axvline(x=0, color='grey', linewidth=1)
    ax.set_ylabel('Trial Number')
    ax.set_xlabel('Time from Stimulus (s)')

    #Raster for Button Press################################################
    plot_rasters(ax2, error['bpTimesRaster'], errorColor, 0)
    plot_rasters(ax2, correctCongruent['bpTimesRaster'], congruentColor, len(error['bpTimesRaster']))
    plot_rasters(ax2, correctIncongruent['bpTimesRaster'], incongruentColor, len(correctCongruent['bpTimesRaster']) + len(error['bpTimesRaster']))

    #perform ttest and set plot title to results
    rce2_ttest, rce2_pvalue, ric2_ttest, ric2_pvalue = t_test(
        np.array(correctCongruent['spikeRatesBp']),
        np.array(correctIncongruent['spikeRatesBp']),
        np.array(error['spikeRatesBp'])
    )
    if len(error['trial']) >=5:
        if rce2_ttest is not None and rce2_pvalue is not None:
            ce2_text = f'T-test (correct/error): t={rce2_ttest:.2f}, p={rce2_pvalue:.3f}\n'
        else:
            ce2_text = 'Insufficient data for Correct/Error T-tests'
    else:
        ce2_text = 'Insufficient data for Correct/Error T-tests' 
    if ric2_ttest is not None and ric2_pvalue is not None: 
        ci2_text = f'T-test (incongruent/congruent): t={ric2_ttest:.2f}, p={ric2_pvalue:.3f}'
    else: 
        ci2_text = 'Insufficient data for Incongruent/Congruent T-tests'
    
    ax2.set_title(ce2_text+'\n'+ci2_text)

    # Add vertical grey line at x=0 (stimulus onset)
    ax2.axvline(x=0, color='grey', linewidth=1)
    ax2.set_xlabel('Time from Button Press (s)')
    
    #Create PSTH Plot
    psthTimes=np.arange(-.5,1.5,.02)

    #average calculations stimulus
    ccMovingAvgStimulus=calc_avg(correctCongruent['movingAvgStimulus'], target_shape=len(psthTimes))
    ciMovingAvgStimulus=calc_avg(correctIncongruent['movingAvgStimulus'],target_shape=len(psthTimes))
    errMovingAvgStimulus=calc_avg(error['movingAvgStimulus'],target_shape=len(psthTimes))

    psth.plot(psthTimes,ccMovingAvgStimulus/.2 if ccMovingAvgStimulus is not None else np.zeros_like(psthTimes), color=congruentColor,label='Correct Congruent')
    psth.plot(psthTimes,ciMovingAvgStimulus/.2 if ciMovingAvgStimulus is not None else np.zeros_like(psthTimes), color=incongruentColor, label='Incorrect Congruent')
    psth.plot(psthTimes,errMovingAvgStimulus/.2 if errMovingAvgStimulus is not None else np.zeros_like(psthTimes), color=errorColor, label='Error')
    psth.axvline(x=0, color='grey', linewidth=1)
    psth.set_ylabel('Spike Rate (Hz)')
    psth.set_xlabel('Time from Stimulus (s)')

    #average calculations button press
    ccMovingAvgBp=calc_avg(correctCongruent['movingAvgBp'],target_shape=len(psthTimes))
    ciMovingAvgBp=calc_avg(correctIncongruent['movingAvgBp'],target_shape=len(psthTimes))
    errMovingAvgBp=calc_avg(error['movingAvgBp'],target_shape=len(psthTimes))

    psth2.plot(psthTimes,ccMovingAvgBp/.2 if ccMovingAvgBp is not None else np.zeros_like(psthTimes), color=congruentColor,label='Correct Congruent')
    psth2.plot(psthTimes,ciMovingAvgBp/.2 if ciMovingAvgBp is not None else np.zeros_like(psthTimes), color=incongruentColor, label='Incorrect Congruent')
    psth2.plot(psthTimes,errMovingAvgBp/.2 if errMovingAvgBp is not None else np.zeros_like(psthTimes), color=errorColor, label='Error')
    psth2.axvline(x=0, color='grey', linewidth=1)
    psth2.set_xlabel('Time from Button Press (s)')

    #Roc analysis:
    fpr, tpr, roc_auc = roc_analysis(data['stimSpikeCount'],data['labels'], time_window=0.5)
    fpr2, tpr2, roc_auc2 = roc_analysis(data['bpSpikeCount'],data['labels'], time_window=1.0)
    
    if fpr is not None and tpr is not None: 
        roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        roc.plot([0,1],[0,1], linestyle='--',label='Reference Line')
        roc.set_xlabel('False Positive Rate')
        roc.set_ylabel('True Positive Rate')
        roc.set_title('Logistic Regression: Error/Correct Classification of \n Spike Rates for Stimulus Time Window (0,.5s)')
        roc.legend()
    if fpr2 is not None and tpr2 is not None: 
        roc2.plot(fpr2, tpr2, label=f'ROC Curve (AUC = {roc_auc2:.2f})')
        roc2.plot([0,1],[0,1], linestyle='--',label='Reference Line')
        roc2.set_xlabel('False Positive Rate')
        roc2.set_title('Logistic Regression: Error/Correct Classification of \n Spike Rates for Button Press Window (0,1s)')
        roc2.legend()

    #title the whole figure
    fig.suptitle('Neuron '+os.path.splitext(file_name)[0])

    # Adjust layout
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)  # Increase padding between subplots
    
    if output_folder is not None: 
        plt.savefig(output_folder+os.path.splitext(file_name)[0]+'_Analysis.png',format='png', dpi=100, bbox_inches='tight')
    else:
        plt.savefig(os.path.splitext(file_name)[0]+'_Analysis.png',format='png', dpi=100, bbox_inches='tight')
    
    plt.close(fig)

