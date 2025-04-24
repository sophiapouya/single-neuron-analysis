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
    plot_rasters(ax, correctCongruent['stimulusTimesRaster'], congruentColor, 0)
    plot_rasters(ax, correctIncongruent['stimulusTimesRaster'], incongruentColor, len(correctCongruent['stimulusTimesRaster']))
    plot_rasters(ax, error['stimulusTimesRaster'], errorColor, len(correctCongruent['stimulusTimesRaster']) + len(correctIncongruent['stimulusTimesRaster']))
          
    #perform ttest and set plot title to results
    rce_ttest, rce_pvalue, ric_ttest, ric_pvalue = t_test(np.array(correctCongruent['stimSpikeCount']),np.array(correctIncongruent['stimSpikeCount']),np.array(error['stimSpikeCount']))
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
    plot_rasters(ax2, correctCongruent['bpTimesRaster'], congruentColor, 0)
    plot_rasters(ax2, correctIncongruent['bpTimesRaster'], incongruentColor, len(correctCongruent['bpTimesRaster']))
    plot_rasters(ax2, error['bpTimesRaster'], errorColor, len(correctCongruent['bpTimesRaster']) + len(correctIncongruent['bpTimesRaster']))

    #perform ttest and set plot title to results
    rce2_ttest, rce2_pvalue, ric2_ttest, ric2_pvalue = t_test(
        np.array(correctCongruent['bpSpikeCount']),
        np.array(correctIncongruent['bpSpikeCount']),
        np.array(error['bpSpikeCount'])
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

import numpy as np
from scipy.io import loadmat

def load_data(mat_file):

    #load mat file
    data=loadmat(mat_file)

    #add categories for the following data bins
    data['stimulusTimesRaster'] = []
    data['bpTimesRaster'] = []
    data['labels'], data['labels_ce'],  data['labels_ic'] = [], [], []
    data['trialType'] = []
    data['trialAvgStimulus'] = []
    data['trialAvgBp'] = []
    data['spikeRatesStim'] = [] #this is for the whole time window on the PSTH plot (2 seconds)
    data['spikeRatesBp'] = []   #this is for the whole time window on the PSTH plot (2 seconds)
    data['stimSpikeCount'] = [] #this is over the 1 second after stimulus onset; divide by 0.5 for spike rate
    data['bpSpikeCount'] = []  #Important note: this is ONLY for the button press window and is the same as the total spike count since it's divided by 1 (1 second after button press)

    #flatten the 2D events array
    data['events'] = np.concatenate((data['events'][:, 0], data['events'][:, 1]))

    return data

#function for calculating the moving average
def moving_avg(stimulusStartTime, data, i, stimulusEndTime, binSize, stepSize):
    
    stimulusStartTime = stimulusStartTime - binSize*1_000   #increase time window to prevent dropping off look on graphs
    stimulusEndTime = stimulusEndTime + binSize*1_000

    timeStamp=stimulusStartTime

    trialAvg= []        #numpy array to store 100 values from loop of running averages

    #iterate over the all 100 time stamps between start and end 
    #in addition to 5 time stamps padded onto start and end time
    total_itr = int((stimulusEndTime - stimulusStartTime)/(stepSize*1_000))
    
    for j in range(total_itr):
        binEnd=timeStamp + binSize/2*1_000      #microseconds
        binStart=timeStamp - binSize/2*1_000    #microseconds
        
        if binEnd > stimulusEndTime:
            binEnd=stimulusEndTime
        
        if binStart < stimulusStartTime:
            binStart=stimulusStartTime 
        
        stimulusTimes=data['timestampsOfCell'][(data['timestampsOfCell'] >= binStart) & (data['timestampsOfCell'] <= binEnd)]
        stimulusTimes=stimulusTimes-data['events'][i]
        if (timeStamp>=(stimulusStartTime + binSize*1_000)) and (timeStamp<=(stimulusEndTime - binSize*1_000)):
            if len(trialAvg) < (total_itr - (binSize/stepSize*2)):     
                spikeCount= len(stimulusTimes)
                trialAvg.append(spikeCount)
        timeStamp=timeStamp+stepSize*1_000      #microseconds
    #print(len(trialAvg))
    return trialAvg

def bin_data(data):

    #create dictionarys to store the three categories                   
    correctCongruent =  {'stimulusTimesRaster': [], 'bpTimesRaster': [], 'stimSpikeCount': [], 
                         'bpSpikeCount':[], 'spikeRates':[],'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}     
    correctIncongruent= {'stimulusTimesRaster': [], 'bpTimesRaster': [], 'stimSpikeCount': [], 
                         'bpSpikeCount':[], 'spikeRates':[],'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}
    error = {'stimulusTimesRaster': [], 'bpTimesRaster': [],'stimSpikeCount': [], 
                         'bpSpikeCount':[],'spikeRates':[],'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}
    
    #button press event times were flattened from 2d array to 1d-> need length of data set to correctly index
    bpIndex = int(len(data['events'])/2)

    #global variables
    BIN_SIZE = 200      #200ms
    STEP_SIZE = 20      #20ms
    TIME_WINDOW = 2

    for i in range(len(data['answers'])):
        
        # Determine trial type (1-9)
        word = data['textsPresented'][i]  # 1 = red, 2 = green, 3 = blue
        color = data['colorsPresented'][i]  # 1 = red, 2 = green, 3 = blue
        trial_type = (word - 1) * 3 + color  # Unique label for each trial type (1-9)
        data['trialType'].append(trial_type)

        stimulusStartTime, stimulusEndTime = data['events'][i] - (0.5 * 1_000_000), data['events'][i] + (1.5 * 1_000_000)
        bpStartTime, bpEndTime = data['events'][i + bpIndex] - (0.5 * 1_000_000), data['events'][i + bpIndex] + (1.5 * 1_000_000)

        trialAvgStimulus = moving_avg(stimulusStartTime, data, i, stimulusEndTime, BIN_SIZE, STEP_SIZE)
        data['trialAvgStimulus'].append(trialAvgStimulus)

        trialAvgBp = moving_avg(bpStartTime, data, i, bpEndTime, BIN_SIZE, STEP_SIZE)
        data['trialAvgBp'].append(trialAvgBp)

        #stimulus spike times in time window, relative to event time
        stimulusTimesRaster=data['timestampsOfCell'][(data['timestampsOfCell'] >= stimulusStartTime) & (data['timestampsOfCell'] <= stimulusEndTime)]-data['events'][i]  
        stimTimesSpikeCount=data['timestampsOfCell'][(data['timestampsOfCell'] >= stimulusStartTime +(0.5 * 1_000_000)) & (data['timestampsOfCell'] <= stimulusEndTime-(.5 *1_000_000))]-data['events'][i]

        data['stimulusTimesRaster'].append(stimulusTimesRaster)
        data['stimSpikeCount'].append(len(stimTimesSpikeCount))

        #append the spike rate-> I think it's 2 b/c that's the whole time window
        data['spikeRatesStim'].append(len(stimulusTimesRaster)/TIME_WINDOW)

        #bp spike times in time window, relative to event time
        bpTimesRaster=data['timestampsOfCell'][(data['timestampsOfCell'] >= bpStartTime) & (data['timestampsOfCell'] <= bpEndTime)]-data['events'][i+bpIndex]
        bpTimesSpikeCount=data['timestampsOfCell'][(data['timestampsOfCell'] >= bpStartTime + (0.5 * 1_000_000)) & (data['timestampsOfCell'] <= bpEndTime-(.5*1_000_000))]-data['events'][i+bpIndex]

        data['bpTimesRaster'].append(bpTimesRaster)
        data['bpSpikeCount'].append(len(bpTimesSpikeCount))
        
        #append the spike rate-> I think it's 2 b/c that's the whole time window
        data['spikeRatesBp'].append(len(bpTimesRaster)/TIME_WINDOW)
        
        #correct trial
        if data['answers'][i] == data['colorsPresented'][i]:
            data['labels_ce'].append(1)
            data['labels'].append(1)

            if (data['colorsPresented'][i]==data['textsPresented'][i]):
                #correct congruent trial
                data['labels_ic'].append(1)
                correctCongruent['trial'].append(i)
                correctCongruent['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
                correctCongruent['bpTimesRaster'].append(data['bpTimesRaster'][i])
                correctCongruent['stimSpikeCount'].append(data['stimSpikeCount'][i])
                correctCongruent['bpSpikeCount'].append(data['bpSpikeCount'][i])
                correctCongruent['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
                correctCongruent['movingAvgBp'].append(data['trialAvgBp'][i])

            else:
                #incorrect congruent
                data['labels_ic'].append(0)
                correctIncongruent['trial'].append(i)
                correctIncongruent['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
                correctIncongruent['bpTimesRaster'].append(data['bpTimesRaster'][i])
                correctIncongruent['stimSpikeCount'].append(data['stimSpikeCount'][i])
                correctIncongruent['bpSpikeCount'].append(data['bpSpikeCount'][i])
                correctIncongruent['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
                correctIncongruent['movingAvgBp'].append(data['trialAvgBp'][i])
        
        else:
            #add label for incorrect trial-> 0 (roc curve)
            data['labels_ce'].append(0)
            data['labels'].append(0)
            error['trial'].append(i)
            #error trials -> add whatever other data needed here
            error['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
            error['bpTimesRaster'].append(data['bpTimesRaster'][i])
            error['stimSpikeCount'].append(data['stimSpikeCount'][i])
            error['bpSpikeCount'].append(data['bpSpikeCount'][i])
            error['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
            error['movingAvgBp'].append(data['trialAvgBp'][i])
    

    return data, error, correctCongruent, correctIncongruent

#function to calculate the average across all trials stored in a dictionary
def calc_avg(moving_avg, target_shape):

    moving_avg_arr= np.array(moving_avg)
    
    if moving_avg_arr.size>0:
        moving_avg_arr=np.mean(moving_avg_arr, axis=0)
    else:
        print("no data available for averaging")
        return np.zeros(target_shape)

    return moving_avg_arr

def t_test(cc, ci, err):
    from scipy.stats import ttest_ind, levene

    cc, ci, err = np.array(cc), np.array(ci), np.array(err)
    if cc.size == 0 or ci.size == 0:
        print("Insufficient data for congruent/incongruent trials.")
        return None, None, None, None

    # Test for variance equality
    _, p_levene = levene(cc, ci)
    equal_var = p_levene >= 0.05

    print(f"Levene's test p-value: {p_levene}")
    print(f"Using {'equal' if equal_var else 'unequal'} variances for t-test.")

    # Perform t-tests
    correct = np.concatenate([cc, ci])
    if err.size >= 5 and correct.size >= 5:
        ce_tt, ce_pv = ttest_ind(correct, err, equal_var=equal_var)
    else:
        ce_tt, ce_pv = None, None

    if cc.size >= 5 and ci.size >= 5:
        ic_tt, ic_pv = ttest_ind(ci, cc, equal_var=equal_var)
    else:
        ic_tt, ic_pv = None, None

    return ce_tt, ce_pv, ic_tt, ic_pv