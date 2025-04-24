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
    correctCongruent =  {'stimulusTimesRaster': [], 'bpTimesRaster': [], 'stimSpikeCount': [], 'stimulusRasterSpikes': [],
                         'bpSpikeCount':[], 'spikeRatesStim':[],'spikeRatesBp':[], 'bpRasterSpikes': [],
                         'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}     
    correctIncongruent= {'stimulusTimesRaster': [], 'bpTimesRaster': [], 'stimSpikeCount': [], 'stimulusRasterSpikes': [],
                         'bpSpikeCount':[], 'spikeRatesStim':[],'spikeRatesBp':[], 'bpRasterSpikes': [],
                         'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}
    error = {'stimulusTimesRaster': [], 'bpTimesRaster': [],'stimSpikeCount': [], 'stimulusRasterSpikes': [],
                         'bpSpikeCount':[],'spikeRatesStim':[],'spikeRatesBp':[], 'bpRasterSpikes': [],
                         'movingAvgStimulus':[],'movingAvgBp':[],'trial':[]}
    
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
        # data['stimulusRasterSpikes'].append(len(stimulusTimesRaster))
        data['stimSpikeCount'].append(len(stimTimesSpikeCount))

        #append the spike rate-> I think it's 2 b/c that's the whole time window
        data['spikeRatesStim'].append(len(stimulusTimesRaster)/TIME_WINDOW)

        #bp spike times in time window, relative to event time
        bpTimesRaster=data['timestampsOfCell'][(data['timestampsOfCell'] >= bpStartTime) & (data['timestampsOfCell'] <= bpEndTime)]-data['events'][i+bpIndex]
        bpTimesSpikeCount=data['timestampsOfCell'][(data['timestampsOfCell'] >= bpStartTime + (0.5 * 1_000_000)) & (data['timestampsOfCell'] <= bpEndTime-(.5*1_000_000))]-data['events'][i+bpIndex]

        data['bpTimesRaster'].append(bpTimesRaster)
        # data['bpRasterSpikes'].append(len(bpTimesRaster))
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
                correctCongruent['spikeRatesStim'].append(data['spikeRatesStim'])
                correctCongruent['spikeRatesBp'].append(data['spikeRatesBp'])
                correctCongruent['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
                correctCongruent['stimulusRasterSpikes'].append(len(data['stimulusTimesRaster'][i]))
                correctCongruent['bpTimesRaster'].append(data['bpTimesRaster'][i])
                correctCongruent['bpRasterSpikes'].append(len(data['bpTimesRaster'][i]))
                correctCongruent['stimSpikeCount'].append(data['stimSpikeCount'][i])
                correctCongruent['bpSpikeCount'].append(data['bpSpikeCount'][i])
                correctCongruent['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
                correctCongruent['movingAvgBp'].append(data['trialAvgBp'][i])

            else:
                #incorrect congruent
                data['labels_ic'].append(0)
                correctIncongruent['trial'].append(i)
                correctIncongruent['spikeRatesStim'].append(data['spikeRatesStim'])
                correctIncongruent['spikeRatesBp'].append(data['spikeRatesBp'])
                correctIncongruent['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
                correctIncongruent['stimulusRasterSpikes'].append(len(data['stimulusTimesRaster'][i]))
                correctIncongruent['bpTimesRaster'].append(data['bpTimesRaster'][i])
                correctIncongruent['bpRasterSpikes'].append(len(data['bpTimesRaster'][i]))
                correctIncongruent['stimSpikeCount'].append(data['stimSpikeCount'][i])
                correctIncongruent['bpSpikeCount'].append(data['bpSpikeCount'][i])
                correctIncongruent['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
                correctIncongruent['movingAvgBp'].append(data['trialAvgBp'][i])
        
        else:
            #add label for incorrect trial-> 0 (roc curve)
            data['labels_ce'].append(0)
            data['labels'].append(0)
            error['trial'].append(i)
            error['spikeRatesStim'].append(data['spikeRatesStim'])
            error['spikeRatesBp'].append(data['spikeRatesBp'])

            #error trials -> add whatever other data needed here
            error['stimulusTimesRaster'].append(data['stimulusTimesRaster'][i])
            error['stimulusRasterSpikes'].append(len(data['stimulusTimesRaster'][i]))
            error['bpTimesRaster'].append(data['bpTimesRaster'][i])
            error['stimSpikeCount'].append(data['stimSpikeCount'][i])
            error['bpSpikeCount'].append(data['bpSpikeCount'][i])
            error['bpRasterSpikes'].append(len(data['bpTimesRaster'][i]))
            error['movingAvgStimulus'].append(data['trialAvgStimulus'][i])
            error['movingAvgBp'].append(data['trialAvgBp'][i])
    

    return data, error, correctCongruent, correctIncongruent
