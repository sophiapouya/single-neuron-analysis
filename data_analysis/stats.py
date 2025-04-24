import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc

#function to calculate the average across all trials stored in a dictionary
def calc_avg(moving_avg, target_shape):

    moving_avg_arr= np.array(moving_avg)
    
    if moving_avg_arr.size>0:
        moving_avg_arr=np.mean(moving_avg_arr, axis=0)
    else:
        print("no data available for averaging")
        return np.zeros(target_shape)

    return moving_avg_arr

def roc_analysis(data_spike_rates, data_labels, time_window):
    spikeRatesStim=np.array(data_spike_rates)/time_window
    labels=np.array(data_labels)
    
    if labels.size ==0 or spikeRatesStim.size==0:
        print("Empty data, ROC analysis cannot be performed.")
        return None, None, None

    #Check if labels contain at least two class (some trials had no errors)
    if len(np.unique(labels)) < 2:
        print('Only one type found in labels, ROC analysis not performed')
        return None, None, None
    
    #choose a logistic regression classifer for a binary choice-> error or correct based on the spike rates
    classifier = LogisticRegression()
    classifier.fit(spikeRatesStim.reshape(-1,1),labels)
    probabilities = classifier.predict_proba(spikeRatesStim.reshape(-1,1))[:,1]

    fpr, tpr, thresholds = roc_curve(labels,probabilities)
    roc_auc=auc(fpr, tpr)

    return fpr, tpr, roc_auc


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

