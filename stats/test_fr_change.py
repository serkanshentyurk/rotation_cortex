import numpy as np

from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from typing import Literal, Optional

def prep_ttest(obj, stimulus_start_second, stimulus_until_second):
	baseline_spikes = obj.interest_data['firing_rate_baseline'][:,:,:].mean(axis=2)
	stimulus_spikes = obj.interest_data['firing_rate_stimulus'][:,:,int(100*stimulus_start_second):int(100*stimulus_until_second)].mean(axis=2)
	return baseline_spikes, stimulus_spikes

def test_ttest(baseline_rate,stimulus_rate, parametric_test = True, one_sided = 'increased',
                         multiple_correction = 'bonferroni', mc_alpha = 0.05):
    if parametric_test:
        if one_sided is False:
            stat, p_val = ttest_rel(baseline_rate, stimulus_rate)
        elif one_sided in ['increased', 'upregulated']:
            # Perform one-sided t-test
            stat, p_val = ttest_rel(baseline_rate, stimulus_rate, alternative='greater')
        elif one_sided in ['decreased', 'downregulated']:
            # Perform one-sided t-test
            stat, p_val = ttest_rel(baseline_rate, stimulus_rate, alternative='less')
        else:
            raise ValueError('Invalid value for one_sided. Must be False, increased, or decreased.')
    else: # Non-parametric test
        if one_sided is False:
            # Perform two-sided Wilcoxon test
            stat, p_val = wilcoxon(baseline_rate, stimulus_rate)
        elif one_sided in ['increased', 'upregulated']: 
            # Perform one-sided Wilcoxon test
            stat, p_val = wilcoxon(baseline_rate, stimulus_rate, alternative='greater')
        elif one_sided in ['decreased', 'downregulated']:
            # Perform one-sided Wilcoxon test
            stat, p_val = wilcoxon(baseline_rate, stimulus_rate, alternative='less')
        else:
            raise ValueError('Invalid value for one_sided. Must be False, increased, or decreased.')
    n_significant = np.sum(p_val < 0.05)  
          
    if multiple_correction is False:
        rejected = None
        corrected_p_vals = None
        n_significant_mc = None
    else:
        rejected, corrected_p_vals, _, _ = multipletests(p_val, alpha=mc_alpha, method=multiple_correction)
        n_significant_mc = rejected.sum()
    return stat, p_val, rejected, corrected_p_vals, n_significant, n_significant_mc

def test_fr_change_ttest(self, stimulus_start_second, stimulus_until_second,
                         parametric_test = True, 
                         one_sided = False,
                         multiple_correction = 'bonferroni'):
    baseline_spikes, stimulus_spikes = prep_ttest(self, stimulus_start_second, stimulus_until_second)
    n_trials = baseline_spikes.shape[1]
    stat, p_val, rejected, corrected_p_vals, n_significant, n_significant_mc = test_ttest(baseline_spikes.T,
                                                                                          stimulus_spikes.T, 
                                                                                          parametric_test = parametric_test, 
                                                                                          one_sided = one_sided, 
                                                                                          multiple_correction = multiple_correction)
    return stat, p_val, rejected, corrected_p_vals, n_significant, n_significant_mc, n_trials