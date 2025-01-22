from collections import defaultdict
from typing import Optional, Literal

from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

import numpy as np
import pandas as pd


def format_spike(spikes):
    # Group spikes by neuron
    neuron_spikes = defaultdict(list)
    for t, c in zip(spikes.times, spikes.clusters):
        neuron_spikes[c].append(t)

    # Convert defaultdict to regular dict (optional)
    neuron_spikes = dict(neuron_spikes)
    # Convert lists to NumPy arrays for fast operations
    for neuron, spike_times in neuron_spikes.items():
        neuron_spikes[neuron] = np.array(spike_times)
    
    return neuron_spikes

def extract_spike(trials, neuron_spikes, baseline_start:float = -0.5):
    '''
    Extracts spikes for each neuron in the trials dataframe. 
    Returns a dictionary of aligned spikes and baseline spikes.
    Dictionary keys are neuron IDs and values are lists of lists of spike times.
    
    Args:
        trials (pd.DataFrame): The trials dataframe.
        neuron_spikes (dict): Dictionary of neuron spikes.
        pre_stimulus_window (float): The pre-stimulus window duration.
        
    Returns:
        dict: Dictionary of aligned spikes.
        dict: Dictionary of baseline spikes.
    
    '''
    stimulus_spikes = defaultdict(list)
    baseline_spikes = defaultdict(list)

    for index, row in trials.iterrows():
        stim_onset = row["timeline_audPeriodOn"]
        stim_offset = stim_onset + 0.5
        baseline_onset = stim_onset + baseline_start
        for neuron, spike_times in neuron_spikes.items():        
            trial_spikes = spike_times[(spike_times >= stim_onset) & (spike_times <= stim_offset)] - stim_onset
            stimulus_spikes[neuron].append(trial_spikes.tolist())
            
            trial_baseline = spike_times[(spike_times >= baseline_onset) & (spike_times < stim_onset)] - stim_onset
            baseline_spikes[neuron].append(trial_baseline.tolist())
            

    stimulus_spikes = dict(stimulus_spikes)
    baseline_spikes = dict(baseline_spikes)
    
    return stimulus_spikes, baseline_spikes

def calculate_firing_rate(spike_times_dict, bin_size:float, period_start: Optional[float] = 0, period_end: Optional[float] = None, time_window: Optional[float] = 1):
    """
    Calculate firing rates for neurons by counting spikes in small time bins.
    
    Parameters:
    - spike_times_dict: Dictionary where keys are neuron IDs and values are lists of spike times.
    - time_window: The total duration over which to calculate firing rates (in seconds).
    - bin_size: The size of the time bins for counting spikes (in seconds).
    - period_start: Start time of the period for firing rate calculation.
    - period_end: End time of the period for firing rate calculation (None means until the end of the time window).
    
    Returns:
    - firing_rates: Dictionary where keys are neuron IDs and values are lists of firing rates for each bin.
    """
    firing_rates = defaultdict(list)

    # If period_end is not specified, use the time window duration
    if period_end is None:
        period_end = time_window + period_start
        
    times = np.arange(period_start, period_end, bin_size)
    
    # Iterate through each neuron
    for neuron_id in spike_times_dict.keys():
        all_spike_counts = []
        for spike_times in spike_times_dict[neuron_id]:
            
            # Convert to numpy array
            spike_times = np.array(spike_times)

            # Calculate number of bins
            num_bins = int((period_end - period_start) / bin_size)
            # Create an array to hold spike counts for each bin
            spike_counts = np.zeros(num_bins)
            # Count spikes in each bin
            for t in spike_times:
                if period_start <= t < period_end:
                    bin_index = int((t - period_start) // bin_size)  # Calculate which bin the spike belongs to
                    if bin_index < num_bins:
                        spike_counts[bin_index] += 1
            spike_counts = spike_counts / bin_size  # Convert spike counts to firing rates
            all_spike_counts.append(spike_counts)
        firing_rates[neuron_id] = np.array(all_spike_counts)

    return dict(firing_rates), times


def test_fr_change_stimulus(obj, parametric_test: bool = False, one_sided: bool = False, 
                            up_regulated: bool = False, multiple_correction: bool = True, mc_test: str = 'fdr_bh', 
                            mc_alpha: float = 0.05):
    '''
    Test for significant changes in firing rates between baseline and stimulus periods.
    
    Args:
        obj (object): The object containing the formatted data.
        parametric_test (bool): Whether to use a parametric test (default: False).
        one_sided (bool): Whether to use a one-sided test (default: False).
        up_regulated (bool): Whether to test for up-regulation (default: False).
        multiple_correction (bool): Whether to apply multiple testing correction (default: True).
        mc_test (str): The multiple testing correction method (default: 'fdr_bh').
        mc_alpha (float): The significance level for multiple testing correction (default: 0.05).
        
    Returns:
        dict: A dictionary containing p-values, test statistics, and significance status for each neuron.
    '''
    p_vals = defaultdict(list)
    all_p_vals = []
    n_significant = 0
    # Loop through neurons to compare baseline and stimulus firing rates
    for neuron_id in obj.formatted_data:
        baseline_rate = obj.formatted_data[neuron_id]['firing_rate']['baseline'].mean(axis = 1)
        stimulus_rate = obj.formatted_data[neuron_id]['firing_rate']['stimulus'].mean(axis = 1)

        # Check if baseline and stimulus rates have the same length
        min_length = min(len(baseline_rate), len(stimulus_rate))
        baseline_rate = baseline_rate[:min_length]
        stimulus_rate = stimulus_rate[:min_length]

        if parametric_test:
            if one_sided:
                if up_regulated:
                    # Perform one-sided t-test
                    stat, p_val = ttest_rel(baseline_rate, stimulus_rate, alternative='greater')
                else:
                    # Perform one-sided t-test
                    stat, p_val = ttest_rel(baseline_rate, stimulus_rate, alternative='less') 
            else:
                # Perform paired t-test or Wilcoxon signed-rank test
                stat, p_val = ttest_rel(baseline_rate, stimulus_rate)      
        else: # Non-parametric test
            if one_sided:
                if up_regulated:
                    # Perform one-sided Wilcoxon test
                    stat, p_val = wilcoxon(baseline_rate, stimulus_rate, alternative='greater')
                else:
                    # Perform one-sided Wilcoxon test
                    stat, p_val = wilcoxon(baseline_rate, stimulus_rate, alternative='less')
            else:
                # Alternatively, use the following for the Wilcoxon test:
                stat, p_val = wilcoxon(baseline_rate, stimulus_rate)
        
        p_vals[neuron_id] = (p_val, stat, p_val < 0.05)
        if p_val < 0.05:
            n_significant += 1
        all_p_vals.append(p_val)

    # Apply multiple testing correction if specified
    n_significant_mc = 0
    if multiple_correction:
        rejected, corrected_p_vals, _, _ = multipletests(all_p_vals, alpha=mc_alpha, method=mc_test)
        n_significant_mc = rejected.sum()
        # Update p_vals with corrected p-values and significance status
        for idx, neuron_id in enumerate(p_vals.keys()):
            original_p_val, stat, _ = p_vals[neuron_id]
            p_vals[neuron_id] = (corrected_p_vals[idx], stat, rejected[idx])
    n_total = len(p_vals)
    p_vals['info'] = {'n_significant': n_significant, 'n_significant_mc': n_significant_mc, 'n_total' : n_total,
                    'multiple_correction': multiple_correction, 'mc_test': mc_test, 'mc_alpha': mc_alpha, 
                    'one_sided': one_sided, 'up_regulated': up_regulated, 'mask_info': obj.mask_info}
    p_vals = dict(p_vals)
    return p_vals