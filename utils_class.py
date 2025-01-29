from collections import defaultdict
from typing import Optional, Literal

from scipy.linalg import eigh
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

def calculate_firing_rate(spike_times_dict, bin_size:float, period_start: Optional[float] = 0, period_end: Optional[float] = None, time_window: Optional[float] = 1, normalise = True):
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
            if normalise:
                pass
                # means = np.mean(spike_counts)
                # std_devs = np.std(spike_counts)
                # spike_counts = (spike_counts - means) / std_devs
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
    # store_p_vals = np.empty([len(obj.formatted_data), 2])
    # Loop through neurons to compare baseline and stimulus firing rates
    for neuron_id in obj.formatted_data:
        baseline_rate_all = obj.formatted_data[neuron_id]['firing_rate']['baseline'].mean(axis = 1)
        baseline_rate = baseline_rate_all[obj.mask_interest_indices]
        stimulus_rate_all = obj.formatted_data[neuron_id]['firing_rate']['stimulus'].mean(axis = 1)
        stimulus_rate = stimulus_rate_all[obj.mask_interest_indices]

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
        # store_p_vals[neuron_id] = [neuron_id, p_val]
        
    # Apply multiple testing correction if specified
    n_significant_mc = 0
    if multiple_correction:
        rejected, corrected_p_vals, _, _ = multipletests(all_p_vals, alpha=mc_alpha, method=mc_test)
        n_significant_mc = rejected.sum()
        # Update p_vals with corrected p-values and significance status
        for idx, neuron_id in enumerate(p_vals.keys()):
            _, stat, _ = p_vals[neuron_id]
            
            p_vals[neuron_id] = (corrected_p_vals[idx], stat, rejected[idx])
            # store_p_vals[neuron_id] = [neuron_id, corrected_p_vals[idx]]
            
    n_total = len(p_vals)
    p_vals['info'] = {'n_significant': n_significant, 'n_significant_mc': n_significant_mc, 'n_total' : n_total,
                    'multiple_correction': multiple_correction, 'mc_test': mc_test, 'mc_alpha': mc_alpha, 
                    'one_sided': one_sided, 'up_regulated': up_regulated, 'mask_info': obj.mask_info}
    p_vals = dict(p_vals)
    return p_vals


def calculate_csp(obj):
	X_baseline_all = obj.formatted_data_array['firing_rate']['baseline'] # neuron - trial - fr step
	X_stimulus_all = obj.formatted_data_array['firing_rate']['stimulus']
	X_baseline_all_avg = np.mean(X_baseline_all, axis=1) # average firing rate of neuron X over all trials
	X_stimulus_all_avg = np.mean(X_stimulus_all, axis=1)

	# 1. Compute covariance matrices for baseline and stimulus
	C_baseline = np.cov(X_baseline_all_avg.T)
	C_stimulus = np.cov(X_stimulus_all_avg.T)

	# 2. Perform eigenvalue decomposition to get CSP filters
	eigenvalues, eigenvectors = eigh(C_baseline, C_stimulus)

	# 3. Sort the eigenvalues and eigenvectors
	sorted_indices = np.argsort(eigenvalues)[::-1]
	eigenvectors_sorted = eigenvectors[:, sorted_indices]

	# 4. Project the data onto the CSP space (top filters)
	X_baseline_csp = np.dot(X_baseline_all_avg, eigenvectors_sorted[:, :2])  
	X_stimulus_csp = np.dot(X_stimulus_all_avg, eigenvectors_sorted[:, :2])

	# 5. Calculate variance for each neuron in the CSP space
	baseline_variance = np.var(X_baseline_csp, axis=1)
	stimulus_variance = np.var(X_stimulus_csp, axis=1)

	# 6. Compare variances to identify neurons with significant changes
	variance_ratio = stimulus_variance / baseline_variance  # Ratio of stimulus to baseline variance

	# 7. Neurons with a high variance ratio are considered responsive to the stimulus
	responsive_neurons = np.where(variance_ratio > 1.5)[0]  # Choose an appropriate threshold

	mean_baseline = np.mean(X_baseline_all_avg, axis=1)  # Mean firing rate of neuron X_avg(over trials) during baseline
	mean_stimulus = np.mean(X_stimulus_all_avg, axis=1)  # Mean firing rate of neuron X_avg(over trials) during stimulus
	increased_neurons = np.array([neuron for neuron in responsive_neurons if mean_stimulus[neuron] > mean_baseline[neuron]])
	decreased_neurons = np.array([neuron for neuron in responsive_neurons if mean_stimulus[neuron] < mean_baseline[neuron]])
 
	return variance_ratio, increased_neurons, decreased_neurons


def calculate_ccsp(obj, reg_param=1e-5):
	X_baseline_all = obj.formatted_data_array['firing_rate']['baseline'] # neuron - trial - fr step
	X_stimulus_all = obj.formatted_data_array['firing_rate']['stimulus']
	X_baseline_all_avg = np.mean(X_baseline_all, axis=1) # average firing rate of neuron X over all trials
	X_stimulus_all_avg = np.mean(X_stimulus_all, axis=1)
 
	# 1. Compute correlation matrices
	R_baseline = np.corrcoef(X_baseline_all.reshape(786, -1))  # Neurons x Neurons
	R_stimulus = np.corrcoef(X_stimulus_all.reshape(786, -1))  # Neurons x Neurons

	R_baseline = np.nan_to_num(R_baseline, nan=0.0, posinf=0.0, neginf=0.0)
	R_stimulus = np.nan_to_num(R_stimulus, nan=0.0, posinf=0.0, neginf=0.0)

	R_baseline += reg_param * np.eye(R_baseline.shape[0])
	R_stimulus += reg_param * np.eye(R_stimulus.shape[0])


	# 2. Compute eigenvalue decomposition for CCSP
	eigenvalues, eigenvectors = eigh(R_stimulus, R_baseline)

	# eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(R_baseline) @ R_stimulus)

	# 3. Sort eigenvalues and eigenvectors
	idx = np.argsort(eigenvalues)[::-1]
	eigenvectors = eigenvectors[:, idx]

	# 4. Apply spatial filters to project data
	W = eigenvectors  # Spatial filters
	Y_baseline = W.T @ X_baseline_all.reshape(786, -1)
	Y_stimulus = W.T @ X_stimulus_all.reshape(786, -1)

	# 5. Analyze variance changes
	variance_baseline = np.var(Y_baseline, axis=1)
	variance_stimulus = np.var(Y_stimulus, axis=1)
	variance_ratio = variance_stimulus / variance_baseline

	# Find neurons/components with significant changes
	responsive_components = np.where(variance_ratio > 1.5)[0]
	mean_baseline = np.mean(X_baseline_all_avg, axis=1)  # Mean firing rate of neuron X_avg(over trials) during baseline
	mean_stimulus = np.mean(X_stimulus_all_avg, axis=1)  # Mean firing rate of neuron X_avg(over trials) during stimulus
	increased_neurons = np.array([neuron for neuron in responsive_components if mean_stimulus[neuron] > mean_baseline[neuron]])
	decreased_neurons = np.array([neuron for neuron in responsive_components if mean_stimulus[neuron] < mean_baseline[neuron]])

	return variance_ratio, increased_neurons, decreased_neurons

def calculate_ttest(obj,parametric_test = True,one_sided = True, multiple_correction = True,
                    mc_test:Literal['bonferroni', 'sidak', 'holm', 'fdr_bh'] = 'bonferroni', mc_alpha = 0.05):
	up_regulated = True
	obj.test_firing_rate_change_stimulus(parametric_test, one_sided, up_regulated, multiple_correction, mc_test, mc_alpha)
	# Initialize an empty list to store neuron IDs with significant results
	increased_neurons = []
	# Iterate through all items in obj.pvals
	for neuron_id, value in obj.p_vals.items():
		# Check if the value is a tuple with 3 elements
		if len(value) == 3:
			# Unpack the tuple into x, y, and significance
			_,_, significance = value
			# Check if the significance is True
			if significance:
				# If significance is True, append the neuron_id to the list
				increased_neurons.append(neuron_id)
	if one_sided:
		up_regulated = False
		obj.test_firing_rate_change_stimulus(parametric_test, one_sided, up_regulated, multiple_correction, mc_test, mc_alpha)
		# Initialize an empty list to store neuron IDs with significant results
		decreased_neurons = []
		# Iterate through all items in obj.pvals
		for neuron_id, value in obj.p_vals.items():
			# Check if the value is a tuple with 3 elements
			if len(value) == 3:
				# Unpack the tuple into x, y, and significance
				_,_, significance = value
				# Check if the significance is True
				if significance:
					# If significance is True, append the neuron_id to the list
					decreased_neurons.append(neuron_id)
	return _, increased_neurons, decreased_neurons
		