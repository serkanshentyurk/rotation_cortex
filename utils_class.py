from collections import defaultdict
from typing import Optional, Literal

from scipy.linalg import eigh
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from stats import *

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


def select_trials(obj, blank_trial = None, visual_trial = None, auditory_trial = None, coherent_trial = None, conflict_trial = None,
                  vis_stim_loc = None, aud_stim_loc = None, choice = None):
	trials = obj.trials_formatted
	if blank_trial:
		trials = trials[trials.is_blankTrial]
	elif blank_trial == False:
		trials = trials[~trials.is_blankTrial]
	if visual_trial:
		trials = trials[trials.is_visualTrial]
	elif visual_trial == False:
		trials = trials[~trials.is_visualTrial]
	if auditory_trial:
		trials = trials[trials.is_auditoryTrial]
	elif auditory_trial == False:
		trials = trials[~trials.is_auditoryTrial]
	if coherent_trial:
		trials = trials[trials.is_coherentTrial]
	elif coherent_trial == False:
		trials = trials[~trials.is_coherentTrial]
	if conflict_trial:
		trials = trials[trials.is_conflictTrial]
	elif conflict_trial == False:
		trials = trials[~trials.is_conflictTrial]
	if vis_stim_loc in ['left', 'l']:
		trials = trials[trials.vis_loc == 'l']
	elif vis_stim_loc in ['right', 'r']:
		trials = trials[trials.vis_loc == 'r']
	elif vis_stim_loc in ['c', 'o', 'center', 'centre', 'off']:
		trials = trials[trials.vis_loc == 'o']
	if aud_stim_loc in ['left', 'l']:
		trials = trials[trials.aud_loc == 'l']
	elif aud_stim_loc in ['right', 'r']:
		trials = trials[trials.aud_loc == 'r']
	elif aud_stim_loc in ['c', 'o', 'center', 'centre', 'off']:
		trials = trials[trials.aud_loc == 'c']
	if choice in ['left', 'l', 0]:
		trials = trials[trials.choice == 0]
	elif choice in ['right', 'r', 1]:
		trials = trials[trials.choice == 1]
	elif choice in ['no', 'n', -1]:
		trials = trials[trials.choice == -1]
	return trials, np.array(trials.index)


def test_fr_change_ttest(baseline_rate,stimulus_rate, parametric_test = True, one_sided = True, up_regulated = True,
                         multiple_correction = True, mc_alpha = 0.05, mc_method = 'fdr_bh'):  
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
    n_significant = np.sum(p_val < 0.05)  
          
    if multiple_correction:
        rejected, corrected_p_vals, _, _ = multipletests(p_val, alpha=mc_alpha, method=mc_method)
        n_significant_mc = rejected.sum()
    else:
        rejected = None
        n_significant_mc = None
        corrected_p_vals = None
    return stat, p_val, rejected, corrected_p_vals, n_significant, n_significant_mc


def calculate_ttest(obj,parametric_test = True,one_sided = True, multiple_correction = True,
                    mc_test:Literal['bonferroni', 'sidak', 'holm', 'fdr_bh'] = 'bonferroni', mc_alpha = 0.05):
	up_regulated = True
	obj.test_firing_rate_change_stimulus(parametric_test, one_sided, up_regulated, multiple_correction, mc_test, mc_alpha)
	increased_neurons = []
	increased_p_vals = []
 
	# Iterate through all items in obj.pvals
	for neuron_id, value in obj.p_vals.items():
		# Check if the value is a tuple with 3 elements
		if len(value) == 3:
			# Unpack the tuple into x, y, and significance
			p_value,_, significance = value
			# Check if the significance is True
			if significance:
				# If significance is True, append the neuron_id to the list
				increased_neurons.append(neuron_id)
				increased_p_vals.append(p_value)
	if one_sided:
		up_regulated = False
		obj.test_firing_rate_change_stimulus(parametric_test, one_sided, up_regulated, multiple_correction, mc_test, mc_alpha)
		# Initialize an empty list to store neuron IDs with significant results
		decreased_neurons = []
		decreased_p_vals = []
		# Iterate through all items in obj.pvals
		for neuron_id, value in obj.p_vals.items():
			# Check if the value is a tuple with 3 elements
			if len(value) == 3:
				# Unpack the tuple into x, y, and significance
				p_value,_, significance = value
				# Check if the significance is True
				if significance:
					# If significance is True, append the neuron_id to the list
					decreased_neurons.append(neuron_id)
					decreased_p_vals.append(p_value)
	return [increased_p_vals, decreased_p_vals], increased_neurons, decreased_neurons
		

def calculate_cccp_ccsp(obj):
	p_of_choice_probability = []
	aud_p_of_stim_probability = []
	vis_p_of_stim_probability = []
	trial_choice = obj.trials_formatted['choice'].values

	for neuron_idx in range(len(obj.formatted_data)):
		neuron = obj.formatted_data[neuron_idx]
		spike_counts = []
		for trial_idx in range(len(neuron['spikes']['baseline'])):
			spike_times_relative = np.array(neuron['spikes']['baseline'][trial_idx] + neuron['spikes']['stimulus'][trial_idx])
			n_spike_per_trial = len(spike_times_relative)
			spike_counts.append(n_spike_per_trial)
		spike_counts = np.array(spike_counts)
  
		_, p_of_choice_prob = cal_combined_conditions_choice_prob_numpy(spike_counts,
																			obj.trials_formatted['cond_id'],
																			trial_choice,
																			verbose = False)
  
		_, vis_p_of_stim_prob = cal_combined_conditions_choice_stim_numpy(spike_counts,
																		obj.trials_formatted['aud_loc'].values,
																		obj.trials_formatted['vis_loc'].values,
																		trial_choice,
																		unique_vis_cond = ['l', 'r'],
																		unique_aud_cond = ['l', 'r'],
																		unique_choice_cond = [-1,0,1],
																		verbose=False,
																		test_type='visLeftRight')
		_, aud_p_of_stim_prob = cal_combined_conditions_choice_stim_numpy(spike_counts,
																		obj.trials_formatted['aud_loc'].values,
																		obj.trials_formatted['vis_loc'].values,
																		trial_choice,
																		unique_vis_cond = ['l', 'r'],
																		unique_aud_cond = ['l', 'r'],
																		unique_choice_cond = [-1,0,1],
																		verbose=False,
																		test_type='audLeftRight')
		p_of_choice_probability.append(p_of_choice_prob)
		aud_p_of_stim_probability.append(aud_p_of_stim_prob)
		vis_p_of_stim_probability.append(vis_p_of_stim_prob)

	p_of_choice_probability = np.array(p_of_choice_probability)[np.where(obj.all_cluster_data.bombcell_class == 'good')[0]]
	aud_p_of_stim_probability = np.array(aud_p_of_stim_probability)[np.where(obj.all_cluster_data.bombcell_class == 'good')[0]]
	vis_p_of_stim_probability = np.array(vis_p_of_stim_probability)[np.where(obj.all_cluster_data.bombcell_class == 'good')[0]]
 
	return p_of_choice_probability, aud_p_of_stim_probability, vis_p_of_stim_probability


