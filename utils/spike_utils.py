import numpy as np
from collections import defaultdict
from typing import Optional, Literal

def standardise_fr(fr_baseline, fr_stimulus):
    '''
    Standardises the firing rates for each neuron.
    
    Args:
        fr_baseline (np.ndarray): The baseline firing rates.
        fr_stimulus (np.ndarray): The stimulus firing rates.
        
    Returns:
        tuple: A tuple containing the standardised baseline firing rates and the standardised stimulus firing rates.
    '''
    
    baseline_flat = fr_baseline.reshape((fr_baseline.shape[0], -1))
    
    # Calculate the mean and standard deviation of the baseline firing rates
    baseline_means = np.mean(baseline_flat, axis=1)
    baseline_stds = np.std(baseline_flat, axis=1)
    
    # Avoid division by zero
    zero_var_neurons = baseline_stds == 0
    baseline_stds[zero_var_neurons] = 1

    # Standardise the firing rates
    standardised_fr_baseline = (fr_baseline - baseline_means[:, None, None]) / baseline_stds[:, None, None]
    standardised_fr_stimulus = (fr_stimulus - baseline_means[:, None, None]) / baseline_stds[:, None, None]
    
    # Set the firing rates of neurons with zero variance to zero
    standardised_fr_baseline[zero_var_neurons, :, :] = 0
    standardised_fr_stimulus[zero_var_neurons, :, :] = 0
    
    return standardised_fr_baseline, standardised_fr_stimulus

    
def format_spike(spikes, bombcell_keys):
    '''
    Iterates through the spikes dataframe and groups spikes by neuron.
    Returns a dictionary where keys are neuron IDs and values are NumPy arrays of spike times.
    
    Args:
        spikes (pd.DataFrame): The spikes dataframe.
        bombcell_keys (list): List of bombcell keys.
        
    Returns:
        dict: Dictionary of neuron spikes.
    '''
    # Group spikes by neuron
    neuron_spikes = defaultdict(list)
    for t, c in zip(spikes['times'], spikes['clusters']):
        neuron_spikes[c].append(t)

    # Convert defaultdict to regular dict (optional)
    neuron_spikes = dict(neuron_spikes)
    neuron_spikes_copy = neuron_spikes.copy()
    # Convert lists to NumPy arrays for fast operations
    for neuron, spike_times in neuron_spikes_copy.items():
        if neuron in bombcell_keys:
            neuron_spikes[neuron] = np.array(spike_times)
        else:
            del neuron_spikes[neuron]
    return neuron_spikes

def extract_spike(trials_formatted, neuron_spikes, bombcell_keys, baseline_start:float = -0.2):
    '''
    Extracts spikes for each neuron in the trials dataframe. 
    It contains only the spikes that fall within the stimulus window.
    It sets stimulus onset as time 0.
    
    Returns a dictionary of aligned spikes and baseline spikes.
    Dictionary keys are neuron IDs and values are lists of lists of spike times.
    
    Args:
        trials_formatted (pd.DataFrame): The formatted trials data.
        neuron_spikes (dict): Dictionary of neuron spikes.
        bombcell_keys (list): List of bombcell keys.
        baseline_start (float): Start time for the baseline period.
        
    Returns:
        tuple: A tuple containing the baseline spikes and stimulus spikes.
    '''
    stimulus_spikes = defaultdict(list)
    baseline_spikes = defaultdict(list)
    
    # Iterate through trials and extract spikes
    for index, row in trials_formatted.iterrows():
        # Auditory period on is set as the stimulus onset
        stim_onset = row["timeline_audPeriodOn"]
        # Stimulus offset is 0.5 seconds after the onset
        stim_offset = stim_onset + 0.5
        # Baseline onset is `baseline_start` seconds before the stimulus onset
        baseline_onset = stim_onset + baseline_start
        
        # Extract spikes for each neuron -- only the bombcells
        for neuron in bombcell_keys:
            # Choose the neuron
            spike_times = neuron_spikes[neuron]
            
            # Extract spikes within the stimulus windows
            trial_spikes = spike_times[(spike_times >= stim_onset) & (spike_times <= stim_offset)] - stim_onset
            stimulus_spikes[neuron].append(trial_spikes.tolist())
            
            # Extract spikes within the baseline period
            trial_baseline = spike_times[(spike_times >= baseline_onset) & (spike_times < stim_onset)] - stim_onset
            baseline_spikes[neuron].append(trial_baseline.tolist())
        
    stimulus_spikes = dict(stimulus_spikes)
    baseline_spikes = dict(baseline_spikes)
    return baseline_spikes, stimulus_spikes


def calculate_firing_rate(
    x_spikes: dict, 
    bin_size: float, 
    period_start: float = 0, 
    period_end: Optional[float] = None
) -> tuple:
    """
    Calculate firing rates using np.histogram for each neuron and trial.
    
    Args:
        x_spikes (dict): Dictionary of neuron spikes.
        bin_size (float): The size of the bins in seconds.
        period_start (float): The start time of the period.
        period_end (float): The end time of the period.
    
    Returns:
        tuple: (firing_rates, bin_edges) where firing_rates is a dict of neuron IDs and 
               bin_edges is an array of bin start times.
    """
    # Set the period end if not provided
    if period_end is None:
        period_end = period_start + 1
    
    bins = np.arange(period_start, period_end + bin_size, bin_size)
    keys = np.sort(np.array(list(x_spikes.keys())))
    firing_rates = np.empty((len(x_spikes), len(x_spikes[keys[0]]), len(bins) - 1))
    
    # Calculate firing rates for each neuron
    for i, key in enumerate(keys):
        # choose the neuron
        trials = x_spikes[key]
        # Calculate the spike counts for each trial
        spike_counts = np.array([np.histogram(np.array(trial), bins=bins)[0] / bin_size for trial in trials])
        firing_rates[i,:] = spike_counts
    		
    return firing_rates, bins[:-1]


def format_fr(fr_baseline, fr_stimulus, bins_baseline, bins_stimulus, baseline_spikes, stimulus_spikes):
    '''
    Formats the firing rates for each neuron.
    
    Args:
        fr_baseline (np.ndarray): The baseline firing rates.
        fr_stimulus (np.ndarray): The stimulus firing rates.
        bins_baseline (np.ndarray): The baseline bins.
        bins_stimulus (np.ndarray): The stimulus bins.
        baseline_spikes (dict): The baseline spikes.
        stimulus_spikes (dict): The stimulus spikes.
        
    Returns:
        spikes_formatted (dict): A dictionary containing info for each neuron. The keys are the neuron IDs.
        fr_baseline_standardised (np.array): The standardised baseline firing rates. The shape is (n_neurons, n_trials, n_bins).
        fr_stimulus_standardised (np.array): The standardised stimulus firing rates. The shape is (n_neurons, n_trials, n_bins).
    '''

    # Standardise the firing rates
    fr_baseline_standardised, fr_stimulus_standardised = standardise_fr(fr_baseline, fr_stimulus)
    keys = np.sort(np.array(list(baseline_spikes.keys())))

    spikes_formatted = dict()
    
    # Iterate through the neurons and format the spikes
    for i, neuron_id in enumerate(keys):
        # Set the keys of the dictionary to the neuron IDs
        spikes_formatted[neuron_id] = {'spikes': {'baseline': baseline_spikes[neuron_id], 
                                                'stimulus': stimulus_spikes[neuron_id]},
                                    'firing_rate': {'baseline': fr_baseline_standardised[i], # standardised firing rate
                                                    'stimulus': fr_stimulus_standardised[i]},
                                    'firing_rate_raw': {'baseline': fr_baseline[i], # unstandardised firing rate
                                                    'stimulus': fr_stimulus[i]},
                                    'times': {'baseline': bins_baseline,
                                                'stimulus': bins_stimulus}}
    return spikes_formatted, fr_baseline_standardised, fr_stimulus_standardised
    
    
    
def organise_spikes(obj, bin_size: float = 0.01, baseline_start: float = -0.5, stimulus_end: float = 0.5):
    '''
    Organise spikes to be used for later analysis.
    fortmatted_data is a dictionary and the keys are the neuron IDs.
    The values are dictionaries containing the spike times for the baseline and stimulus periods.
    
    spikes_formatted is a dictionary containing the firing rates for each neuron.
    The keys are the neuron IDs and the values are dictionaries containing
    - the standardised baseline firing rates
    - the standardised stimulus firing rates
    - the start times for the first move and choice move
    - neuron IDs
    
    The values are numpy arrays.
    
    Args:
        obj: Object containing all the data.
        bin_size (float): The size of the bins in seconds.
        baseline_start (float): Start time for the baseline period.
        stimulus_end (float): End time for the stimulus period.
    Returns:
        data_formatted: A dictionary containing the spike times for each neuron. Keys are neuron IDs.
        spikes_formatted: A dictionary containing the firing rates, start_time, and neuron IDs.
                        Firing rates are stored in a 3D numpy array shaped (n_neurons, n_trials, n_bins).
    '''
    # Format the spike data
    neuron_spikes = format_spike(obj.all_spike_data, obj.bombcell_keys)
    # Extract the spikes for the baseline and stimulus periods
    baseline_spikes, stimulus_spikes = extract_spike(obj.trials_formatted, neuron_spikes, obj.bombcell_keys, baseline_start= baseline_start)

    # calculate the firing rates
    fr_baseline, bins_baseline = calculate_firing_rate(baseline_spikes, bin_size=bin_size, period_start = baseline_start, period_end =0)
    fr_stimulus, bins_stimulus = calculate_firing_rate(stimulus_spikes, bin_size=bin_size, period_start = 0, period_end = stimulus_end)
    
    obj.baseline_times = bins_baseline
    obj.stimulus_times = bins_stimulus
    obj.times = np.concatenate((bins_baseline, bins_stimulus))
    
    # Format the firing rates
    data_formatted, fr_baseline_standardised, fr_stimulus_standardised = format_fr(fr_baseline, fr_stimulus, 
                                                                                    bins_baseline, bins_stimulus, 
                                                                                    baseline_spikes, stimulus_spikes)
    timeline_first_move = obj.trials_formatted.timeline_firstMoveOn - obj.trials_formatted.timeline_audPeriodOn
    timeline_choice_move = obj.trials_formatted.timeline_choiceMoveOn - obj.trials_formatted.timeline_audPeriodOn
    
    # Format the spikes - dictionary containing the firing rates - 3D numpy array.
    spikes_formatted = {'firing_rate':{'baseline': fr_baseline_standardised,
                                        'stimulus': fr_stimulus_standardised},
                        'start_time': {'first_move': timeline_first_move,
                                        'choice_move': timeline_choice_move},
                        'neuron_ids': np.array(obj.bombcell_keys)}
    
    return data_formatted, spikes_formatted
    