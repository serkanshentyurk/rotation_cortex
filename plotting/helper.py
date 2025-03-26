import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def configure_axes(ax, title='', xlabel='Time (s)', ylabel='Firing rate (Hz)',
                   xlim=None, ylim=None, vlines=None, hlines=None):
    """
    Helper to configure common axis settings.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to configure.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        vlines (list): X positions to draw vertical lines.
        hlines (list): Y positions to draw horizontal lines.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if vlines:
        for v in vlines:
            ax.axvline(v, color='black', linestyle='--')
    if hlines:
        for h in hlines:
            ax.axhline(h, color='black', linestyle='--')

def inspect_neuron_spikes_single(obj_plotter, neuron_idx = 0, plot = False):
    obj = obj_plotter.sd
    neuron_id = obj.bombcell_keys[neuron_idx]
    neuron = obj.formatted_data[neuron_id]
    neuron_spike_counts_baseline = []
    for trial in neuron['spikes']['baseline']:
        neuron_spike_counts_baseline.append(len(trial))
    neuron_spike_counts_baseline= np.array(neuron_spike_counts_baseline).reshape(-1, 1)
    neuron_spike_counts_stim = []
    for trial in neuron['spikes']['stimulus']:
        neuron_spike_counts_stim.append(len(trial))
    neuron_spike_counts_stim = np.array(neuron_spike_counts_stim).reshape(-1, 1)
    neuron_spike_counts = np.concatenate([neuron_spike_counts_baseline, neuron_spike_counts_stim], axis=1)
    if plot:
        plt.figure(figsize = (10,5))
        plt.hist(neuron_spike_counts.sum(axis=1), bins=10, alpha=0.7, label='Total')
        plt.xlim(0,20)
        title_now = '$n_{trial}$ = ' + str(neuron_spike_counts.shape[0]) + '\nCCCP: ' + str(round(obj.cccp_values[neuron_idx],4)) + ' - CCSP_VIS: ' + str(round(obj.ccsp_vis_values[neuron_idx],4)) + ' - CCSP_AUD: ' + str(round(obj.ccsp_aud_values[neuron_idx],4))
        plt.title(title_now, fontsize = 10)
        plt.xlabel('#Spike in a trial')
        plt.ylabel('Count')
        plt.suptitle(f'Spike Count Histogram of a Neuron {int(obj.bombcell_keys[neuron_idx])}', fontsize = 14)
        plt.tight_layout()
        plt.show()
    return neuron_spike_counts

def inspect_neuron_spikes(obj_plotter, neuron_idxs = [], plot = True, title_holder = ''):
    if len(neuron_idxs) == 0:
        raise ValueError('neuron_idxs must be a non-empty list')
    neuron_spike_counts_all = []
    for neuron_idx in neuron_idxs:
        neuron_spike_counts = inspect_neuron_spikes_single(obj_plotter, neuron_idx = neuron_idx)
        neuron_spike_counts_all.append(neuron_spike_counts.sum(axis=1))
    neuron_spike_counts_all = np.array(neuron_spike_counts_all)
    if plot:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=neuron_spike_counts_all.T)
        plt.ylabel('Spike count in a single trial')
        plt.xlabel('Significant Neuron ID')
        plt.xticks(ticks = np.arange(len(neuron_idxs)), labels = neuron_idxs)
        plt.title(title_holder, fontsize = 10)
    plt.suptitle('Spike Count Distribution of Significant Neurons', fontsize = 14)
    plt.tight_layout()
    plt.show()
    return np.array(neuron_spike_counts_all) 
    