from typing import Optional, Literal, List
from collections import defaultdict
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from utils_class import *
from utils import normalize_p_values

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

import matplotlib.colors as mcolors
import seaborn as sns

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("tab10").colors)

from pca_utils import apply_dim_red_to_mean_fr_reduce_fr, apply_dim_red_to_mean_fr_reduce_neuron

class session_data:
    '''
    A class to store all data for a single session. 
    This includes the formatted event data, cluster data, and spike data. 
    '''
    def __init__(self, formatted_events, formatted_cluster_data, spikes, session_no, session_date, animal_id,
                 only_validTrails = True,
                 baseline_start = -0.5, stimulus_end = 0.5, binsize = 0.01, normalise = True, bombcell = True):
        '''
        Initialize the session data object with the provided data. Inputs the output of the format_data function.
        
        Args:
        - formatted_events: The formatted event data for the session.
        - formatted_cluster_data: The formatted cluster data for the session.
        - spikes: The spike data for the session.
        - dominant_modality: The dominant modality for the session.
        - session_no: The session number.
        - session_date: The date of the session.
        '''
        self.all_event_data = formatted_events
        self.all_cluster_data = formatted_cluster_data
        self.all_spike_data = spikes
        self.animal_id = animal_id
        self.session_no = session_no
        self.session_date = session_date
        # self.spikes = defaultdict(list)
        self.only_validTrial = only_validTrails
        
        self.format_trials()
        # self.create_mask()
        self.format_spikes(baseline_start=baseline_start, stimulus_end = stimulus_end, bin_size = binsize, bombcell = bombcell)


        self.conditions = [
            {'aud_stim_loc': 'r', 'vis_stim_loc': 'l'},
            {'aud_stim_loc': 'r', 'vis_stim_loc': 'o'},
            {'aud_stim_loc': 'r', 'vis_stim_loc': 'r'},
            {'aud_stim_loc': 'c', 'vis_stim_loc': 'l'},
            {'aud_stim_loc': 'c', 'vis_stim_loc': 'o'},
            {'aud_stim_loc': 'c', 'vis_stim_loc': 'r'},
            {'aud_stim_loc': 'l', 'vis_stim_loc': 'l'},
            {'aud_stim_loc': 'l', 'vis_stim_loc': 'o'},
            {'aud_stim_loc': 'l', 'vis_stim_loc': 'r'}
        ]
        self.x_y_labels = [
            ['', 'Right'],
            ['', ''],
            ['', ''],
            ['', 'Auditory Stimulus\n\nCenter'],
            ['', ''],
            ['', ''],
            ['Left', 'Left'],
            ['Off\n\nVisual Stimulus', ''],
            ['Right', '']
        ]
    def extract_spikes(self, baseline_start = None):
        '''
        Extract the spikes for each neuron during the baseline and stimulus periods.
        
        Args:
        - pre_stimulus_window: The duration of the pre-stimulus window to consider for the baseline period.
        '''
        if baseline_start is not None:
            self.baseline_start = baseline_start
            
        stimulus_spikes, baseline_spikes = extract_spike(self.trials_formatted, self.spikes, self.baseline_start)
        self.stimulus_on = self.trials_formatted.timeline_audPeriodOn
        self.stimulus_off = self.trials_formatted.timeline_audPeriodOff
        self.stimulus_duration = self.stimulus_off - self.stimulus_on
        
        return stimulus_spikes, baseline_spikes
    
    
    def format_trials(self):
        if self.only_validTrial:
            trials = self.all_event_data[self.all_event_data.is_validTrial]
        else:
            trials = self.all_event_data
            
        trials.reset_index(drop=True, inplace=True)
        # Define conditions and choices for vis_loc
        vis_conditions = [
            trials['visR'] > 0,
            trials['visL'] > 0
        ]
        vis_choices = ['r', 'l']

        # Define conditions and choices for aud_loc
        aud_conditions = [
            trials['audL'] > 0,
            trials['audR'] > 0
        ]
        aud_choices = ['l', 'r']

        # Create the new columns
        trials['vis_loc'] = np.select(vis_conditions, vis_choices, default='o')
        trials['aud_loc'] = np.select(aud_conditions, aud_choices, default='c')
        self.trials_formatted = trials
        
        
        condition_mapping = {
            ('r', 'l'): 0,
            ('r', 'o'): 1,
            ('r', 'r'): 2,
            ('c', 'l'): 3,
            ('c', 'o'): 4,
            ('c', 'r'): 5,
            ('l', 'l'): 6,
            ('l', 'o'): 7,
            ('l', 'r'): 8
        }

        # Add the cond_id column using the mapping
        self.trials_formatted['cond_id'] = self.trials_formatted.apply(
            lambda row: condition_mapping.get((row['aud_loc'], row['vis_loc'])), axis=1)
    
    
    def format_spikes(self, baseline_start = -0.5, stimulus_end = 0.5, bin_size = 0.01,bombcell = True):
        '''
        Format the spike data for the session. This will create a dictionary of spike times for each neuron.
        '''
        self.baseline_start = baseline_start
        self.stimulus_stop = stimulus_end
        
        
        self.spikes = format_spike(self.all_spike_data)
        stimulus_spikes, baseline_spikes = self.extract_spikes(baseline_start)


        stimulus_firing_rates, stimulus_times = calculate_firing_rate(stimulus_spikes, bin_size,
                                                                      period_start = 0, period_end = stimulus_end)
        baseline_firing_rates, baseline_times = calculate_firing_rate(baseline_spikes, bin_size,
                                                                      period_start = baseline_start, period_end = 0)      
        
        
        formatted_spikes = defaultdict(dict)
        if bombcell:
            bombcells = np.where(self.all_cluster_data.bombcell_class == 'good')[0]
            keys = sorted(bombcells)
        else:
            bombcells = stimulus_firing_rates
            keys = sorted(bombcells.keys())
        for neuron_id in stimulus_firing_rates:
            formatted_spikes[neuron_id] = {"spikes": {"baseline": baseline_spikes[neuron_id],
                                                      "stimulus": stimulus_spikes[neuron_id]},
                                           "firing_rate": {"stimulus": stimulus_firing_rates[neuron_id],
                                                           "baseline": baseline_firing_rates[neuron_id]},
                                           "times": {"stimulus": stimulus_times,
                                                     "baseline": baseline_times}}
        self.formatted_data = formatted_spikes
        self.stimulus_times = stimulus_times
        self.baseline_times = baseline_times
        self.times = np.concatenate([baseline_times, stimulus_times])
        
        baseline_arrays = [baseline_firing_rates[key] for key in keys]  # Extract arrays in the order of keys
        stimulus_arrays = [stimulus_firing_rates[key] for key in keys]  # Extract arrays in the order of keys

        baseline_3d = np.stack(baseline_arrays, axis=0)
        stimulus_3d = np.stack(stimulus_arrays, axis=0)
        
        timeline_first_move = self.trials_formatted.timeline_firstMoveOn - self.trials_formatted.timeline_audPeriodOn
        timeline_choice_move = self.trials_formatted.timeline_choiceMoveOn - self.trials_formatted.timeline_audPeriodOn
        
        ## Standardisation
        # Calculate mean and st
        baseline_means = baseline_3d.reshape(186,-1).mean(axis=1)
        baseline_stds = baseline_3d.reshape(186,-1).std(axis=1)
        zero_var_neurons = baseline_stds == 0
        baseline_stds[zero_var_neurons] = 1  
        # Standardize
        normalized_baseline = (baseline_3d - baseline_means[:, None, None]) / baseline_stds[:, None, None]
        normalized_baseline[zero_var_neurons, :, :] = 0
        normalized_stimulus = (stimulus_3d - baseline_means[:, None, None]) / baseline_stds[:, None, None]


        self.spikes_formatted = {'firing_rate': {'baseline': normalized_baseline, 
                                                 'stimulus': normalized_stimulus},
                                 'firing_rate_raw': {'baseline': baseline_3d,
                                                     'stimulus': stimulus_3d},
                                 'neuron_ids': np.array(keys),
                                 'start_time': {'first_move': timeline_first_move.values, 
                                                    'choice_move': timeline_choice_move.values}}
        
    def select_data_interest(self, neuron_indices = None, **kwargs):
        filtered_trials, original_trial_indices = select_trials(self, **kwargs)
        
        fr_baseline = self.spikes_formatted['firing_rate']['baseline'][:,original_trial_indices]
        fr_stimulus = self.spikes_formatted['firing_rate']['stimulus'][:,original_trial_indices]
        if neuron_indices is not None:
            fr_baseline = fr_baseline[neuron_indices]
            fr_stimulus = fr_stimulus[neuron_indices]
        fr_all = np.concatenate([fr_baseline, fr_stimulus], axis=2)
        start_time_first_move = self.spikes_formatted['start_time']['first_move'][original_trial_indices]
        start_time_choice_move = self.spikes_formatted['start_time']['choice_move'][original_trial_indices]
        
        interest_data = {'firing_rate': fr_all, 
                        'firing_rate_baseline': fr_baseline,
                        'firing_rate_stimulus': fr_stimulus,
                        'start_time': {'first_move': start_time_first_move, 'choice_move': start_time_choice_move},
                        'neuron_ids': self.spikes_formatted['neuron_ids'],
                        'trial_indices': original_trial_indices,
                        'trials': filtered_trials}
        
        self.interest_data = interest_data
        
        
    def plot_PSTH(self, figsize = (10, 6), xlim = None, ylim = None,
                  filter_info= None, 
                  color_based_on = 'vis',
                  only_average = True):

        if filter_info is None:
            n_filter = 1
            filter_info = [None]
        else:
            n_filter = len(filter_info)
            
        n_freq = self.interest_data['firing_rate'].shape[2]
        n_neuron = self.interest_data['firing_rate'].shape[0]

        fig, axs = plt.subplots(n_filter, 1, figsize=figsize)
        for i in range(n_filter):
            if n_filter > 1:
                ax = axs[i]
            else:
                ax = axs
                
            if filter_info[i] is not None:
                self.select_data_interest(filter_info[i])
                plot_title = filter_info
            else:
                self.select_data_interest()
                plot_title = 'All Trials'
                
            ax.axvline(0, color='black', linestyle='--')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing rate (Hz)')
            if color_based_on is not None:
                if color_based_on in ['visual', 'vis', 'v']:
                    l_index = self.interest_data['trials'].vis_loc == 'l'
                    r_index = self.interest_data['trials'].vis_loc == 'r'
                    c_index = self.interest_data['trials'].vis_loc == 'o'
                    legend_title = 'Vis Stim'
                    plot_title_2 = 'Visual Stimulus'
                elif color_based_on in ['audio', 'aud', 'a']:
                    l_index = self.interest_data['trials'].aud_loc == 'l'
                    r_index = self.interest_data['trials'].aud_loc == 'r'
                    c_index = self.interest_data['trials'].aud_loc == 'c'
                    legend_title = 'Aud Stim'
                    plot_title_2 = 'Audio Stimulus'
                elif color_based_on in ['choice', 'ch', 'c']:
                    l_index = self.interest_data['trials'].choice == -1
                    r_index = self.interest_data['trials'].choice == 1
                    c_index = self.interest_data['trials'].choice == 0
                    legend_title = 'Choice'
                    plot_title_2 = 'Choice'
                
                #individual trials
                if not only_average:
                    ax.plot(self.times, self.interest_data['firing_rate'][:, l_index].mean(axis=1).reshape(n_freq,-1), c = 'blue',label = 'Left', alpha = 0.5)
                    ax.plot(self.times, self.interest_data['firing_rate'][:, r_index].mean(axis=1).reshape(n_freq,-1), c = 'red',label = 'Right', alpha = 0.5)        
                    ax.plot(self.times, self.interest_data['firing_rate'][:, c_index].mean(axis=1).reshape(n_freq,-1), c = 'green',label = 'Center', alpha = 0.5)
                
                # means
                ax.plot(self.times, self.interest_data['firing_rate'][:, l_index].mean(axis=1).mean(axis=0), c = 'blue',label = 'Right', alpha = 1)
                ax.plot(self.times, self.interest_data['firing_rate'][:, r_index].mean(axis=1).mean(axis=0), c = 'red',label = 'Right', alpha = 1)
                ax.plot(self.times, self.interest_data['firing_rate'][:, c_index].mean(axis=1).mean(axis=0), c = 'green',label = 'Center', alpha = 1)
                
                legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Left'),
                                Line2D([0], [0], color='red', lw=2, label='Right'),
                                Line2D([0], [0], color='green', lw=2, label='Center')]
                ax.legend(handles=legend_elements, title=legend_title, title_fontsize = 12, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_title(f'Firing Rates Before and After Stimulus {plot_title} - Colored by {plot_title_2}')
            else:
                # individual trials
                if not only_average:
                    ax.plot(self.times, self.interest_data['firing_rate'].mean(axis=1).reshape(n_freq,-1), c = 'gray',label = 'All neurons', alpha = 0.5) 
                # means
                ax.plot(self.times, self.interest_data['firing_rate'].mean(axis=1).mean(axis=0), c = 'black',label = 'All neurons', alpha = 1)
                ax.set_title(f'Firing Rates Before and After Stimulus - {plot_title}')
        plt.suptitle(f"PSTH - Session Date and Number: {self.session_date} & {self.session_no}")
        plt.tight_layout()
        plt.show()
        
        
    
    def plot_PSTH_grid(self, only_average = True, xlim = (-0.2,0.5), ylim = None):    
        n_freq = self.interest_data['firing_rate'].shape[2]

        fig, axes = plt.subplots(3, 3, figsize=(15,15))

        for i in range(3):
            for j in range(3):
                mask = self.conditions[i*3+j]
                ax = axes[i, j]
                self.select_data_interest(**mask) # visual_trial = True
                l_index = self.interest_data['trials'].choice == 0
                r_index = self.interest_data['trials'].choice == 1
                c_index = self.interest_data['trials'].choice == -1
                
                #individual trials
                if not only_average:
                    ax.plot(self.times, self.interest_data['firing_rate'][:, l_index].mean(axis=1).reshape(n_freq,-1), c = 'blue',label = 'Left', alpha = 0.5)
                    ax.plot(self.times, self.interest_data['firing_rate'][:, r_index].mean(axis=1).reshape(n_freq,-1), c = 'red',label = 'Right', alpha = 0.5)        
                    ax.plot(self.times, self.interest_data['firing_rate'][:, c_index].mean(axis=1).reshape(n_freq,-1), c = 'green',label = 'Center', alpha = 0.5)
                        
                # means
                ax.plot(self.times, self.interest_data['firing_rate'][:, l_index].mean(axis=1).mean(axis=0), c = 'blue',label = 'Left', alpha = 1)
                ax.plot(self.times, self.interest_data['firing_rate'][:, r_index].mean(axis=1).mean(axis=0), c = 'red',label = 'Right', alpha = 1)
                ax.plot(self.times, self.interest_data['firing_rate'][:, c_index].mean(axis=1).mean(axis=0), c = 'green',label = 'Center', alpha = 1)
                ax.set_ylim(ylim)    
                ax.set_xlim(xlim)
                ax.axvline(0, c = 'black', ls = '--')
                ax.set_xlabel(self.x_y_labels[i*3+j][0])
                ax.set_ylabel(self.x_y_labels[i*3+j][1])

        axes[0,1].set_title('Blue --> Left Choice, Green --> No Choice, Red --> Right Choice')
        plt.suptitle(f'PSTH - {self.animal_id} - {self.session_date} - {self.session_no}', fontsize = 15)
        plt.tight_layout()
        plt.show()

    def plot_raster(self, neuron_ids=[0], first_move=True, sort_choice_loc=True, sort_response_time=True, xlim=(-0.2, 0.5), figsize=(10, 6), ax = None, x_y_labels = None):
        if first_move:
            timeline = self.interest_data['start_time']['first_move']
        else:
            timeline = self.interest_data['start_time']['choice_move']   
            
        if neuron_ids is None:
            neuron_ids = self.interest_data['neuron_ids']
        if type(neuron_ids) == int:
            neuron_ids = [neuron_ids]
            
        
        if not sort_choice_loc and not sort_response_time:
            sorted_index = np.arange(self.interest_data['neuron_ids'].shape[0])
        else:
            df = pd.DataFrame({'response_direction': self.interest_data['trials']['response_direction'],
                            'timeline': timeline})
            if sort_choice_loc and sort_response_time:
                df_sorted = df.sort_values(by=['response_direction', 'timeline'])
                sorted_index = df_sorted.index
            elif sort_choice_loc:
                df_sorted = df.sort_values(by=['response_direction'])
                sorted_index = df_sorted.index
            elif sort_response_time:
                df_sorted = df.sort_values(by=['timeline'])
                sorted_index = df_sorted.index
                
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim)
        
        if x_y_labels is None:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial number')
            ax.set_title('Raster plot')
        else:
            ax.set_xlabel(x_y_labels[0])
            ax.set_ylabel(x_y_labels[1])
        ax.axvline(0, color='black', linestyle='--')
        
        colors = ['gray', 'blue', 'red']
        for neuron_id in neuron_ids:
            baseline_spikes = self.formatted_data[neuron_id]['spikes']['baseline']
            stimulus_spikes = self.formatted_data[neuron_id]['spikes']['stimulus']
            
            for trial_num in range(len(sorted_index)):
                x_baseline = baseline_spikes[sorted_index[trial_num]]
                x_stimulus = stimulus_spikes[sorted_index[trial_num]]
                ax.plot(x_baseline, np.repeat(trial_num, len(x_baseline)),'o', ms =2, 
                                color=colors[int(df_sorted.response_direction.to_numpy()[trial_num])])
                ax.plot(x_stimulus, np.repeat(trial_num, len(x_stimulus)),'o', ms =2, 
                                color=colors[int(df_sorted.response_direction.to_numpy()[trial_num])])
                ax.plot(df_sorted.timeline.to_numpy()[trial_num], trial_num, 'o', ms = 2, color='black')

    def plot_raster_grid(self, neuron_ids = [], first_move = True, sort_choice_loc=True, sort_response_time=True,xlim=(-0.2, 0.5), figsize=(12, 12)):  

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        for i in range(3):
            for j in range(3):
                mask = self.conditions[i*3+j]
                self.select_data_interest(**mask) # visual_trial = True
                
                self.plot_raster(neuron_ids=neuron_ids, first_move=first_move, sort_choice_loc=sort_choice_loc, sort_response_time=sort_response_time, 
                        xlim=xlim, figsize=figsize, ax = axes[i,j], x_y_labels = self.x_y_labels[i*3+j])
        
        plt.suptitle(f'Raster Plot - Neuron: {neuron_ids} - Animal: {self.animal_id} - Date: {self.session_date}')
        plt.tight_layout()
        plt.show()
        
    def find_significant_neurons(self,method = 'ttest',
                             parametric_test = True, one_sided = True, up_regulated = True,
                             multiple_correction = True, mc_alpha = 0.05, mc_method = 'fdr_bh'):
        if method == 'ttest':
            baseline_rate = self.interest_data['firing_rate_baseline'].mean(axis=2).T
            stimulus_rate = self.interest_data['firing_rate_stimulus'].mean(axis=2).T

            stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc = test_fr_change_ttest(baseline_rate,stimulus_rate, 
                                                                                            parametric_test, one_sided, up_regulated, 
                                                                                            multiple_correction, mc_alpha, mc_method)
            return stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc
        else:
            print('Method not implemented')
    
    def significant_neurons(self, method = 'ttest', parametric_test = True,  multiple_correction = True, mc_alpha = 0.05, mc_method = 'bonferroni'):
        all_results = []
        stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc = self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = False, up_regulated= True, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        all_results.append([stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc])
        stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc = self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = True, up_regulated= True, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        all_results.append([stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc])
        stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc = self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = True, up_regulated= False, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        all_results.append([stats, p_vals, rejected, corrected_p_vals, n_significant, n_significant_mc])
        self.significant_neurons_results = all_results
        
        results_up = self.find_significant_neurons_grid(method = method, parametric_test = parametric_test, one_sided = True, up_regulated = True, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        results_down = self.find_significant_neurons_grid(method = method, parametric_test = parametric_test, one_sided = True, up_regulated = False, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        self.significant_neurons_results_grid = [results_up, results_down]
        
        
    def find_significant_neurons_grid(self, method = 'ttest', masks = None,
                             parametric_test = True, one_sided = True, up_regulated = True,
                             multiple_correction = True, mc_alpha = 0.05, mc_method = 'fdr_bh'):
        if masks is None:
            masks = self.conditions
        results = []
        for mask in masks:
            self.select_data_interest(**mask)
            result_current = []
            result_current.append(self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                             multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method))
            
            mask_choice_left = mask.copy()
            mask_choice_left['choice'] = 'l'
            self.select_data_interest(**mask_choice_left)
            result_current.append(self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                             multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method))
            
            mask_choice_right = mask.copy()
            mask_choice_right['choice'] = 'r'
            self.select_data_interest(**mask_choice_right)
            result_current.append(self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                             multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method))
            
            mask_choice_no = mask.copy()
            mask_choice_no['choice'] = 'n'
            self.select_data_interest(**mask_choice_no)
            result_current.append(self.find_significant_neurons(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                             multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method))
            results.append(result_current)
            
        return results
    

    def visualise_significant_fr_change(self,               
                                        method='ttest',
                                        parametric_test=True,
                                        one_sided=True,
                                        multiple_correction=True,
                                        mc_alpha=0.05,
                                        mc_method='bonferroni'):
        
        choice_labels = ['All', 'Left', 'Right','No']
        choice_filters = [None, {'choice': 'l'}, {'choice': 'r'},{'choice': 'n'}]
        percentages = {}
        trial_counts = {}

        # Total number of neurons (assumed constant across conditions)
        total_neurons = len(self.spikes_formatted['neuron_ids'])

        # Loop over each condition.
        for cond_idx, cond in enumerate(self.conditions):
            percentages[cond_idx] = {}
            trial_counts[cond_idx] = {}
            
            # Loop over each choice group in the new order.
            for i, choice_filter in enumerate(choice_filters):
                # Build the filter for the current condition and choice.
                filters = cond.copy()
                if choice_filter is not None:
                    filters.update(choice_filter)
                
                # Select trials for this combination.
                self.select_data_interest(**filters)
                n_trials = len(self.interest_data['trials'])
                trial_counts[cond_idx][choice_labels[i]] = n_trials
                
                # Run significance tests.
                # (Assumes find_significant_neurons returns: _, _, _, _, n_sig, _)
                # For increased firing rates:
                _, _, _, _, _, n_sig_inc = self.find_significant_neurons(
                    method=method,
                    parametric_test=parametric_test,
                    one_sided=one_sided,
                    up_regulated=True,
                    multiple_correction=multiple_correction,
                    mc_alpha=mc_alpha,
                    mc_method=mc_method
                )
                
                if one_sided:
                    # For decreased firing rates:
                    _, _, _, _, _, n_sig_dec = self.find_significant_neurons(
                        method=method,
                        parametric_test=parametric_test,
                        one_sided=one_sided,
                        up_regulated=False,
                        multiple_correction=multiple_correction,
                        mc_alpha=mc_alpha,
                        mc_method=mc_method
                        )
                else:
                    n_sig_dec = np.zeros(n_sig_inc.shape)
                
                # Calculate percentages.
                perc_inc =  n_sig_inc / total_neurons
                perc_dec = n_sig_dec / total_neurons
                perc_no = 1 - (perc_inc + perc_dec)
                
                percentages[cond_idx][choice_labels[i]] = (perc_inc, perc_dec, perc_no)

        # ---------------------------
        # 3. Plot the 3x3 Grid with Custom Axis Labels and Trial Counts
        # ---------------------------
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True)

        # Loop over the 9 conditions.
        for cond_idx, ax in enumerate(axes.flatten()):
            # Set custom x and y axis labels from the provided x_y_labels.
            ax.set_xlabel(self.x_y_labels[cond_idx][0])
            ax.set_ylabel(self.x_y_labels[cond_idx][1])
            
            # x positions for the 4 bars (one per choice group).
            x = np.arange(len(choice_labels))
            
            # Extract the percentage values for each choice group.
            inc_vals = [percentages[cond_idx][lbl][0] for lbl in choice_labels]
            dec_vals = [percentages[cond_idx][lbl][1] for lbl in choice_labels]
            no_vals  = [percentages[cond_idx][lbl][2] for lbl in choice_labels]
            
            # Plot the stacked bar chart.
            bar_inc = ax.bar(x, inc_vals, label='Increased', color='red')
            bar_dec = ax.bar(x, dec_vals, bottom=inc_vals, label='Decreased', color='blue')
            bottom_stack = np.array(inc_vals) + np.array(dec_vals)
            bar_no  = ax.bar(x, no_vals, bottom=bottom_stack, label='No Change', color='gray')
            
            ax.set_xticks(x)
            ax.set_xticklabels(choice_labels, rotation=45)
            
            # Annotate each bar with the number of trials.
            for j, lbl in enumerate(choice_labels):
                ntrials = trial_counts[cond_idx][lbl]
                # Total height of the stacked bar.
                total_height = inc_vals[j] + dec_vals[j] + no_vals[j]
                # Place the text just above the bar.
                ax.text(x[j], total_height , f"n={ntrials}", ha='center', va='bottom', fontsize=10)

        # Add a global legend.
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        fig.suptitle("Frequency of Neurons with Significant Firing Rate Changes\n(Trial Counts Per Condition & Choice)", fontsize=16)

        plt.tight_layout()
        plt.show()

    def plot_neuron_significant_fr_change(self, include_choice = True, grid = True, normalise_column = 0,
                                        method='ttest', parametric_test=True, one_sided=True, multiple_correction=True, mc_alpha=0.05, mc_method='bonferroni'):
        results_up = self.find_significant_neurons_grid(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = True, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        results_down = self.find_significant_neurons_grid(method = method, parametric_test = parametric_test, one_sided = one_sided, up_regulated = False, multiple_correction = multiple_correction, mc_alpha = mc_alpha, mc_method = mc_method)
        
        if include_choice:
            p_vals_up = np.ones((len(results_up[0][0][2]), len(results_up)*len(results_up[0])))
            p_vals_down = np.ones((len(results_down[0][0][2]), len(results_down)*len(results_down[0])))
            x_labels = []

            for cond in range(len(results_up)):
                p_vals_up[:, 4*cond][results_up[cond][0][2]] = results_up[cond][0][3][results_up[cond][0][2]]
                p_vals_up[:, 4*cond+1][results_up[cond][1][2]] = results_up[cond][1][3][results_up[cond][1][2]]
                p_vals_up[:, 4*cond+2][results_up[cond][2][2]] = results_up[cond][2][3][results_up[cond][2][2]]
                p_vals_up[:, 4*cond+3][results_up[cond][3][2]] = results_up[cond][3][3][results_up[cond][3][2]]
                
                p_vals_down[:, 4*cond][results_down[cond][0][2]] = results_down[cond][0][3][results_down[cond][0][2]]
                p_vals_down[:, 4*cond+1][results_down[cond][1][2]] = results_down[cond][1][3][results_down[cond][1][2]]
                p_vals_down[:, 4*cond+2][results_down[cond][2][2]] = results_down[cond][2][3][results_down[cond][2][2]]
                p_vals_down[:, 4*cond+3][results_down[cond][3][2]] = results_down[cond][3][3][results_down[cond][3][2]]
                
                # x labels
                lab = f"{self.conditions[cond]}"
                lab_upd = lab[2:5] + ':' + lab[18:19] + ' - ' + lab[23:26] + ':' + lab[39:40]
                lab_upd_0 = lab_upd + ' - choice: all'
                lab_upd_1 = lab_upd + ' - choice: left'
                lab_upd_2 = lab_upd + ' - choice: right'
                lab_upd_3 = lab_upd + ' - choice: none'
                x_labels.append(lab_upd_0)
                x_labels.append(lab_upd_1)
                x_labels.append(lab_upd_2)
                x_labels.append(lab_upd_3)

        else:
            p_vals_up = np.ones((len(results_up[0][0][2]), len(results_up)))
            p_vals_down = np.ones((len(results_down[0][0][2]), len(results_down)))
            x_labels = []
            for cond in range(len(results_up)):
                p_vals_up[:, cond][results_up[cond][0][2]] = results_up[cond][0][3][results_up[cond][0][2]]
                p_vals_down[:, cond][results_down[cond][0][2]] = results_down[cond][0][3][results_down[cond][0][2]]
                lab = f"{self.conditions[cond]}"
                lab_upd = lab[2:5] + ':' + lab[18:19] + ' - ' + lab[23:26] + ':' + lab[39:40]
                x_labels.append(lab_upd)
    
        # Normalize p-values (set NaNs to 0 so they don't affect neutral neurons)
        norm_p_up = np.where(np.isnan(p_vals_up), 0, normalize_p_values(p_vals_up))
        norm_p_down = np.where(np.isnan(p_vals_down), 0, normalize_p_values(p_vals_down))
        p_vals = norm_p_up - norm_p_down
    
        if grid:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                current_p_vals = p_vals[:,4*i:4*i+4]
                sorted_current_p_vals = current_p_vals[current_p_vals[:,0].argsort()]
                sns.heatmap(sorted_current_p_vals, cmap="coolwarm", center=0, cbar=True,
                            xticklabels=[f"{x_labels[i*4+j]}" for j in range(4)],
                            yticklabels='',
                            cbar_kws={"label": "Log-Normalized p-value"},
                            ax=ax)
                ax.set_xlabel(f"{self.x_y_labels[i][0]}")
                ax.set_ylabel(f"{self.x_y_labels[i][1]}")

            plt.suptitle("Ordered Heatmap of Neuronal Significance (Intensity = P-Value)")
            plt.tight_layout()
            plt.show()
    
        else:
            sorted_p_vals = p_vals[p_vals[:,normalise_column].argsort()]
            plt.figure(figsize=(12, 8))
            sns.heatmap(sorted_p_vals, cmap="coolwarm", center=0, cbar=True,
                    xticklabels=[f"{x_labels[i]}" for i in range(sorted_p_vals.shape[1])],
                    # yticklabels=[f"N{sorted_neuron_order[i]}" for i in range(len(sorted_neuron_order))]
                    yticklabels='',
                    cbar_kws={"label": "Log-Normalized p-value"})

            if len(x_labels) == 36:
                plt.axvline(x=4, color='k', linewidth=1)
                plt.axvline(x=8, color='k', linewidth=1)
                plt.axvline(x=12, color='k', linewidth=1)
                plt.axvline(x=16, color='k', linewidth=1)
                plt.axvline(x=20, color='k', linewidth=1)
                plt.axvline(x=24, color='k', linewidth=1)
                plt.axvline(x=28, color='k', linewidth=1)
                plt.axvline(x=32, color='k', linewidth=1)
                plt.axvline(x=12, color='r', linewidth=1, linestyle='-.')
                plt.axvline(x=24, color='r', linewidth=1, linestyle='-.')
            else:
                plt.axvline(x=3, color='k', linewidth=1)
                plt.axvline(x=6, color='k', linewidth=1)
                plt.xticks(rotation=45)
                plt.xlabel("Conditions")
                plt.ylabel("Neurons (Ordered by First Column Significance)")
                plt.title("Ordered Heatmap of Neuronal Significance (Intensity = P-Value)")

            plt.show()

    def apply_dim_red_to_mean_fr(self, reduce_freq, plot_projection = True, plot_var = True, var_log = False, pc_x = 0, pc_y=1, return_obj = False,
                                 pca = True, umap = False, line = True, random_state =42):
        if reduce_freq:
            dim_red_obj = apply_dim_red_to_mean_fr_reduce_fr(self, plot_projection=plot_projection, plot_var = plot_var, var_log = var_log, pc_x = pc_x, pc_y = pc_y,
                                                             pca = pca, umap = umap, random_state = random_state, line = line)
        else:
            dim_red_obj = apply_dim_red_to_mean_fr_reduce_neuron(self, plot_projection=plot_projection, plot_var = plot_var, var_log = var_log, pc_x = pc_x, pc_y = pc_y,
                                                             pca = pca, umap = umap, random_state = random_state, line = line)
        if return_obj:
            return dim_red_obj
        
    def apply_dim_red_to_mean_fr_reduce_fr_grid(self, pc_x = 0, pc_y = 1, significant_neuron = None,
                                            pca = True, umap = False, random_state = 42, xlim = None, ylim = None):
        increase_fr = self.significant_neurons_results_grid[0]
        decrease_fr = self.significant_neurons_results_grid[1]

        
        dim_red_obj = self.apply_dim_red_to_mean_fr(reduce_freq = True, plot_projection = False, plot_var = False, return_obj = True,
                                                pca = pca, umap = umap, random_state = random_state)
        
        custom_order = []
        for row_group in range(0, 6, 2):  # Iterate over row pairs (0-1, 2-3, 4-5)
            for col in range(6):          # Iterate over columns
                custom_order.append((row_group, col))
                custom_order.append((row_group + 1, col))
        choice = [None, 'left', 'right', 'n']
        fig, axs = plt.subplots(6, 6, figsize=(12, 12), sharex=True, sharey=True)

        for index, (r, c) in enumerate(custom_order):
            ax = axs[r, c]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            trial = index//4

            choice_no = index%4

            mask = self.conditions[trial]
            mask_choice= mask.copy()
            mask_choice['choice'] = choice[choice_no]  
            norm = plt.Normalize(vmin=-1, vmax=1)  

            self.select_data_interest(neuron_indices = significant_neuron, **mask_choice)
            n_trial = self.interest_data['firing_rate'].shape[1]
            if choice_no == 0:
                ax.set_title(f'Choice: All, n_trial = {n_trial}', fontsize=8)
            elif choice_no == 3:
                ax.set_title(f'Choice: None, n_trial = {n_trial}', fontsize=8)
            else:
                ax.set_title(f'Choice: {choice[choice_no]}, n_trial = {n_trial}', fontsize=8)

            if self.interest_data['firing_rate'].shape[1] == 0:
                pass
            else:
                colors = np.zeros(self.interest_data['firing_rate'].shape[0])
                increase_indices = increase_fr[trial][choice_no][2]
                decrease_indices = decrease_fr[trial][choice_no][2]
                if significant_neuron is not None:
                    increase_indices = np.array(increase_indices)[significant_neuron]
                    decrease_indices = np.array(decrease_indices)[significant_neuron]
                colors[increase_indices] = 1
                colors[decrease_indices] = -1
                projected_data = dim_red_obj.transform(self.interest_data['firing_rate'].mean(axis=1))
                ax.scatter(projected_data[:, pc_x], projected_data[:, pc_y], c=colors, alpha=0.5, cmap='coolwarm', norm = norm)
                ax.set_xlabel(f'n_inc = {np.sum(increase_indices)}, n_dec = {np.sum(decrease_indices)}', fontsize=8)

        # Add separation lines
        line_color = "black"
        line_width = 2

        # Add vertical lines between columns 2 & 4
        for x_pos in [2.15, 4.05]:
            fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color=line_color, linewidth=line_width))

        # Add horizontal lines between rows 1 & 3
        for y_pos in [2/6, 3.9/6]:
            fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color=line_color, linewidth=line_width))

        plt.suptitle('PC space for trial types and choices', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def apply_dim_red_to_mean_fr_reduce_neuron_grid(self, pc_x = 0, pc_y = 1, significant_neuron = None,
                                                    pca = True, umap = False, random_state = 42, xlim = None, ylim = None):
        dim_red_obj = self.apply_dim_red_to_mean_fr(reduce_freq = False, plot_projection = False, plot_var = False, return_obj = True,
                                                pca = pca, umap = umap, random_state = random_state)
        custom_order = []
        for row_group in range(0, 6, 2):  # Iterate over row pairs (0-1, 2-3, 4-5)
            for col in range(6):          # Iterate over columns
                custom_order.append((row_group, col))
                custom_order.append((row_group + 1, col))
                
        choice = [None, 'left', 'right', 'n']

        fig, axs = plt.subplots(6, 6, figsize=(12, 12), sharex=True, sharey=True)
        for index, (r, c) in enumerate(custom_order):
            ax = axs[r, c]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            trial = index//4
            
            choice_no = index%4

            mask = self.conditions[trial]
            mask_choice= mask.copy()
            mask_choice['choice'] = choice[choice_no]  
            norm = plt.Normalize(vmin=-0.5, vmax=0.5)  

            self.select_data_interest(neuron_indices= significant_neuron, **mask_choice)
            n_trial = self.interest_data['firing_rate'].shape[1]
            if choice_no == 0:
                ax.set_title(f'Choice: All, n_trial = {n_trial}', fontsize=8)
            elif choice_no == 3:
                ax.set_title(f'Choice: None, n_trial = {n_trial}', fontsize=8)
            else:
                ax.set_title(f'Choice: {choice[choice_no]}, n_trial = {n_trial}', fontsize=8)
            
            if self.interest_data['firing_rate'].shape[1] == 0:
                pass
            else:
                colors = self.times
                n_trial = self.interest_data['firing_rate'].shape[1]
                increase_indices = self.significant_neurons_results_grid[0][trial][choice_no][2]
                decrease_indices = self.significant_neurons_results_grid[1][trial][choice_no][2]
                if significant_neuron is not None:
                    increase_indices = np.array(increase_indices)[significant_neuron]
                    decrease_indices = np.array(decrease_indices)[significant_neuron]
                    
                projected_data = dim_red_obj.transform(self.interest_data['firing_rate'].mean(axis=1).T)
                ax.scatter(projected_data[:, pc_x], projected_data[:, pc_y], c=colors, alpha=0.5, cmap='coolwarm', norm = norm)
                ax.set_xlabel(f'n_inc = {np.sum(increase_indices)}, n_dec = {np.sum(decrease_indices)}', fontsize=8)

        # Add separation lines
        line_color = "black"
        line_width = 2

        # Add vertical lines between columns 2 & 4
        for x_pos in [2.15, 4.05]:
            fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color=line_color, linewidth=line_width))

        # Add horizontal lines between rows 1 & 3
        for y_pos in [2/6, 3.9/6]:
            fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color=line_color, linewidth=line_width))

        plt.suptitle('PC space for trial types and choices', fontsize=16)
        plt.tight_layout();plt.show()
        
    def calculate_cccp_ccsp(self, plot = True, sort_by:Optional[Literal['cccp', 'ccsp-vid', 'ccsp-aud']] = 'cccp'):
        self.cccp_values, self.ccsp_aud_values, self.ccsp_vis_values = calculate_cccp_ccsp(self)
        if plot:
            self.plot_cccp_ccsp(sort_by = sort_by)
    
    def plot_cccp_ccsp(self, sort_by:Optional[Literal['cccp','ccsp-vid', 'ccsp-aud']] = 'cccp', figsize = (12, 8)):
        if sort_by == 'cccp':
            sort_indices = self.cccp_values.argsort()
        elif sort_by == 'ccsp-vis':
            sort_indices = self.ccsp_vis_values.argsort()
        elif sort_by == 'ccsp-aud':
            sort_indices = self.ccsp_aud_values.argsort()
        elif sort_by == None:
            sort_indices = np.arange(len(self.cccp_values))
        else:
            raise ValueError('sort_by should be either cccp or ccsp-vis or ccsp-aud')
        
        fig, axs = plt.subplots(2,1,figsize=figsize)
        ax = axs[0]
        ax.axhline(0.025, color = 'r', linestyle = '--')
        ax.axhline(0.975, color = 'r', linestyle = '--')
        ax.semilogy(self.cccp_values[sort_indices], 'o', ms = 4, label = f'Choice\nn={np.sum(self.cccp_values < 0.025)}')
        ax.semilogy(self.ccsp_aud_values[sort_indices], 'o', ms = 4, label = f'Auditory stim\nn={np.sum(self.ccsp_aud_values < 0.025)}')
        ax.semilogy(self.ccsp_vis_values[sort_indices], 'o', ms = 4, label = f'Visual stim\nn={np.sum(self.ccsp_vis_values < 0.025)}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('p-value')
        ax.set_title(f'CCCP and CCSP, n_significant = {np.sum(np.min([self.cccp_values, self.ccsp_aud_values, self.ccsp_vis_values], axis=0) < 0.025)}')

        ax = axs[1]
        ax.axhline(0.025, color = 'r', linestyle = '--')
        ax.axhline(0.975, color = 'r', linestyle = '--')
        ax.semilogy(1-self.cccp_values[sort_indices], 'o', ms = 4, label = f'Choice\nn={np.sum(1-self.cccp_values < 0.025)}')
        ax.semilogy(1-self.ccsp_aud_values[sort_indices], 'o', ms = 4, label = f'Auditory stim\nn={np.sum(1-self.ccsp_aud_values < 0.025)}')
        ax.semilogy(1-self.ccsp_vis_values[sort_indices], 'o', ms = 4, label = f'Visual stim\nn={np.sum(1-self.ccsp_vis_values < 0.025)}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('p-value')
        ax.set_title(f'1-CCCP and 1-CCSP, n_significant = {np.sum(np.max([self.cccp_values, self.ccsp_aud_values, self.ccsp_vis_values], axis=0) > 0.975)}')
        plt.tight_layout()
        plt.show()


    def plot_heatmap_trials(self, cmap = 'magma',center = None, vmin = -1, vmax = 2, median = False, sort_based_on = 'cccp', figsize = (12, 10), vline_col = 'green', hline_col = 'red',
                            only_significant = False, bar_n_sample = 5):
        cmap = sns.color_palette(cmap, as_cmap=True)
        if sort_based_on == 'cccp':
            sort_based_on_data = self.cccp_values
        elif sort_based_on == 'ccsp-vis':
            sort_based_on_data = self.ccsp_vis_values
        elif sort_based_on == 'ccsp-aud':	
            sort_based_on_data = self.ccsp_aud_values
        else:	
            raise ValueError('sort_based_on should be either cccp or ccsp-vis or ccsp-aud')

        sort_indices = sort_based_on_data.argsort()
        down_limit = np.where(sort_based_on_data[sort_indices] > 0.025)[0][0] 
        up_limit = np.where(sort_based_on_data[sort_indices] > 0.975)[0][0] 
        if only_significant:
            sort_indices_significant = np.array(np.concatenate([np.arange(down_limit),
                                                                np.arange(up_limit, 
                                                                          len(sort_indices))
                                                                ]),
                                                dtype = int)

        self.select_data_interest()
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                current_mask = self.conditions[i*3+j]
                self.select_data_interest(**current_mask)
                n_trial = self.interest_data['firing_rate'].shape[1]
                if n_trial == 0:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    continue
                data_for_heatmap = self.interest_data['firing_rate'].mean(axis=1)
                data_for_heatmap = data_for_heatmap[sort_indices]
                if only_significant:
                    data_for_heatmap = data_for_heatmap[sort_indices_significant]
                else:
                    ax.axhline(up_limit, color = hline_col, linewidth = 1, linestyle = '--')
                if median:
                    move_time = np.nanmedian(self.interest_data['start_time']['first_move'])
                else:
                    move_time = np.nanmean(self.interest_data['start_time']['first_move'])
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
                sns.heatmap(data_for_heatmap, ax=ax, xticklabels=False, yticklabels=False, cbar=False, center=center, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
                ax.axvline(50, color=vline_col, linewidth=1)
                ax.axvline(move_time * 100 + 50, color=vline_col, linewidth=1, linestyle='--')
                ax.axhline(down_limit, color=hline_col, linewidth=1, linestyle='--')
                ax.set_xlim(30,100)
                ax.set_xticks([50, move_time * 100 + 50])
                ax.set_xticklabels(['Stimulus', f'Move={round(move_time,3)}'], fontsize=8)
                ax.set_xlabel(self.x_y_labels[i*3+j][0])
                ax.set_ylabel(self.x_y_labels[i*3+j][1])
                ax.set_title(f'n_trial: {n_trial}', fontsize=8)
                
                # Bar position (outside the heatmap)
                bar_x_start = 30  # Place bar further left
                bar_width = 1  # Width of the rectangle
                bar_y_start =  0  # Align to row

                # Create and add rectangle (1-row-length bar)
                bar = patches.Rectangle((bar_x_start, bar_y_start), bar_width, bar_n_sample, 
                                        linewidth=2, edgecolor="black", facecolor="black")
                ax.add_patch(bar)

                # Add text next to the rectangle
                ax.text(bar_x_start - 1.5, bar_n_sample/2, f"{bar_n_sample}\nSample", fontsize=5, 
                        verticalalignment='center', horizontalalignment='center', color="black", rotation=90)
                
        # Add a big colorbar
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])  # [left, bottom, width, height] (adjustable)
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
        fig.colorbar(sm, cax=cbar_ax, label='Firing rate (Hz)')
        
        plt.suptitle(f'Firing rate of neurons in different conditions - Sorted by: {sort_based_on}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        
    def plot_heatmap_trials_choice(self, cmap = 'magma', vmin = -1, vmax = 2, center = 0, median = False, sort_based_on = 'cccp', vline_col = 'green', hline_col = 'red',
                                   only_significant = False, bar_n_sample =5):
        cmap = sns.color_palette(cmap, as_cmap=True)
        if sort_based_on == 'cccp':
            sort_based_on_data = self.cccp_values
        elif sort_based_on == 'ccsp-vis':	
            sort_based_on_data = self.ccsp_vis_values
        elif sort_based_on == 'ccsp-aud':
            sort_based_on_data = self.ccsp_aud_values
        else:
            raise ValueError('sort_based_on should be either cccp or ccsp-vis or ccsp-aud')

        sort_indices = sort_based_on_data.argsort()
        down_limit = np.where(sort_based_on_data[sort_indices] > 0.025)[0][0] 
        up_limit = np.where(sort_based_on_data[sort_indices] > 0.975)[0][0] 
        if only_significant:
            sort_indices_significant = np.array(np.concatenate([np.arange(down_limit),
                                                                   np.arange(up_limit, 
                                                                            len(sort_indices))
                                                                ]),
                                                dtype = int)

        custom_order = []
        for row_group in range(0, 6, 2):  # Iterate over row pairs (0-1, 2-3, 4-5)
            for col in range(6):          # Iterate over columns
                custom_order.append((row_group, col))
                custom_order.append((row_group + 1, col))
        choice = [None, 'left', 'right', 'n']
    
        fig, axs = plt.subplots(6, 6, figsize=(20, 20), sharex=False, sharey=True)
        for index, (r, c) in enumerate(custom_order):
            ax = axs[r, c]
            trial = index//4
            choice_no = index%4
            mask = self.conditions[trial]
            mask_choice= mask.copy()
            mask_choice['choice'] = choice[choice_no]  
            self.select_data_interest(**mask_choice) 
            n_trial = self.interest_data['firing_rate'].shape[1]
            if choice_no == 0:
                ax.set_title(f'Choice: All, n_trial = {n_trial}', fontsize=8)
            elif choice_no == 3:
                ax.set_title(f'Choice: None, n_trial = {n_trial}', fontsize=8)
            else:
                ax.set_title(f'Choice: {choice[choice_no]}, n_trial = {n_trial}', fontsize=8)   
            if n_trial == 0:
                continue
            data_for_heatmap = self.interest_data['firing_rate'].mean(axis=1)
            data_for_heatmap = data_for_heatmap[sort_indices]
            if only_significant:
                data_for_heatmap = data_for_heatmap[sort_indices_significant]
            else:
                ax.axhline(up_limit, color = hline_col, linewidth = 1, linestyle = '--')
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            sns.heatmap(data_for_heatmap, ax=ax, xticklabels=False, yticklabels=False, cbar=False, vmin=vmin, vmax=vmax, cmap=cmap, center = center, norm=norm)
            ax.axhline(down_limit, color=hline_col, linewidth=0.5, linestyle='--')
            ax.set_xlim(30,100)
            if mask_choice['choice'] != 'n':
                if median:
                    move_time = np.nanmedian(self.interest_data['start_time']['first_move'])
                else:
                    move_time = np.nanmean(self.interest_data['start_time']['first_move'])	
                ax.set_xticks([50, move_time * 100 + 50])
                ax.set_xticklabels(['Stim.', f'Move = {round(move_time,3)}'], fontsize=8)
                ax.axvline(move_time * 100 + 50, color=vline_col, linewidth=1, linestyle='--')
            else:
                ax.set_xticks([50])
                ax.set_xticklabels(['Stim.'], fontsize=8)
            ax.axvline(50, color=vline_col, linewidth=1)
            
            # Bar position (outside the heatmap)
            bar_x_start = 30  # Place bar further left
            bar_width = 1  # Width of the rectangle
            bar_y_start =  0  # Align to row

            # Create and add rectangle (1-row-length bar)
            bar = patches.Rectangle((bar_x_start, bar_y_start), bar_width, bar_n_sample, 
                                    linewidth=2, edgecolor="black", facecolor="black")
            ax.add_patch(bar)

            # Add text next to the rectangle
            ax.text(bar_x_start - 1.5, bar_n_sample/2, f"{bar_n_sample}\nSample", fontsize=5, 
                    verticalalignment='center', horizontalalignment='center', color="black", rotation=90)

        # Add separation lines
        line_color = "black"
        line_width = 2

        # Add vertical lines between columns 2 & 4
        for x_pos in [2, 4.0]:
            fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color=line_color, linewidth=line_width))

        # Add horizontal lines between rows 1 & 3
        for y_pos in [2.05/6, 3.95/6]:
            fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color=line_color, linewidth=line_width))

        # Add a big colorbar
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])  # [left, bottom, width, height] (adjustable)
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
        fig.colorbar(sm, cax=cbar_ax, label='Firing rate (Hz)')

        plt.suptitle('Firing rate of neurons in different conditions and choice\n', fontsize=16)
        plt.tight_layout()
        plt.show()