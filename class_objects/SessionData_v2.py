from typing import Optional, Literal, List

from utils.trial_utils import *
from utils.spike_utils import *
from utils.utils import *

from plotting.session_plotter import SessionPlotter

from stats.cccp_ccsp import *
from stats.test_fr_change import test_fr_change_ttest

import pandas as pd


# from utils.pca_utils import apply_dim_red_to_mean_fr_reduce_fr, apply_dim_red_to_mean_fr_reduce_neuron

class SessionData:
    '''
    A class to store all data for a single session. 
    This includes the formatted event data, cluster data, and spike data. 
    '''
    def __init__(self, formatted_events, 
                 formatted_cluster_0, formatted_cluster_1, 
                 spikes_0, spikes_1, 
                 session_no, session_date, animal_id,
                 only_validTrails = True,
                 baseline_start = -0.2, stimulus_end = 0.5, binsize = 0.01, 
                 bombcell = True):
        '''
        Initialize the SessionData object with the formatted event, cluster, and spike data.
        Parameters:
        formatted_events (pd.DataFrame): The formatted event data for the session.
        formatted_cluster_0 (pd.DataFrame): The formatted cluster data for the first probe.
        formatted_cluster_1 (pd.DataFrame): The formatted cluster data for the second probe.
        spikes_0 (pd.DataFrame): The spike data for the first probe.
        spikes_1 (pd.DataFrame): The spike data for the second probe.
        session_no (int): The session number.
        session_date (str): The date of the session.
        animal_id (int): The animal ID.
        only_validTrails (bool): Whether to only include valid trials.
        baseline_start (float): The start time for the baseline period.
        stimulus_end (float): The end time for the stimulus period.
        binsize (float): The bin size for firing rate calculation.
        bombcell (bool): Whether to include bomb cells.
        
        Returns:
        None
        '''
        
        # prepare cluster data - combine probes
        formatted_cluster_0['probe'] = 0
        formatted_cluster_1['probe'] = 1
        formatted_cluster_1.clusterID += len(formatted_cluster_0)
        formatted_cluster_1.cluster_id += len(formatted_cluster_0)
        formatted_cluster_data_ind = pd.concat([formatted_cluster_0, formatted_cluster_1], axis=0)
        
        # update cluster data
        self.all_cluster_data = formatted_cluster_data_ind.reset_index(drop=True)
        
        # update bombcell keys
        self.bombcell_keys = np.sort(self.all_cluster_data[self.all_cluster_data.bombcell_class == 'good'].cluster_id.unique())
        
        # spike data - combine probes
        spikes_1.clusters += + len(formatted_cluster_0)
        self.all_spike_data = merge_spikes(spikes_0, spikes_1)
        
        self.all_event_data = formatted_events
        self.animal_id = animal_id
        self.session_no = session_no
        self.session_date = session_date
        self.only_validTrial = only_validTrails
        
        # format trials and stored in self.trials_formatted
        self.format_trials()
        
        ##### format spikes and stored in self.formatted_data and self.spikes_formatted
        ### self.formatted_data is a dictionary and the keys are the Neuron IDs.
        # each key contains 'spikes', 'firing_rate', 'firing_rate_raw', 'times'
        
        ### self.spikes_formatted is a dictionary and the keys are 'firing_rate', 'start_time', 'neuron_ids'
        # firing_rate contains 'baseline' and 'stimulus' keys, each containing np.array shaped (n_neurons, n_trials, n_bins(=n_timepoints))
        # start_time contains 'first_move' and 'choice_move' keys, each containing np.array shaped (n_trials,)
        self.format_spikes(baseline_start=baseline_start, 
                           stimulus_end = stimulus_end, 
                           bin_size = binsize, bombcell = bombcell)

        # this is different trials - visual and auditory stimulus locations
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
        # this is the labels for the x and y axis for the grid plot
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

        self.ccxp_window_values = None
        
    def format_trials(self):
        self.trials_formatted = format_trials(self)


    def format_spikes(self, baseline_start = -0.5, stimulus_end = 0.5, bin_size = 0.01):
        '''
        Format the spike data for the session. Creates self.formatted_data and self.spikes_formatted.
        
        self.formatted_data is a dictionary and the keys are the Neuron IDs.
        Each key contains 'spikes', 'firing_rate', 'firing_rate_raw', 'times'.
        
        self.spikes_formatted is a dictionary and the keys are 'firing_rate', 'start_time', 'neuron_ids'.
        'firing_rate' contains 'baseline' and 'stimulus' keys, each containing np.array shaped (n_neurons, n_trials, n_bins(=n_timepoints)).
        'start_time' contains 'first_move' and 'choice_move' keys, each containing np.array shaped (n_trials,).
        
        Parameters:
        baseline_start (float): The start time for the baseline period.
        stimulus_end (float): The end time for the stimulus period.
        bin_size (float): The bin size for firing rate calculation.
        
        Returns:
        None
        '''
        
        self.baseline_start = baseline_start
        self.stimulus_stop = stimulus_end
        
        self.formatted_data, self.spikes_formatted = organise_spikes(self, 
                                                                     bin_size = bin_size, 
                                                                     baseline_start = baseline_start, 
                                                                     stimulus_end = stimulus_end)
        
    def select_data_interest(self, neuron_indices = None, **kwargs):
        '''
        Select the data of interest for the session.
        You can select a subset of neurons and/or trials based on the kwargs.
        To select different trials (e.g., based on the auditory stimulus location), self.conditions can be used.
        
        It stores the selected data in self.interest_data.
        '''
        filtered_trials, original_trial_indices = select_trials(self, **kwargs)
        
        # filter the trials
        fr_baseline = self.spikes_formatted['firing_rate']['baseline'][:,original_trial_indices]
        fr_stimulus = self.spikes_formatted['firing_rate']['stimulus'][:,original_trial_indices]
        
        # filter the neruons
        if neuron_indices is not None:
            fr_baseline = fr_baseline[neuron_indices]
            fr_stimulus = fr_stimulus[neuron_indices]
        fr_all = np.concatenate([fr_baseline, fr_stimulus], axis=2)
        
        # get the start times
        start_time_first_move = self.spikes_formatted['start_time']['first_move'][original_trial_indices]
        start_time_choice_move = self.spikes_formatted['start_time']['choice_move'][original_trial_indices]
        
        # store the data
        interest_data = {'firing_rate': fr_all, 
                        'firing_rate_baseline': fr_baseline,
                        'firing_rate_stimulus': fr_stimulus,
                        'start_time': {'first_move': start_time_first_move, 'choice_move': start_time_choice_move},
                        'neuron_ids': self.spikes_formatted['neuron_ids'],
                        'trial_indices': original_trial_indices,
                        'trials': filtered_trials}
        
        self.interest_data = interest_data
        
    def test_ttest(self, stimulus_start_second = 0, 
                   stimulus_until_second = 0.2, 
                   parametric_test = True, 
                   multiple_correction = 'bonferroni', return_results = False):
        '''
        Perform a t-test to test for significant changes in firing rate between the baseline and stimulus periods.
        
        Parameters:
        stimulus_start_second (float): The start time for the stimulus period.
        stimulus_until_second (float): The end time for the stimulus period.
        parametric_test (bool): Whether to use a parametric test.
        multiple_correction (str): The multiple correction method to use.
        return_results (bool): Whether to return the results.
        '''
        # initiate the dictionary to store the results
        results = {'stat': [], 
                   'p_val': [], 
                   'rejected': [], 
                   'corrected_p_vals': [], 
                   'n_significant': [], 
                   'n_significant_mc': [],
                   'freq_significant_mc': [],
                   'one_sided': [],
                   'n_trials': []}
        
        # two-sided, increased, decreased
        for one_sided in [False, 'increased', 'decreased']:
            # perform the t-test
            stat, p_val, rejected, corrected_p_vals, n_significant, n_significant_mc, n_trials = test_fr_change_ttest(
                self, 
                stimulus_start_second=stimulus_start_second, 
                stimulus_until_second=stimulus_until_second,
                parametric_test = parametric_test, 
                one_sided = one_sided, 
                multiple_correction = multiple_correction)
            
            # store the results
            results['stat'].append(stat)
            results['p_val'].append(p_val)
            results['rejected'].append(rejected)	
            results['corrected_p_vals'].append(corrected_p_vals)
            results['n_significant'].append(n_significant)
            results['n_significant_mc'].append(n_significant_mc)
            results['freq_significant_mc'].append(n_significant_mc/len(self.spikes_formatted['neuron_ids']))
            results['one_sided'].append(one_sided)
            results['n_trials'].append(n_trials)
        if return_results:
            return results
        else:
            self.ttest_results = results
        
    def test_ttest_grid(self, 
                        stimulus_start_second = 0, 
                        stimulus_until_second = 0.2,
                        parametric_test = True,
                        multiple_correction = 'Bonferroni'):
        '''
        Perform a t-test for each condition in self.conditions.
        
        Parameters:
        stimulus_start_second (float): The start time for the stimulus period.
        stimulus_until_second (float): The end time for the stimulus period.
        parametric_test (bool): Whether to use a parametric test.
        multiple_correction (str): The multiple correction method to use.
        
        '''
        # initiate the dictionary to store the results
        multi_results = {'stat':[], 'p_val':[], 'rejected':[], 'corrected_p_vals':[], 'n_significant':[], 'freq_significant_mc':[], 'n_trials':[], 'one_sided':[]}

        # iterate over the conditions
        for cond in self.conditions:
            # select the data of interest
            self.select_data_interest(**cond)
            
            # perform the t-test
            results = self.test_ttest(stimulus_start_second=stimulus_start_second,
                                    stimulus_until_second=stimulus_until_second, 
                                    parametric_test=parametric_test, 
                                    multiple_correction=multiple_correction, return_results = True)
            
            # store the results
            multi_results['stat'].append(results['stat'])
            multi_results['p_val'].append(results['p_val'])
            multi_results['rejected'].append(results['rejected'])
            multi_results['corrected_p_vals'].append(results['corrected_p_vals'])
            multi_results['n_significant'].append(results['n_significant'])
            multi_results['freq_significant_mc'].append(results['freq_significant_mc'])
            multi_results['one_sided'].append(results['one_sided'])
            multi_results['n_trials'].append(results['n_trials'])
            
        # change the lists to numpy arrays
        multi_results['stat'] = np.array(multi_results['stat'])
        multi_results['p_val'] = np.array(multi_results['p_val'])
        multi_results['rejected'] = np.array(multi_results['rejected'])
        multi_results['corrected_p_vals'] = np.array(multi_results['corrected_p_vals'])
        multi_results['n_significant'] = np.array(multi_results['n_significant'])
        multi_results['freq_significant_mc'] = np.array(multi_results['freq_significant_mc'])
        multi_results['one_sided'] = np.array(multi_results['one_sided'])
        multi_results['n_trials'] = np.array(multi_results['n_trials'])
        
        self.ttest_results_grid = multi_results
    
    def test_cccp_ccsp(self, range_data = [0,0.2], until_movement = False, delta_before_movement = 0.05, start_before_movement = 0.2,
                            plot = True, sort_by:Optional[Literal['cccp', 'ccsp-vid', 'ccsp-aud']] = 'cccp'):
        '''
        Calculate the CCCP and CCSP values for the session.
        If until_movement is chosen, the start_before_movement and delta_before_movement parameters can be used to specify the time before the movement.
        start_before_movement will be the beginning of the window and delta_before_movement will be the end of the window.
        start_before_movement and delta_before_movement are seconds and both positive.
        
        Parameters:
        range_data (list): The range of timepoints to consider.
        until_movement (bool): Whether to consider the timepoints until the movement.
        delta_before_movement (float): The time before the movement.
        start_before_movement (float): The start time before the movement. If until movement is chose, you can choose the start time before the movement.
        plot (bool): Whether to plot the results.
        sort_by (str): The value to sort by.'''
        
        self.ccxp_neuron_ids, self.cccp_values, self.ccsp_aud_values, self.ccsp_vis_values = calculate_cccp_ccsp(self, range_data=range_data,
                                                                                           until_movement=until_movement,
                                                                                           start_before_movement = start_before_movement,
                                                                                           delta_before_movement = delta_before_movement)
        if plot:
            plotter = SessionPlotter(self)
            plotter.plot_cccp_ccsp(sort_by = sort_by)

        
    def test_cccp_ccsp_window(self, start_time = -0.5, end_time = 0.5, step = 0.1, plot = True, figsize = (10,10)):
        '''
        Calculate the CCCP and CCSP values for different time windows.
        
        Parameters:
        start_time (float): The start time.
        end_time (float): The end time.
        step (float): The step size.
        plot (bool): Whether to plot the results.
        figsize (tuple): The figure size.
        '''
        
        # calculate the time windows based on the start, end, and step
        time_windows = [[round(start_time + i*step,4), round(start_time + (i+1)*step,4)] for i in range(int((end_time-start_time)/step))]
        
        # initiate the lists to store the values
        cccp_values = []
        ccsp_aud_values = []
        ccsp_vis_values = []
    
        # iterate over the time windows
        for window in time_windows:
            # calculate the CCCP and CCSP values
            self.test_cccp_ccsp(range_data=window, until_movement=False, plot = False)
            
            # store the values
            cccp_values.append(self.cccp_values)
            ccsp_aud_values.append(self.ccsp_aud_values)
            ccsp_vis_values.append(self.ccsp_vis_values)

        # calculate the count and frequency of up and down values
        cccp_values, cccp_up_count, cccp_up_freq, cccp_down_count, cccp_down_freq = return_ccxp_count_freq(cccp_values)
        ccsp_aud_values, ccsp_aud_up_count, ccsp_aud_up_freq, ccsp_aud_down_count, ccsp_aud_down_freq = return_ccxp_count_freq(ccsp_aud_values)
        ccsp_vis_values, ccsp_vis_up_count, ccsp_vis_up_freq, ccsp_vis_down_count, ccsp_vis_down_freq = return_ccxp_count_freq(ccsp_vis_values)
        
        # store the values
        cccp = [cccp_values, cccp_up_count, cccp_up_freq, cccp_down_count, cccp_down_freq]
        ccsp_aud = [ccsp_aud_values, ccsp_aud_up_count, ccsp_aud_up_freq, ccsp_aud_down_count, ccsp_aud_down_freq]
        ccsp_vis = [ccsp_vis_values, ccsp_vis_up_count, ccsp_vis_up_freq, ccsp_vis_down_count, ccsp_vis_down_freq]
        
        self.ccxp_window_values = [cccp, ccsp_aud, ccsp_vis]
        
        # plot the results
        if plot:
            plotter = SessionPlotter(self)
            plotter.plot_ccxp(time_windows = time_windows, figsize = figsize)
            
    

    # def apply_dim_red_to_mean_fr(self, reduce_freq, plot_projection = True, plot_var = True, var_log = False, pc_x = 0, pc_y=1, return_obj = False,
    #                              pca = True, umap = False, line = True, random_state =42):
    #     if reduce_freq:
    #         dim_red_obj = apply_dim_red_to_mean_fr_reduce_fr(self, plot_projection=plot_projection, plot_var = plot_var, var_log = var_log, pc_x = pc_x, pc_y = pc_y,
    #                                                          pca = pca, umap = umap, random_state = random_state, line = line)
    #     else:
    #         dim_red_obj = apply_dim_red_to_mean_fr_reduce_neuron(self, plot_projection=plot_projection, plot_var = plot_var, var_log = var_log, pc_x = pc_x, pc_y = pc_y,
    #                                                          pca = pca, umap = umap, random_state = random_state, line = line)
    #     if return_obj:
    #         return dim_red_obj
        
    # def apply_dim_red_to_mean_fr_reduce_fr_grid(self, pc_x = 0, pc_y = 1, significant_neuron = None,
    #                                         pca = True, umap = False, random_state = 42, xlim = None, ylim = None):
    #     increase_fr = self.significant_neurons_results_grid[0]
    #     decrease_fr = self.significant_neurons_results_grid[1]

        
    #     dim_red_obj = self.apply_dim_red_to_mean_fr(reduce_freq = True, plot_projection = False, plot_var = False, return_obj = True,
    #                                             pca = pca, umap = umap, random_state = random_state)
        
    #     custom_order = []
    #     for row_group in range(0, 6, 2):  # Iterate over row pairs (0-1, 2-3, 4-5)
    #         for col in range(6):          # Iterate over columns
    #             custom_order.append((row_group, col))
    #             custom_order.append((row_group + 1, col))
    #     choice = [None, 'left', 'n', 'right']
    #     fig, axs = plt.subplots(6, 6, figsize=(12, 12), sharex=True, sharey=True)

    #     for index, (r, c) in enumerate(custom_order):
    #         ax = axs[r, c]
    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)
    #         trial = index//4

    #         choice_no = index%4

    #         mask = self.conditions[trial]
    #         mask_choice= mask.copy()
    #         mask_choice['choice'] = choice[choice_no]  
    #         norm = plt.Normalize(vmin=-1, vmax=1)  

    #         self.select_data_interest(neuron_indices = significant_neuron, **mask_choice)
    #         n_trial = self.interest_data['firing_rate'].shape[1]
    #         if choice_no == 0:
    #             ax.set_title(f'Choice: All, n_trial = {n_trial}', fontsize=8)
    #         elif choice_no == 2:
    #             ax.set_title(f'Choice: None, n_trial = {n_trial}', fontsize=8)
    #         else:
    #             ax.set_title(f'Choice: {choice[choice_no]}, n_trial = {n_trial}', fontsize=8)

    #         if self.interest_data['firing_rate'].shape[1] == 0:
    #             pass
    #         else:
    #             colors = np.zeros(self.interest_data['firing_rate'].shape[0])
    #             increase_indices = increase_fr[trial][choice_no][2]
    #             decrease_indices = decrease_fr[trial][choice_no][2]
    #             if significant_neuron is not None:
    #                 increase_indices = np.array(increase_indices)[significant_neuron]
    #                 decrease_indices = np.array(decrease_indices)[significant_neuron]
    #             colors[increase_indices] = 1
    #             colors[decrease_indices] = -1
    #             projected_data = dim_red_obj.transform(self.interest_data['firing_rate'].mean(axis=1))
    #             ax.scatter(projected_data[:, pc_x], projected_data[:, pc_y], c=colors, alpha=0.5, cmap='coolwarm', norm = norm)
    #             ax.set_xlabel(f'n_inc = {np.sum(increase_indices)}, n_dec = {np.sum(decrease_indices)}', fontsize=8)

    #     # Add separation lines
    #     line_color = "black"
    #     line_width = 2

    #     # Add vertical lines between columns 2 & 4
    #     for x_pos in [2.15, 4.05]:
    #         fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color=line_color, linewidth=line_width))

    #     # Add horizontal lines between rows 1 & 3
    #     for y_pos in [2/6, 3.9/6]:
    #         fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color=line_color, linewidth=line_width))

    #     plt.suptitle('PC space for trial types and choices', fontsize=16)
    #     plt.tight_layout()
    #     plt.show()
        
    # def apply_dim_red_to_mean_fr_reduce_neuron_grid(self, pc_x = 0, pc_y = 1, significant_neuron = None,
    #                                                 pca = True, umap = False, random_state = 42, xlim = None, ylim = None):
    #     dim_red_obj = self.apply_dim_red_to_mean_fr(reduce_freq = False, plot_projection = False, plot_var = False, return_obj = True,
    #                                             pca = pca, umap = umap, random_state = random_state)
    #     custom_order = []
    #     for row_group in range(0, 6, 2):  # Iterate over row pairs (0-1, 2-3, 4-5)
    #         for col in range(6):          # Iterate over columns
    #             custom_order.append((row_group, col))
    #             custom_order.append((row_group + 1, col))
                
    #     choice = [None, 'left', 'n', 'right']

    #     fig, axs = plt.subplots(6, 6, figsize=(12, 12), sharex=True, sharey=True)
    #     for index, (r, c) in enumerate(custom_order):
    #         ax = axs[r, c]
    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)
    #         trial = index//4
            
    #         choice_no = index%4

    #         mask = self.conditions[trial]
    #         mask_choice= mask.copy()
    #         mask_choice['choice'] = choice[choice_no]  
    #         norm = plt.Normalize(vmin=-0.5, vmax=0.5)  

    #         self.select_data_interest(neuron_indices= significant_neuron, **mask_choice)
    #         n_trial = self.interest_data['firing_rate'].shape[1]
    #         if choice_no == 0:
    #             ax.set_title(f'Choice: All, n_trial = {n_trial}', fontsize=8)
    #         elif choice_no == 2:
    #             ax.set_title(f'Choice: None, n_trial = {n_trial}', fontsize=8)
    #         else:
    #             ax.set_title(f'Choice: {choice[choice_no]}, n_trial = {n_trial}', fontsize=8)
            
    #         if self.interest_data['firing_rate'].shape[1] == 0:
    #             pass
    #         else:
    #             colors = self.times
    #             n_trial = self.interest_data['firing_rate'].shape[1]
    #             increase_indices = self.significant_neurons_results_grid[0][trial][choice_no][2]
    #             decrease_indices = self.significant_neurons_results_grid[1][trial][choice_no][2]
    #             if significant_neuron is not None:
    #                 increase_indices = np.array(increase_indices)[significant_neuron]
    #                 decrease_indices = np.array(decrease_indices)[significant_neuron]
                    
    #             projected_data = dim_red_obj.transform(self.interest_data['firing_rate'].mean(axis=1).T)
    #             ax.scatter(projected_data[:, pc_x], projected_data[:, pc_y], c=colors, alpha=0.5, cmap='coolwarm', norm = norm)
    #             ax.set_xlabel(f'n_inc = {np.sum(increase_indices)}, n_dec = {np.sum(decrease_indices)}', fontsize=8)

    #     # Add separation lines
    #     line_color = "black"
    #     line_width = 2

    #     # Add vertical lines between columns 2 & 4
    #     for x_pos in [2.15, 4.05]:
    #         fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color=line_color, linewidth=line_width))

    #     # Add horizontal lines between rows 1 & 3
    #     for y_pos in [2/6, 3.9/6]:
    #         fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color=line_color, linewidth=line_width))

    #     plt.suptitle('PC space for trial types and choices', fontsize=16)
    #     plt.tight_layout();plt.show()