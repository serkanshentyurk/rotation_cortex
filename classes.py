from typing import Optional, Literal, List
from collections import defaultdict
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from utils_class import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

class exp_data:
    def __init__(self):
        self.session_data_obj = defaultdict()
        self.session_dates = []
        self.dominance = []
        
    def add_session(self, session_object):
        self.session_data_obj[session_object.session_no] = session_object
        self.session_dates.append(session_object.session_date)
        self.dominance.append(session_object.dominant_modality)    
        
    def return_mask(self,**kwargs):   
        for obj_ind in self.session_data_obj:
            self.session_data_obj[obj_ind].return_mask(**kwargs)
         
    def test_firing_rate_change_stimulus_sessions(self, 
                                                  parametric_test = True, 
                                                  one_sided = True, 
                                                  multiple_correction = True, 
                                                  mc_test:Literal['bonferroni', 'sidak', 'holm', 'fdr_bh'] = 'fdr_bh', 
                                                  mc_alpha = 0.05,
                                                  plot = True,
                                                  masks = None):

        freq_up_regulated_masks = []
        freq_down_regulated_masks = []
        n_total_masks = []
        freq_no_change_masks = []

        if type(masks) != list:
            masks = [masks]

        for mask in masks:
            freq_up_regulated = []
            freq_down_regulated = []
            n_total = []
            freq_no_change = []
            for obj_index in self.session_data_obj:
                obj = self.session_data_obj[obj_index]
                if mask == None:
                    obj.return_mask()
                else:
                    obj.return_mask(**mask)
                up_regulated = False
                obj.test_firing_rate_change_stimulus(parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                                                multiple_correction = multiple_correction, mc_test = mc_test, mc_alpha = mc_alpha)
                n_total_trial = obj.p_vals['info']['n_total']
                n_total.append(n_total_trial)
        
                n_down_regulated_trial = obj.p_vals['info']['n_significant_mc']
                freq_down_regulated.append(n_down_regulated_trial / n_total_trial)
                
                up_regulated = True
                obj.test_firing_rate_change_stimulus(parametric_test = parametric_test, one_sided = one_sided, up_regulated = up_regulated,
                                                multiple_correction = multiple_correction, mc_test = mc_test, mc_alpha = mc_alpha)
                
                n_up_regulated_trial = obj.p_vals['info']['n_significant_mc']
                freq_up_regulated.append(n_up_regulated_trial / n_total_trial)
                
                no_change_trial_freq = 1 - n_up_regulated_trial / n_total_trial - n_down_regulated_trial / n_total_trial
                freq_no_change.append(no_change_trial_freq)
        
            freq_up_regulated_masks.append(freq_up_regulated)
            freq_down_regulated_masks.append(freq_down_regulated)
            n_total_masks.append(n_total)
            freq_no_change_masks.append(freq_no_change)

        self.test_firing_rate_change_stimulus_sessions_results = {'freq_up_regulated': freq_up_regulated_masks,
                                                                        'freq_down_regulated': freq_down_regulated_masks,
                                                                        'freq_no_change': freq_no_change_masks,
                                                                        'n_total': n_total_masks,
                                                                        'info': {'mask': masks,
                                                                                'parametric_test': parametric_test,
                                                                                'one_sided': one_sided,
                                                                                'multiple_correction': multiple_correction,
                                                                                'mc_test': mc_test, 
                                                                                'mc_alpha': mc_alpha}}

    
        if plot:
            self.plot_firing_rate_change_stimulus_sessions()
    
    def plot_firing_rate_change_stimulus_sessions(self, figsize = (12,6), xlim = None, ylim = None):
        
        session_info = self.test_firing_rate_change_stimulus_sessions_results['info']
        colours = ['green', 'red','gray']

        if len(self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated']) == 1:
            fig, axs = plt.subplots(1, 1, figsize = figsize)
            axs = [axs]
            plt.suptitle(f"Frequency of Significant Changes in Firing Rate  (p < {session_info['mc_alpha']}; Multiple Correction: {session_info['multiple_correction']}-{session_info['mc_test']}) across sessions - All Stimuli")
       
        else:
            fig, axs = plt.subplots(3,3, figsize = figsize, sharex=True, sharey=True)
            axs = axs.flatten() if isinstance(axs, np.ndarray) else axs

            x_labels = ['Left', 'Off', 'Right']
            y_labels = ['Right', 'Center', 'Left']
            fig.text(0.5, -0.03, 'Visual stimulus', ha='center', fontsize=12)
            fig.text(-0.03, 0.5, 'Auditory stimulus', va='center', rotation='vertical', fontsize=12)
            for i in range(3):
                axs[i+6].set_xlabel(f'{x_labels[i]}')  # x labels for the bottom row
                axs[3*i].set_ylabel(f'{y_labels[i]}')  # y labels for the left column
            plt.suptitle(f"Frequency of Significant Changes in Firing Rate  (p < {session_info['mc_alpha']}; Multiple Correction: {session_info['multiple_correction']}-{session_info['mc_test']}) across sessions")

        for i in range(len(self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated'])):
            ax = axs[i]
            ax.bar(self.session_dates, self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated'][i], 
                    label='Increased Activity', color = colours[0], alpha = 0.7)
        
            ax.bar(self.session_dates, self.test_firing_rate_change_stimulus_sessions_results['freq_down_regulated'][i], 
                    label='Decreased Activity', color = colours[1], alpha = 0.7,
                    bottom=self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated'][i])
        
            ax.bar(self.session_dates, self.test_firing_rate_change_stimulus_sessions_results['freq_no_change'][i], 
                    label = 'No Change', color = colours[2], alpha = 0.7,
                    bottom=np.array(self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated'][i])+np.array(self.test_firing_rate_change_stimulus_sessions_results['freq_down_regulated'][i]))
            if len(self.test_firing_rate_change_stimulus_sessions_results['freq_up_regulated']) > 1:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xticks(self.session_dates)
                ax.set_xticklabels(self.session_dates, rotation = 45)

        # Display the plot
        plt.tight_layout()
        plt.show()
        
       
            
class session_data:
    '''
    A class to store all data for a single session. 
    This includes the formatted event data, cluster data, and spike data. 
    '''
    def __init__(self, formatted_events, formatted_cluster_data, spikes, 
                 dominant_modality, session_no, session_date, 
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
        self.dominant_modality = dominant_modality
        self.session_no = session_no
        self.session_date = session_date
        self.spikes = defaultdict(list)
        self.only_validTrial = only_validTrails
        
        self.format_trials()
        self.create_mask()
        self.format_spikes(baseline_start=baseline_start, stimulus_end = stimulus_end, bin_size = binsize, normalise = normalise, bombcell = bombcell)

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
    
    
    def create_mask(self):
        visual_mask = []
        vis_all = np.array(self.trials_formatted.is_visualTrial)
        vis_left = np.array(self.trials_formatted.vis_loc == 'l')
        vis_right = np.array(self.trials_formatted.vis_loc == 'r')
        vis_off = np.array(self.trials_formatted.vis_loc == 'o')
        visual_mask.append([vis_all, vis_left, vis_right, vis_off])
        self.visual_mask = visual_mask
        
        audio_mask = []
        audio_all = np.array(self.trials_formatted.is_auditoryTrial)
        audio_left = np.array(self.trials_formatted.aud_loc == 'l')
        audio_right = np.array(self.trials_formatted.aud_loc == 'r')
        audio_center = np.array(self.trials_formatted.aud_loc == 'c')
        audio_mask.append([audio_all, audio_left, audio_right, audio_center])
        self.audio_mask = audio_mask
        
        coherent_mask = []
        coherent_mask.append(np.array(self.trials_formatted.is_coherentTrial))
        self.coherent_mask = coherent_mask
        
        conflict_mask = []
        conflict_mask.append(np.array(self.trials_formatted.is_conflictTrial))
        self.conflict_mask = conflict_mask
        
        choice_left = []
        choice_right = []
        choice_no = []
        choice_left.append(np.array(self.trials_formatted.choice == -1))
        choice_right.append(np.array(self.trials_formatted.choice == 1))
        choice_no.append(np.array(self.trials_formatted.choice == 0))
        self.choice_left = choice_left
        self.choice_right = choice_right
        self.choice_no = choice_no
        
        result_correct_mask = []
        result_incorrect_mask = []
        result_none_mask = []
        result_correct_mask.append(np.array(self.trials_formatted.feedback == 1))
        result_incorrect_mask.append(np.array(self.trials_formatted.feedback == -1))
        result_none_mask.append(np.array(self.trials_formatted.feedback == 0))
        self.result_correct_mask = result_correct_mask
        self.result_incorrect_mask = result_incorrect_mask
        self.result_none_mask = result_none_mask
        
    def return_mask(self,
                    modality: Optional[Literal['visual', 'audio', 'all', 'coherent', 'conflict', 'v', 'a']] = None, 
                    stim_loc: Optional[Literal['left', 'right', 'all', 'stim_left', 'stim_right', 'l', 'r']] = None,
                    vis_loc: Optional[Literal['left', 'right', 'all', 'l', 'r', 'o', 'c']] = None,
                    aud_loc: Optional[Literal['left', 'right', 'all', 'l', 'r', 'o', 'c']] = None,
                    choice_correct: Optional[Literal['correct', 'incorrect', 'all', 'no_choice', 'c', 'i']] = None,
                    choice_loc: Optional[Literal['left', 'right', 'all', 'no_choice', 'l', 'r']] = None, 
                    conflict_info: Optional[Literal['visual_left', 'visual_right', 'audio_left', 'audio_right', 'all', 'v_l', 'v_r', 'a_l', 'a_r']] = None):
        if modality in ['visual', 'v']:
            map_stim = self.visual_mask[0]
            if stim_loc in ['left','stim_left', 'l']:
                map_stim = map_stim[1]
            elif stim_loc in ['right', 'stim_right' 'r']:
                map_stim = map_stim[2]
            elif stim_loc in ['all', None]:
                map_stim = map_stim[0]
            else:
                raise ValueError('stim_loc must be either "left", "right", "all" or None')
        elif modality in ['audio', 'a']:
            map_stim = self.audio_mask[0]
            if stim_loc in ['left','stim_left', 'l']:
                map_stim = map_stim[1]
            elif stim_loc in ['right', 'stim_right', 'r']:
                map_stim = map_stim[2]
            elif stim_loc in ['all', None]:
                map_stim = map_stim[0]
            else:
                raise ValueError('stim_loc must be either "left", "right", "all" or None')
        elif modality in ['all', None]:
            if stim_loc in ['left', 'stim_left', 'l']:
                map_stim = np.logical_or(self.visual_mask[0][1], self.audio_mask[0][1])
            elif stim_loc in ['right', 'stim_right' 'r']:
                map_stim = np.logical_or(self.visual_mask[0][2], self.audio_mask[0][2])
            elif stim_loc in ['all', None]:
                map_stim = np.ones(self.visual_mask[0][0].shape).astype(bool)
            else:
                raise ValueError('stim_loc must be either "left", "right", "all" or None')
        elif modality == 'coherent':
            map_stim = self.coherent_mask[0]
            if stim_loc in ['left', 'stim_left', 'l']:
                map_stim = np.logical_and(map_stim[0], self.visual_mask[0][1])
            elif stim_loc in ['right', 'stim_right', 'r']:
                map_stim = np.logical_and(map_stim[0], self.visual_mask[0][2])
            elif stim_loc in ['all', None]:
                map_stim = map_stim[0]
            else:
                raise ValueError('stim_loc must be either "left", "right", "all" or None')
        elif modality == 'conflict':
            map_stim = self.conflict_mask[0]
            if conflict_info in ['visual_left', 'visual_l', 'vl', 'v_l']:
                map_stim = np.logical_and(map_stim[0], self.visual_mask[0][1])
            elif conflict_info in ['visual_right', 'visual_r', 'vr', 'v_r']:
                map_stim = np.logical_and(map_stim[0], self.visual_mask[0][2])
            elif conflict_info in ['audio_left', 'audio_l', 'al', 'a_l']:
                map_stim = np.logical_and(map_stim[0], self.audio_mask[0][1])
            elif conflict_info in ['audio_right', 'audio_r', 'ar', 'a_r']:
                map_stim = np.logical_and(map_stim[0], self.audio_mask[0][2])
            elif conflict_info in ['all', None]:
                map_stim = map_stim[0]
        else:
            raise ValueError('modality must be either "visual", "audio", "coherent", "conflict", "all" or None')

        if vis_loc in [None, 'all']:
            map_stim = map_stim
        elif vis_loc in ['left', 'l']:
            map_stim = np.logical_and(map_stim, self.visual_mask[0][1])
        elif vis_loc in ['right', 'r']:
            map_stim = np.logical_and(map_stim, self.visual_mask[0][2])
        elif vis_loc in ['o', 'c']:
            map_stim = np.logical_and(map_stim, self.visual_mask[0][3])
        else:
            raise ValueError('vis_loc must be either "left", "right", "all" or None')
        if aud_loc in [None, 'all']:
            map_stim = map_stim
        elif aud_loc in ['left', 'l']:
            map_stim = np.logical_and(map_stim, self.audio_mask[0][1])
        elif aud_loc in ['right', 'r']:
            map_stim = np.logical_and(map_stim, self.audio_mask[0][2])
        elif aud_loc in ['c', 'o']:
            map_stim = np.logical_and(map_stim, self.audio_mask[0][3])
        else:
            raise ValueError('aud_loc must be either "left", "right", "all" or None')
        
        if choice_correct in ['correct', 'c']:
            map_stim = np.logical_and(map_stim, self.result_correct_mask[0])
        elif choice_correct in ['incorrect', 'i']:
            map_stim = np.logical_and(map_stim, self.result_incorrect_mask[0])
        elif choice_correct in ['all', None]:
            map_stim = map_stim
        elif choice_correct == 'no_choice':
            map_stim = np.logical_and(map_stim, self.result_none_mask[0])
        else:
            raise ValueError('choice_correct must be either "correct", "incorrect", "no_choice", "all" or None')
        if choice_loc in ['left', 'l']:
            map_stim = np.logical_and(map_stim, self.choice_left[0])
        elif choice_loc in ['right', 'r']:
            map_stim = np.logical_and(map_stim, self.choice_right[0])  
        elif choice_loc in ['all', None]:
            map_stim = map_stim
        elif choice_loc == 'no_choice':
            map_stim = np.logical_and(map_stim, self.choice_no[0])
        else:
            raise ValueError('choice_loc must be either "left", "right", "no_choice", "all" or None')
        self.trials_map = map_stim
        self.trials_map_indices = np.arange(0,self.trials_map.shape[0],1)[self.trials_map]
        self.mask_info = {'modality': modality, 
                          'stim_loc': stim_loc, 
                          'choice_correct': choice_correct, 
                          'choice_loc': choice_loc, 
                          'conflict_info': conflict_info,
                          'vis_loc': vis_loc,
                          'aud_loc': aud_loc}

    def format_spikes(self, baseline_start = -0.5, stimulus_end = 0.5, bin_size = 0.01, normalise = True, bombcell = True):
        '''
        Format the spike data for the session. This will create a dictionary of spike times for each neuron.
        '''
        self.baseline_start = baseline_start
        self.stimulus_stop = stimulus_end
        
        
        self.spikes = format_spike(self.all_spike_data)
        stimulus_spikes, baseline_spikes = self.extract_spikes(baseline_start)


        stimulus_firing_rates, stimulus_times = calculate_firing_rate(stimulus_spikes, bin_size, normalise = normalise,
                                                                      period_start = 0, period_end = stimulus_end)
        baseline_firing_rates, baseline_times = calculate_firing_rate(baseline_spikes, bin_size, normalise = normalise,
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
        
        baseline_arrays = [baseline_firing_rates[key] for key in keys]  # Extract arrays in the order of keys
        stimulus_arrays = [stimulus_firing_rates[key] for key in keys]  # Extract arrays in the order of keys

        baseline_3d = np.stack(baseline_arrays, axis=0)
        stimulus_3d = np.stack(stimulus_arrays, axis=0)
        
        timeline_first_move = self.trials_formatted.timeline_firstMoveOn - self.trials_formatted.timeline_audPeriodOn
        timeline_choice_move = self.trials_formatted.timeline_choiceMoveOn - self.trials_formatted.timeline_audPeriodOn
        self.spikes_formatted = {'firing_rate': {'baseline': baseline_3d, 
                                                     'stimulus': stimulus_3d},
                                     'neuron_ids': np.array(keys),
                                     'start_time': {'first_move': timeline_first_move.values, 
                                                    'choice_move': timeline_choice_move.values}}

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
            

    def plot_psth(self, figsize = (10, 6), xlim = None, ylim = None,
                  filter_info: Optional[List[dict]] = None, 
                  color_based_on: Optional[Literal['visual', 'audio']] = 'visual',
                  only_average = True):

        if filter_info is None:
            n_filter = 1
        else:
            n_filter = len(filter_info)
            
        n_freq = self.formatted_data[0]['firing_rate']['baseline'].shape[1] + self.formatted_data[0]['firing_rate']['stimulus'].shape[1]
        n_neuron = len(self.formatted_data)

        fig, axs = plt.subplots(n_filter, 1, figsize=figsize)
        for i in range(n_filter):
            average_fr = np.zeros(n_freq)
            average_fr_l = np.zeros(n_freq)
            average_fr_c = np.zeros(n_freq)
            average_fr_r = np.zeros(n_freq)
            if n_filter > 1:
                ax = axs[i]
            else:
                ax = axs
                
            for neuron_data in self.formatted_data.values():
                # Concatenate baseline and stimulus firing rates for the same neuron
                total_times = np.concatenate([neuron_data['times']['baseline'], neuron_data['times']['stimulus']])
            
                if color_based_on is None:
                    total_firing_rates = np.concatenate([neuron_data['firing_rate']['baseline'][self.trials_map].mean(axis = 0), 
                                            neuron_data['firing_rate']['stimulus'][self.trials_map].mean(axis = 0)])
                    
                    average_fr += total_firing_rates
                    if not only_average:
                        ax.plot(total_times[:neuron_data['times']['baseline'].shape[0]+1], total_firing_rates[:neuron_data['times']['baseline'].shape[0]+1], color='gray', alpha=0.5)
                        ax.plot(total_times[neuron_data['times']['baseline'].shape[0]:], total_firing_rates[neuron_data['times']['baseline'].shape[0]:], color='lightblue', alpha=0.5)
                    
                else: 
                    if filter_info is None:
                        filter_to_apply = self.mask_info.copy()
                    else:
                        filter_to_apply = filter_info[i].copy()
                    filter_1 = filter_to_apply.copy()
                    filter_2 = filter_to_apply.copy()
                    filter_3 = filter_to_apply.copy()

                    if color_based_on in ['vision', 'visual', 'vis', 'v']:
                        filter_1['vis_loc'] = 'l'
                        filter_2['vis_loc'] = 'o'
                        filter_3['vis_loc'] = 'r'
                    elif color_based_on in ['audio', 'aud', 'a']:
                        filter_1['aud_loc'] = 'l'
                        filter_2['aud_loc'] = 'c'
                        filter_3['aud_loc'] = 'r'
                    else:
                        raise ValueError('Invalid color_based_on')
                
                    self.return_mask(**filter_1)
                    total_firing_rates_l = np.concatenate([neuron_data['firing_rate']['baseline'][self.trials_map].mean(axis = 0), 
                                            neuron_data['firing_rate']['stimulus'][self.trials_map].mean(axis = 0)])
                    average_fr_l += total_firing_rates_l
                    self.return_mask(**filter_2)
                    total_firing_rates_c = np.concatenate([neuron_data['firing_rate']['baseline'][self.trials_map].mean(axis = 0), 
                                            neuron_data['firing_rate']['stimulus'][self.trials_map].mean(axis = 0)])
                    average_fr_c += total_firing_rates_c
                    self.return_mask(**filter_3)
                    total_firing_rates_r = np.concatenate([neuron_data['firing_rate']['baseline'][self.trials_map].mean(axis = 0), 
                                            neuron_data['firing_rate']['stimulus'][self.trials_map].mean(axis = 0)])
                    average_fr_r += total_firing_rates_r
                    
                    if not only_average:
                        ax.plot(total_times, total_firing_rates_l, color='#4169E1', alpha=0.1, label = f'{color_based_on} - Left')
                        ax.plot(total_times, total_firing_rates_c, color='gray', alpha=0.1, label=f'{color_based_on} - Off/Center')
                        ax.plot(total_times, total_firing_rates_r, color='#D32F2F', alpha=0.1, label=f'{color_based_on} - Right')
                    
            if color_based_on is not None:
                average_fr_l = average_fr_l / n_neuron
                average_fr_c = average_fr_c / n_neuron
                average_fr_r = average_fr_r / n_neuron
                ax.plot(total_times, average_fr_l, color='#4169E1', alpha=1, linewidth=2, label='Average Left')
                ax.plot(total_times, average_fr_c, color='gray', alpha=1, linewidth=2, label='Average Off')
                ax.plot(total_times, average_fr_r, color='#D32F2F', alpha=1, linewidth=2, label='Average Right')
                # Custom legend
                legend_elements = [
                    Line2D([0], [0], color='#4169E1', lw=2, alpha=0.7, label = f'{color_based_on} - Left'),
                    Line2D([0], [0], color='gray', lw=2, alpha=0.7, label = f'{color_based_on} - Off/Center'),
                    Line2D([0], [0], color='#D32F2F', lw=2, alpha=0.7, label = f'{color_based_on} - Right'),

                ]
            else:
                average_fr = average_fr / n_neuron
                ax.plot(total_times[:neuron_data['times']['baseline'].shape[0]+1], average_fr[:neuron_data['times']['baseline'].shape[0]+1], label=f'Average Baseline', color='black', alpha=0.7, linewidth=2)
                ax.plot(total_times[neuron_data['times']['baseline'].shape[0]:], average_fr[neuron_data['times']['baseline'].shape[0]:], label=f'Average Stimulus', color='blue', alpha=0.7, linewidth=2)
                # Custom legend
                legend_elements = [
                    Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Baseline FR'),
                    Line2D([0], [0], color='lightblue', lw=2, alpha=0.5, label='Stimulus FR'),
                    Line2D([0], [0], color='black', lw=2, alpha=0.7, label='Avg Baseline FR'),
                    Line2D([0], [0], color='blue', lw=2, alpha=0.7, label='Avg Stimulus FR')
                ]
            # Add labels and title
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Firing Rate (sp/s)")
            ax.set_title(f"{self.mask_info}", fontsize=10)
            ax.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper right')

            # Adjust the x-axis to remove any gaps between the baseline and stimulus
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axvline(0, color='k', linestyle='--', label="Stimulus Onset")
            
        # Show the plot
        plt.suptitle(f"PSTH - Session Date and Number: {self.session_date} & {self.session_no} - Dominant Modality: {self.dominant_modality}")
        plt.tight_layout()
        plt.show()
        
        
    def plot_spike_raster(self, neuron_ids=None, first_move = True,
                          sort_choice_loc = True, sort_response_time = True,
                          xlim = (-0.2,0.5), figsize = (10,6)):
        """
        Plots a combined spike raster for both baseline (before stimulus) and aligned (during stimulus) spikes,
        using different colors for the two periods on the same plot.
        
        Args:
        - neuron_ids: A list of neuron IDs to plot (default: None, plots all neurons).
        - figsize: The size of the figure to plot.
        """


        if first_move:
            timeline = self.trials_formatted.timeline_firstMoveOn - self.trials_formatted.timeline_audPeriodOn
        else:
            timeline = self.trials_formatted.timeline_choiceMoveOn - self.trials_formatted.timeline_audPeriodOn
            
        if neuron_ids is None:
            neuron_ids = self.formatted_data.keys()
            
            
        if not sort_choice_loc and not sort_response_time:
            sorted_index = self.trials_map_indices
        else:
            df = pd.DataFrame({'response_direction': self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices],
                            'timeline': timeline.to_numpy()[self.trials_map_indices]})
            if sort_choice_loc and sort_response_time:
                df_sorted = df.sort_values(by=['response_direction', 'timeline'])
                sorted_index = df_sorted.index
            elif sort_choice_loc:
                df_sorted = df.sort_values(by=['response_direction'])
                sorted_index = df_sorted.index
            else:
                df_sorted = df.sort_values(by=['timeline'])
                sorted_index = df_sorted.index

        plt.figure(figsize=figsize)

        colors = ['gray','blue', 'red']
            
        for neuron_id in neuron_ids:
            baseline_spikes = self.formatted_data[neuron_id]['spikes']['baseline']
            stimulus_spikes = self.formatted_data[neuron_id]['spikes']['stimulus']
            
            
            for i in range(len(sorted_index)):
                trial_num = sorted_index[i]
                x_baseline = np.array(baseline_spikes[trial_num])
                x_stimulus = np.array(stimulus_spikes[trial_num])
                plt.plot(x_baseline, np.repeat(i, len(x_baseline)),'o', ms =2, 
                        color=colors[int(self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices][trial_num])])
                plt.plot(x_stimulus, np.repeat(i, len(x_stimulus)),'o', ms =2, 
                        color=colors[int(self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices][trial_num])])
                
            plt.plot(timeline.to_numpy()[self.trials_map_indices][sorted_index], range(len(sorted_index)), 'ko', ms = 2)

        plt.axvline(0, color='k', linestyle='--', label="Stimulus Onset")
        plt.title(', '.join([f'{k}: {v}' for k, v in self.mask_info.items()]), fontsize = 10)
        plt.suptitle(f'Neuron ID: {neuron_ids or "All"} -- Session Date and Number: {self.session_date} & {self.session_no} -- Dominant Modality: {self.dominant_modality}')
        plt.xlabel('Time (s)')
        plt.ylabel('Trials')
        plt.yticks([])
        plt.xlim(xlim)
        plt.legend(handles=[Patch(color='blue', label='Left'), 
                            Patch(color='red', label='Right'),
                            Patch(color='gray', label='Center')],bbox_to_anchor=(1, 0.6), title="Choice Direction")

        plt.tight_layout()
        plt.show()
    
    def plot_spike_raster_table(self, neuron_ids = 1, first_move = True, sort_choice_loc = True, sort_response_time = True,
                            xlim = (-0.2, 0.5), figsize = (10,6)):
        masks = [{'aud_loc': 'r', 'vis_loc': 'l'},
                {'aud_loc': 'c', 'vis_loc': 'l'},
                {'aud_loc': 'l', 'vis_loc': 'l'},
                {'aud_loc': 'r', 'vis_loc': 'o'},
                {'aud_loc': 'c', 'vis_loc': 'o'},
                {'aud_loc': 'l', 'vis_loc': 'o'},
                {'aud_loc': 'r', 'vis_loc': 'r'},
                {'aud_loc': 'c', 'vis_loc': 'r'},
                {'aud_loc': 'l', 'vis_loc': 'r'}]

        fig, axs = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
        colors = ['gray','blue', 'red']

        for mask in masks:
            self.return_mask(**mask)
            ax = axs[masks.index(mask) % 3, masks.index(mask) // 3]
            if first_move:
                timeline = self.trials_formatted.timeline_firstMoveOn - self.trials_formatted.timeline_audPeriodOn
            else:
                timeline = self.trials_formatted.timeline_choiceMoveOn - self.trials_formatted.timeline_audPeriodOn
                
            if neuron_ids is None:
                neuron_ids = self.formatted_data.keys()
            elif type(neuron_ids) is int:
                neuron_ids = [neuron_ids]

            if not sort_choice_loc and not sort_response_time:
                sorted_index = self.trials_map_indices
            else:
                df = pd.DataFrame({'response_direction': self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices],
                                'timeline': timeline.to_numpy()[self.trials_map_indices]})
                if sort_choice_loc and sort_response_time:
                    df_sorted = df.sort_values(by=['response_direction', 'timeline'])
                    sorted_index = df_sorted.index
                elif sort_choice_loc:
                    df_sorted = df.sort_values(by=['response_direction'])
                    sorted_index = df_sorted.index
                else:
                    df_sorted = df.sort_values(by=['timeline'])
                    sorted_index = df_sorted.index

            for neuron_id in neuron_ids:
                baseline_spikes = self.formatted_data[neuron_id]['spikes']['baseline']
                stimulus_spikes = self.formatted_data[neuron_id]['spikes']['stimulus']
                
                
                for i in range(len(sorted_index)):
                    trial_num = sorted_index[i]
                    x_baseline = np.array(baseline_spikes[trial_num])
                    x_stimulus = np.array(stimulus_spikes[trial_num])
                    ax.plot(x_baseline, np.repeat(i, len(x_baseline)),'o', ms =2, 
                            color=colors[int(self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices][trial_num])])
                    ax.plot(x_stimulus, np.repeat(i, len(x_stimulus)),'o', ms =2, 
                            color=colors[int(self.trials_formatted.response_direction.to_numpy()[self.trials_map_indices][trial_num])])
                    
                ax.plot(timeline.to_numpy()[self.trials_map_indices][sorted_index], range(len(sorted_index)), 'ko', ms = 2)

            ax.axvline(0, color='k', linestyle='--', label="Stimulus Onset")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xlim)

        x_labels = ['Left', 'Off', 'Right']
        y_labels = ['Right', 'Center', 'Left']
        for i in range(3):
            axs[2, i].set_xlabel(f'{x_labels[i]}')  # x labels for the bottom row
            axs[i, 0].set_ylabel(f'{y_labels[i]}')  # y labels for the left column
        fig.text(0.5, 0.04, 'Visual stimulus', ha='center', fontsize=12)
        fig.text(0.06, 0.5, 'Auditory stimulus', va='center', rotation='vertical', fontsize=12)
        plt.suptitle(f'Neuron ID: {neuron_ids or "All"} -- Session Date and Number: {self.session_date} & {self.session_no} -- Dominant Modality: {self.dominant_modality}')
        plt.tight_layout(rect=[1, 0, 1, 0.95])
        plt.show()
    

    
    
    def test_firing_rate_change_stimulus(self, parametric_test = False, one_sided = False, up_regulated = False, 
                                         multiple_correction = True, mc_test = 'fdr_bh', mc_alpha = 0.05):
        """
        Calculate the change in firing rate for each neuron in the dataset and perform statistical tests to determine significance.
        
        Args:
        - stimulus_duration_interest: The duration of the stimulus period to consider for the analysis.
        - parametric_test: Whether to use a parametric test (True) or a non-parametric test (False).
        - one_sided: Whether to perform a one-sided test (True) or a two-sided test (False).
        - up_regulated: Whether to test for up-regulation (True) or down-regulation (False).
        - multiple_correction: Whether to apply multiple testing correction.
        - mc_test: The multiple correction test to use (e.g., 'bonferroni', 'fdr_bh').
        - mc_alpha: The alpha level for multiple testing correction.
        
        Returns:
        - p_vals: A dictionary where keys are neuron IDs and values are tuples of (p-value, test statistic, significance).
        - n_significant: The number of neurons with significant changes in firing rate.
        - n_significant_mc: The number of neurons with significant changes after multiple testing correction.
        """
        
        p_vals = test_fr_change_stimulus(self, parametric_test, one_sided, up_regulated, multiple_correction, mc_test, mc_alpha)
        self.p_vals = p_vals

    def find_significant_neurons(self, method: Literal['ccsp', 'csp', 'ttest'], reg_param: float = 1e-5,
                                parametric_test = True,one_sided = True, multiple_correction = True,
                                mc_test:Literal['bonferroni', 'sidak', 'holm', 'fdr_bh'] = 'bonferroni', 
                                mc_alpha = 0.05):
        if method == 'csp':
            variance_ratio, increased_neurons, decreased_neurons = calculate_csp(self)
        elif method == 'ccsp':
            variance_ratio, increased_neurons, decreased_neurons = calculate_ccsp(self, reg_param=reg_param)
        elif method == 'ttest':
            variance_ratio, increased_neurons, decreased_neurons = calculate_ttest(self, parametric_test=parametric_test,
                                                                                one_sided=one_sided, multiple_correction=multiple_correction,
                                                                                mc_test=mc_test, mc_alpha=mc_alpha)
        else:
            raise ValueError('Invalid method. Choose from "ccsp", "csp", "ttest"')
        return variance_ratio, increased_neurons, decreased_neurons