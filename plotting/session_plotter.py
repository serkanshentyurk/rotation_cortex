import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd

from plotting.helper import *

class SessionPlotter:
    def __init__(self, session_data):
        """
        Initialize the SessionPlotter with a SessionData instance.
        
        Args:
            session_data: An instance of SessionData that contains all the processed data.
        """
        self.sd = session_data

    def plot_psth(self, figsize=(10, 6), xlim=None, ylim=None, filter_info=None, color_based_on='vis', only_average=True, neuron_only=None):
        sd = self.sd
        # Determine how many filters to apply; default is one (no filtering)
        if filter_info is None:
            n_filter = 1
            filter_info = [None]
        else:
            n_filter = len(filter_info)
        
        n_freq = sd.interest_data['firing_rate'].shape[2]
        fig, axs = plt.subplots(n_filter, 1, figsize=figsize)
        if n_filter == 1:
            axs = [axs]
        
        for i in range(n_filter):
            ax = axs[i]
            # Apply filtering if provided; otherwise, select all trials.
            if filter_info[i] is not None:
                sd.select_data_interest(filter_info[i])
                plot_title = filter_info[i]
            else:
                sd.select_data_interest()
                plot_title = 'All Trials'
            
            # Compute key time markers
            move_time = np.nanmean(sd.interest_data['start_time']['first_move'])
            configure_axes(ax, title='PSTH Plot', xlabel='Time (s)', ylabel='Firing rate (Hz)', xlim=xlim, ylim=ylim, vlines=[0, move_time])
            
            # Determine indices based on color grouping
            if color_based_on is not None:
                if color_based_on in ['visual', 'vis', 'v']:
                    l_index = sd.interest_data['trials'].vis_loc == 'l'
                    r_index = sd.interest_data['trials'].vis_loc == 'r'
                    c_index = sd.interest_data['trials'].vis_loc == 'o'
                    legend_title = 'Visual Stimulus'
                elif color_based_on in ['audio', 'aud', 'a']:
                    l_index = sd.interest_data['trials'].aud_loc == 'l'
                    r_index = sd.interest_data['trials'].aud_loc == 'r'
                    c_index = sd.interest_data['trials'].aud_loc == 'c'
                    legend_title = 'Auditory Stimulus'
                elif color_based_on in ['choice', 'ch', 'c']:
                    l_index = sd.interest_data['trials'].choice == -1
                    r_index = sd.interest_data['trials'].choice == 1
                    c_index = sd.interest_data['trials'].choice == 0
                    legend_title = 'Choice'
                
                # Plot individual trials if desired
                if not only_average:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1).reshape(n_freq, -1),
                            color='blue', label='Left', alpha=0.5)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1).reshape(n_freq, -1),
                            color='red', label='Right', alpha=0.5)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1).reshape(n_freq, -1),
                            color='green', label='Center', alpha=0.5)
                # Plot the mean firing rates
                if neuron_only is None:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1).mean(axis=0),
                            color='blue', label='Left', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1).mean(axis=0),
                            color='red', label='Right', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1).mean(axis=0),
                            color='green', label='Center', alpha=1)
                else:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1)[neuron_only],
                            color='blue', label='Left', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1)[neuron_only],
                            color='red', label='Right', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1)[neuron_only],
                            color='green', label='Center', alpha=1)
                    
                legend_elements = [
                    Line2D([0], [0], color='blue', lw=2, label='Left'),
                    Line2D([0], [0], color='red', lw=2, label='Right'),
                    Line2D([0], [0], color='green', lw=2, label='Center')
                ]
                ax.legend(handles=legend_elements, title=legend_title, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_title(f'Firing Rates - {plot_title} (Colored by {legend_title})')
            else:
                # If no color grouping, simply plot the overall average
                if not only_average:
                    ax.plot(sd.times, sd.interest_data['firing_rate'].mean(axis=1).reshape(n_freq, -1),
                            color='gray', label='All neurons', alpha=0.5)
                ax.plot(sd.times, sd.interest_data['firing_rate'].mean(axis=1).mean(axis=0),
                        color='black', label='All neurons', alpha=1)
                ax.set_title(f'Firing Rates - {plot_title}')
                
        plt.suptitle(f"PSTH - Session: {sd.session_date} & {sd.session_no}")
        plt.tight_layout()
        plt.show()
        
    def plot_psth_grid(self, only_average=True, xlim=(-0.2, 0.5), ylim=None, only_neuron=None):
        """
        Plot a 3x3 grid of PSTH plots for different conditions.
        
        Args:
            only_average (bool): If True, plot only the average firing rate; if False, include individual trial traces.
            xlim (tuple): X-axis limits.
            ylim (tuple): Y-axis limits.
            only_neuron (int or None): If provided, plot data for the specified neuron index only.
        """
        sd = self.sd  # Alias for the SessionData instance.
        n_freq = sd.interest_data['firing_rate'].shape[2]
        
        # Create a 3x3 grid of subplots.
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Loop through each grid cell.
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                # Use the condition for the current subplot (assumed defined in sd.conditions).
                condition = sd.conditions[i * 3 + j]
                sd.select_data_interest(**condition)
                
                # Define indices based on a 'choice' column from the trials.
                l_index = sd.interest_data['trials'].choice == 0
                r_index = sd.interest_data['trials'].choice == 1
                c_index = sd.interest_data['trials'].choice == -1

                # Plot individual trial traces if requested.
                if not only_average:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1).reshape(n_freq, -1),
                            color='blue', label='Left', alpha=0.5)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1).reshape(n_freq, -1),
                            color='red', label='Right', alpha=0.5)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1).reshape(n_freq, -1),
                            color='green', label='Center', alpha=0.5)
                
                # Plot the mean firing rates.
                if only_neuron is not None:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1)[only_neuron],
                            color='blue', label='Left', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1)[only_neuron],
                            color='red', label='Right', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1)[only_neuron],
                            color='green', label='Center', alpha=1)
                else:
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, l_index].mean(axis=1).mean(axis=0),
                            color='blue', label='Left', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, r_index].mean(axis=1).mean(axis=0),
                            color='red', label='Right', alpha=1)
                    ax.plot(sd.times, sd.interest_data['firing_rate'][:, c_index].mean(axis=1).mean(axis=0),
                            color='green', label='Center', alpha=1)
                
                move_time = np.nanmean(sd.interest_data['start_time']['first_move'])
                # Configure the axes using your helper function.
                configure_axes(ax, title='',
                            xlabel=sd.x_y_labels[i * 3 + j][0],
                            ylabel=sd.x_y_labels[i * 3 + j][1],
                            xlim=xlim, ylim=ylim, vlines=[0, move_time])
        
        # Set a title for the entire figure.
        if only_neuron is not None:
            plt.suptitle(
                f'Firing Rate Change of Neuron: {int(np.array(sd.bombcell_keys)[only_neuron])} '
                # f'Probe: {int(sd.all_cluster_data.iloc[np.array(sd.bombcell_keys)[only_neuron]].probe)} - '
                f'of {sd.animal_id} on {sd.session_date[:-2]}\n'
                f'CCCP:{np.format_float_positional(sd.cccp_values[only_neuron], precision=4)} - '
                f'CCSP-Aud:{np.format_float_positional(sd.ccsp_aud_values[only_neuron], precision=4)} - '
                f'CCSP-Vis:{np.format_float_positional(sd.ccsp_vis_values[only_neuron], precision=4)}'
            , fontsize=14)
        else:
            plt.suptitle(f'Animal: {sd.animal_id} - Date: {sd.session_date}')
        

        legend_patches = [
            patches.Patch(color='blue', label='Left Choice'),
            patches.Patch(color='green', label='No Choice'),
            patches.Patch(color='red', label='Right Choice')
        ]
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()


    def plot_raster(self, neuron_idx=0, first_move=True, sort_choice_loc=True, sort_response_time=True,
                    xlim=(-0.2, 0.5), figsize=(10, 6), ax=None, x_y_labels=None):
        sd = self.sd
        # Choose timeline based on first or choice movement
        timeline = sd.interest_data['start_time']['first_move'] if first_move else sd.interest_data['start_time']['choice_move']
        timeline = timeline.to_numpy()
        neuron_id = sd.bombcell_keys[neuron_idx]
        
        # Sorting logic using response direction and timeline
        if not sort_choice_loc and not sort_response_time:
            sorted_index = np.arange(sd.interest_data['neuron_ids'].shape[0])
        else:
            df = pd.DataFrame({
                'response_direction': sd.interest_data['trials']['response_direction'],
                'timeline': timeline
            })
            if sort_choice_loc and sort_response_time:
                df_sorted = df.sort_values(by=['response_direction', 'timeline'])
            elif sort_choice_loc:
                df_sorted = df.sort_values(by=['response_direction'])
            elif sort_response_time:
                df_sorted = df.sort_values(by=['timeline'])
            sorted_index = df_sorted.index
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlim)
        if x_y_labels is None:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial number')
            ax.set_title('Raster Plot')
        else:
            ax.set_xlabel(x_y_labels[0])
            ax.set_ylabel(x_y_labels[1])
        ax.axvline(0, color='black', linestyle='--')
        
        colors = ['gray', 'blue', 'red']
        
        baseline_spikes = sd.formatted_data[neuron_id]['spikes']['baseline']
        stimulus_spikes = sd.formatted_data[neuron_id]['spikes']['stimulus']
        
        for trial_num in sorted_index:
            x_baseline = baseline_spikes[trial_num]
            x_stimulus = stimulus_spikes[trial_num]
            # Color based on response direction from the sorted DataFrame
            
            ax.plot(x_baseline, np.repeat(trial_num, len(x_baseline)), 'o', ms=2,
                    color=colors[int(df_sorted.response_direction.to_numpy()[trial_num])])
            ax.plot(x_stimulus, np.repeat(trial_num, len(x_stimulus)), 'o', ms=2,
                    color=colors[int(df_sorted.response_direction.to_numpy()[trial_num])])
            ax.plot(df_sorted.timeline.to_numpy()[trial_num], trial_num, 'o', ms=2, color='black')

    def plot_raster_grid(self, neuron_idx=0, first_move=True, sort_choice_loc=True, sort_response_time=True,
                         xlim=(-0.2, 0.5), figsize=(12, 12)):
        sd = self.sd
        fig, axes = plt.subplots(3, 3, figsize=figsize, sharey=True)
        for i in range(3):
            for j in range(3):
                mask = sd.conditions[i * 3 + j]
                sd.select_data_interest(**mask)
                self.plot_raster(neuron_idx=neuron_idx, first_move=first_move, sort_choice_loc=sort_choice_loc,
                                 sort_response_time=sort_response_time, xlim=xlim, figsize=figsize,
                                 ax=axes[i, j], x_y_labels=sd.x_y_labels[i * 3 + j])
        # probe_text = f'Probe: {np.int16(sd.all_cluster_data.iloc[np.array(sd.bombcell_keys)[neuron_ids]].probe)}'
        plt.suptitle(f'Raster Plot of Neuron: {int(np.array(sd.bombcell_keys)[neuron_idx])} of {sd.animal_id} on {sd.session_date[:-2]}\n'
                     f'CCCP:{np.format_float_positional(sd.cccp_values[neuron_idx], precision=4)} - CCSP-Aud:{np.format_float_positional(sd.ccsp_aud_values[neuron_idx], precision=4)} - CCSP-Vis:{np.format_float_positional(sd.ccsp_vis_values[neuron_idx], precision=4)}',
                     fontsize = 14)
        plt.tight_layout()
        plt.show()

    def plot_cccp_ccsp(self, sort_by='cccp', figsize=(12, 8)):
        sd = self.sd
        if sort_by == 'cccp':
            sort_indices = sd.cccp_values.argsort()
        elif sort_by == 'ccsp-vis':
            sort_indices = sd.ccsp_vis_values.argsort()
        elif sort_by == 'ccsp-aud':
            sort_indices = sd.ccsp_aud_values.argsort()
        elif sort_by is None:
            sort_indices = np.arange(len(sd.cccp_values))
        else:
            raise ValueError('sort_by should be either cccp, ccsp-vis, or ccsp-aud')
        
        y_transformed_cccp = np.where(sd.cccp_values[sort_indices] < 0.5, 
                         np.log10(sd.cccp_values[sort_indices] + 1e-10), 
                         -np.log10(1 - sd.cccp_values[sort_indices] + 1e-10))
        y_transformed_ccsp_aud = np.where(sd.ccsp_aud_values[sort_indices] < 0.5,
                                np.log10(sd.ccsp_aud_values[sort_indices] + 1e-10),
                                -np.log10(1 - sd.ccsp_aud_values[sort_indices] + 1e-10))
        y_transformed_ccsp_vis = np.where(sd.ccsp_vis_values[sort_indices] < 0.5,
                                np.log10(sd.ccsp_vis_values[sort_indices] + 1e-10),
                                -np.log10(1 - sd.ccsp_vis_values[sort_indices] + 1e-10))
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # First subplot: original p-values
        ax.axhline(np.log10(0.025 + 1e-10), color='r', linestyle='--')
        ax.axhline(-np.log10(0.025+1e-10), color='r', linestyle='--')
        
        cccp_label = '$n_{significant}^{CCCP}$ = '
        cccp_label = cccp_label + str(np.sum(sd.cccp_values < 0.025) + np.sum(sd.cccp_values > 1 - 0.025))
        ax.plot(y_transformed_cccp, 'o', ms=4, label=cccp_label)
        
        ccsp_aud_label = '$n_{significant}^{CCSP-Aud}$ = '
        ccsp_aud_label = ccsp_aud_label + str(np.sum(sd.ccsp_aud_values < 0.025) + np.sum(sd.ccsp_aud_values > 1 - 0.025))
        ax.plot(y_transformed_ccsp_aud, 'o', ms=4, label=ccsp_aud_label)
        
        ccsp_vis_label = '$n_{significant}^{CCSP-Vis}$ = '
        ccsp_vis_label = ccsp_vis_label + str(np.sum(sd.ccsp_vis_values < 0.025) + np.sum(sd.ccsp_vis_values > 1 - 0.025))
        ax.plot(y_transformed_ccsp_vis, 'o', ms=4, label = ccsp_vis_label)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('Neurons')
        ax.set_xticklabels([])
        ax.set_ylabel('$log_{10}$(CCXP)')
        n_significant = np.sum(np.min([sd.cccp_values, sd.ccsp_aud_values, sd.ccsp_vis_values], axis=0) < 0.025) + np.sum(np.max([sd.cccp_values, sd.ccsp_aud_values, sd.ccsp_vis_values], axis=0) > 0.975)
        title = '$n_{significant}^{Neurons}$ = ' + str(n_significant)
        ax.set_title(title, fontsize = 10)
        plt.suptitle(f'CCCP, CCSP-Aud, and CCSP-Vis values of neurons sorted by {sort_by}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_heatmap_trials(self, cmap='magma', center=None, vmin=-1, vmax=2, median=False, sort_based_on='cccp', figsize=(12, 10),
                            vline_col='green', hline_col='red', only_significant=False, bar_n_sample=5):
        sd = self.sd
        cmap = sns.color_palette(cmap, as_cmap=True)
        if sort_based_on == 'cccp':
            sort_data = sd.cccp_values
        elif sort_based_on == 'ccsp-vis':
            sort_data = sd.ccsp_vis_values
        elif sort_based_on == 'ccsp-aud':
            sort_data = sd.ccsp_aud_values
        else:
            raise ValueError('sort_based_on should be either cccp, ccsp-vis, or ccsp-aud')
        
        sort_indices = sort_data.argsort()
        down_limit = np.where(sort_data[sort_indices] > 0.025)[0][0]
        up_limit = np.where(sort_data[sort_indices] > 0.975)[0][0]
        if only_significant:
            sort_indices_significant = np.array(
                np.concatenate([np.arange(down_limit), np.arange(up_limit, len(sort_indices))]),
                dtype=int
            )
        sd.select_data_interest()
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                current_mask = sd.conditions[i * 3 + j]
                sd.select_data_interest(**current_mask)
                n_trial = sd.interest_data['firing_rate'].shape[1]
                if n_trial == 0:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    continue
                data_hm = sd.interest_data['firing_rate'].mean(axis=1)
                data_hm = data_hm[sort_indices]
                if only_significant:
                    data_hm = data_hm[sort_indices_significant]
                else:
                    ax.axhline(up_limit, color=hline_col, linewidth=1, linestyle='--')
                move_time = np.nanmedian(sd.interest_data['start_time']['first_move']) if median else np.nanmean(sd.interest_data['start_time']['first_move'])
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
                sns.heatmap(data_hm, ax=ax, xticklabels=False, yticklabels=False, cbar=False, center=center,
                            vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
                title_now = '$n_{trial}$ = ' + str(n_trial)
                configure_axes(ax, title=title_now, xlabel=sd.x_y_labels[i * 3 + j][0], ylabel=sd.x_y_labels[i * 3 + j][1],
                               vlines=[sd.baseline_times.shape[0], move_time * 100 + sd.baseline_times.shape[0]],
                               hlines=[down_limit])
                ax.set_xticks([sd.baseline_times.shape[0], move_time * 100 + sd.baseline_times.shape[0]])
                ax.set_xticklabels(['Stimulus', f'Move = {round(move_time, 3)}'], fontsize=8)
                
                # Add a bar for sample indication
                bar = patches.Rectangle((0, 0), 1, bar_n_sample, linewidth=2, edgecolor="black", facecolor="black")
                ax.add_patch(bar)
                ax.text(-1.5, bar_n_sample / 2, f"{bar_n_sample}\nSample", fontsize=5,
                        verticalalignment='center', horizontalalignment='center', color="black", rotation=90)
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=cbar_ax, label='Firing rate (Hz) - Fold Change')
        plt.suptitle(f'Fold Change of Firing Rate of Neurons in Different Conditions - Sorted by: {sort_based_on}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_heatmap_trials_choice(self, cmap='magma', vmin=-1, vmax=2, center=0, median=False, sort_based_on='cccp',
                                   vline_col='green', hline_col='red', only_significant=False, bar_n_sample=5, ignore_title = False):
        sd = self.sd
        cmap = sns.color_palette(cmap, as_cmap=True)
        if sort_based_on == 'cccp':
            sort_data = sd.cccp_values
        elif sort_based_on == 'ccsp-vis':
            sort_data = sd.ccsp_vis_values
        elif sort_based_on == 'ccsp-aud':
            sort_data = sd.ccsp_aud_values
        else:
            raise ValueError('sort_based_on should be either cccp, ccsp-vis, or ccsp-aud')
        
        sort_indices = sort_data.argsort()
        down_limit = np.where(sort_data[sort_indices] > 0.025)[0][0]
        up_limit = np.where(sort_data[sort_indices] > 0.975)[0][0]
        if only_significant:
            sort_indices_significant = np.array(
                np.concatenate([np.arange(down_limit), np.arange(up_limit, len(sort_indices))]),
                dtype=int
            )
        custom_order = []
        for row_group in range(0, 6, 2):
            for col in range(6):
                custom_order.append((row_group, col))
                custom_order.append((row_group + 1, col))
        choice = [None, 'left', 'n', 'right']
        fig, axs = plt.subplots(6, 6, figsize=(20, 20), sharex=False, sharey=True)
        for index, (r, c) in enumerate(custom_order):
            ax = axs[r, c]
            trial = index // 4
            choice_no = index % 4
            mask = sd.conditions[trial]
            mask_choice = mask.copy()
            mask_choice['choice'] = choice[choice_no]
            sd.select_data_interest(**mask_choice)
            n_trial = sd.interest_data['firing_rate'].shape[1]
            if choice_no == 0:
                title_now = '$Choice: All, n_{trial}$ = ' + str(n_trial)
                ax.set_title(title_now, fontsize=8)
            elif choice_no == 2:
                title_now = '$Choice: None, n_{trial}$ = ' + str(n_trial)
                ax.set_title(title_now, fontsize=8)
            else:
                title_now = f'$Choice: {choice[choice_no]}, n_{trial}$ = ' + str(n_trial)
                ax.set_title(title_now, fontsize=8)
            if n_trial == 0:
                continue
            data_hm = sd.interest_data['firing_rate'].mean(axis=1)
            data_hm = data_hm[sort_indices]
            if only_significant:
                data_hm = data_hm[sort_indices_significant]
            else:
                ax.axhline(up_limit, color=hline_col, linewidth=1, linestyle='--')
            move_time = np.nanmedian(sd.interest_data['start_time']['first_move']) if median else np.nanmean(sd.interest_data['start_time']['first_move'])
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            sns.heatmap(data_hm, ax=ax, xticklabels=False, yticklabels=False, cbar=False, vmin=vmin, vmax=vmax, cmap=cmap, center=center, norm=norm)
            ax.axhline(down_limit, color=hline_col, linewidth=0.5, linestyle='--')
            if mask_choice['choice'] != 'n':
                ax.set_xticks([sd.baseline_times.shape[0], move_time * 100 + sd.baseline_times.shape[0]])
                ax.set_xticklabels(['Stim.', f'Move = {round(move_time, 3)}'], fontsize=8)
                ax.axvline(move_time * 100 + sd.baseline_times.shape[0], color=vline_col, linewidth=1, linestyle='--')
            else:
                ax.set_xticks([sd.baseline_times.shape[0]])
                ax.set_xticklabels(['Stim.'], fontsize=8)
            ax.axvline(sd.baseline_times.shape[0], color=vline_col, linewidth=1)
            
            bar = patches.Rectangle((0, 0), 1, bar_n_sample, linewidth=2, edgecolor="black", facecolor="black")
            ax.add_patch(bar)
            ax.text(-1.5, bar_n_sample / 2, f"{bar_n_sample}\nSample", fontsize=5, verticalalignment='center', horizontalalignment='center', color="black", rotation=90)
        
        # for x_pos in [2, 4.0]:
        #     fig.add_artist(Line2D([x_pos/6, x_pos/6], [0, 0.95], transform=fig.transFigure, color="black", linewidth=2))
        # for y_pos in [2.05/6, 3.95/6]:
        #     fig.add_artist(Line2D([0.05, 0.95], [0.985 - y_pos, 0.985 - y_pos], transform=fig.transFigure, color="black", linewidth=2))
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=cbar_ax, label='Firing rate (Hz) - Fold Change')
        if not ignore_title:
            plt.suptitle(f'Fold Change of Firing Rate of Neurons in Different Conditions - Sorted by: {sort_based_on}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_ccxp(self, time_windows = None, start_time = -0.5, end_time = 0.5, step = 0.1, figsize = (10,10)):
        sd = self.sd
        values = sd.ccxp_window_values
        
        if values is None:
            sd.calculate_cccp_ccsp_window(start_time = start_time, end_time = end_time, step = step, plot = False)
            values = sd.ccxp_window_values
            
        if time_windows is None:
            time_windows = np.linspace(-0.2, 0.5, len(values[0][0]), endpoint=False)
        else:
            time_windows = np.array(time_windows)[:,0]
        step = time_windows[1] - time_windows[0]
            
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharey=True)
        for ax, value, title in zip(axs, 
                                [values[0], values[1], values[2]],
                                ['CCCP', 'CCSP Aud', 'CCSP Vis']):

            ax.bar(time_windows+step/2, value[2], width=step/2, label='CCXP > 0.975', color='#B40426')
            ax.bar(time_windows+step/2, value[4], width=step/2, bottom = value[2], label='CCXP < 0.025', color='#3B4CC0')
            ax.axvline(0, color = 'black', linestyle = '--')
            title_current = f'The Frequency of Significant {title} Values'
            ax.set_title(title_current, fontsize = 10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency')
        plt.suptitle('The Frequency of Significant CCXP Values at Differnet Windows', fontsize = 14)
        plt.tight_layout()
        plt.show()
        
    def plot_ccxp_change_heatmap(self, label = 'CCXP'):
        sd = self.sd
        if label == 'CCCP':
            vals = np.array(sd.ccxp_window_values[0][0]).T
        elif label in ['CCSP Aud', 'CCSP-Aud', 'ccsp_aud']:
            vals = np.array(sd.ccxp_window_values[1][0]).T
        elif label in ['CCSP Vis', 'CCSP-Vis', 'ccsp_vis']:
            vals = np.array(sd.ccxp_window_values[2][0]).T
        else:
            raise ValueError('label should be either CCCP, CCSP Aud, or CCSP Vis')
        vals[vals > 0.975] = 1
        vals[vals < 0.025] = -1
        vals[(vals <= 0.975) & (vals >= 0.025)] = 0
        vals_sorted = vals[np.argsort(vals[:,0])]
        plt.figure(figsize = (10,10))
        sns.heatmap(vals_sorted, cmap='coolwarm', 
                    xticklabels = [f'{i:.2f}' for i in np.linspace(-0.15, 0.55, len(vals[0]), endpoint=False)],
                    yticklabels = [],
                    cbar_kws={'label': label})
        plt.title(f'{label} Change Heatmap')
        plt.xlabel('Time (s)')
        plt.show()
        
    def plot_significant_psth_raster(self, idx = 0, cccp = True, lower = True, vis = True, plot_psth = True, plot_raster = True, psth_ylim = (-0.5,2)):
        sd = self.sd
        if cccp:
            title_holder = 'CCCP'
            if lower:
                title_holder += ' <0.025'
                idx_neurons = np.where(sd.cccp_values < 0.025)[0]
            else:
                title_holder += ' >0.975'
                idx_neurons = np.where(sd.cccp_values > 0.975)[0]
        elif vis:
            title_holder = 'CCSP VIS'
            if lower:
                title_holder += ' <0.025'
                idx_neurons = np.where(sd.ccsp_vis_values < 0.025)[0]
            else:
                title_holder
                idx_neurons = np.where(sd.ccsp_vis_values > 0.975)[0]
        else:	
            title_holder = 'CCSP AUD'
            if lower:
                title_holder += ' <0.025'
                idx_neurons = np.where(sd.ccsp_aud_values < 0.025)[0]
            else:
                title_holder += ' >0.975'
                idx_neurons = np.where(sd.ccsp_aud_values > 0.975)[0]
        idx_neuron = idx_neurons[idx]
        print(f'There are {len(idx_neurons)} significant neurons')
        if plot_psth:
            self.plot_psth_grid(only_average=True, only_neuron= idx_neuron, ylim = psth_ylim)
        if plot_raster:
            self.plot_raster_grid(neuron_idx = idx_neuron, first_move = True, sort_choice_loc=True, sort_response_time=True, xlim=(-0.2, 0.5), figsize=(12, 12))
        return idx_neurons, title_holder
    
    def plot_inspect_neuron(self, neuron_idxs, title_holder, idx = 0, single = True, all = True):
        if single:
            neuron_spike_counts = inspect_neuron_spikes_single(self, neuron_idx = neuron_idxs[idx], plot = True)
        if all:
            _ = inspect_neuron_spikes(self, neuron_idxs, title_holder = title_holder)