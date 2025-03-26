import numpy as np
import scipy.stats as sstats


def shuffle_each_column(a):
    # shuffle each column independently
    # from: https://stackoverflow.com/questions/49426584/shuffle-independently-within-column-of-numpy-array
    idx = np.random.rand(*a.shape).argsort(0)
    out = a[idx, np.arange(a.shape[1])]

    return out


def cal_mann_whit_numerator(x, y, shuffle_matrix=None):
    """
    Calculates the numerator of the Mann-Whitney U test
    This is one sided and calculates numerator corresponding to U_1,
    corresponding to vector x
    Based on code in:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/stats/stats.py#L6334-L6429
    https://github.com/nsteinme/steinmetz-et-al-2019/blob/master/ccCP/mannWhitneyUshuf.m
    Note this is one sided (we don't take the smaller value)

    Parameters
    -------------

    """

    x = np.asarray(x)
    y = np.asarray(y)

    num_x = len(x)

    # ranked = sstats.rankdata(np.concatenate((x, y)))
    # T = tiecorrect(ranked)

    # this is equivalent to tiedrank in matlab
    ranked = sstats.rankdata(np.concatenate((x, y)))

    # print(np.shape(ranked))

    if shuffle_matrix is None:
        ranked_x = ranked[0:num_x]
        ranked_x_sum = np.sum(ranked_x)
    else:
        ranked_x = ranked[shuffle_matrix]
        ranked_x_sum = np.sum(ranked_x, axis=0)

    # if ranked_x == 0:
    #     print('All values are equal, cannot do test')
    #     mann_whit_numerator_x = None
    # else:
    mann_whit_numerator_x = ranked_x_sum - num_x * (num_x + 1) / 2

    return mann_whit_numerator_x


def cal_combined_conditions_choice_prob_numpy(cell_fr,
                                        stim_cond_id_per_trial,
                                        response_per_trial,
                                        num_shuffle=2000,
                                        verbose=True,
                                        cond_1_val=0,
                                        cond_2_val=1,
                                        choice_cond_dict={'left': 0, 'right': 1},
                                        conditions = [
                                            {'aud_stim_loc': 'r', 'vis_stim_loc': 'l'},
                                            {'aud_stim_loc': 'r', 'vis_stim_loc': 'o'},
                                            {'aud_stim_loc': 'r', 'vis_stim_loc': 'r'},
                                            {'aud_stim_loc': 'c', 'vis_stim_loc': 'l'},
                                            {'aud_stim_loc': 'c', 'vis_stim_loc': 'o'},
                                            {'aud_stim_loc': 'c', 'vis_stim_loc': 'r'},
                                            {'aud_stim_loc': 'l', 'vis_stim_loc': 'l'},
                                            {'aud_stim_loc': 'l', 'vis_stim_loc': 'o'},
                                            {'aud_stim_loc': 'l', 'vis_stim_loc': 'r'}
                                        ]):
    """
    Calculates the combined conditions choice probability for a single neuron
    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    for stim_cond_idx in np.unique(stim_cond_id_per_trial):

        cond_1_trials_to_get = np.where(
            (stim_cond_id_per_trial == stim_cond_idx) &
            (response_per_trial == cond_1_val)
        )[0]

        cond_2_trials_to_get = np.where(
            (stim_cond_id_per_trial == stim_cond_idx) &
            (response_per_trial == cond_2_val)
        )[0]

        choid_cond_1_data = cell_fr[cond_1_trials_to_get]
        choid_cond_2_data = cell_fr[cond_2_trials_to_get]

        num_choice_cond_1_ds_trials = len(choid_cond_1_data)
        num_choice_cond_2_ds_trials = len(choid_cond_2_data)

        if (num_choice_cond_1_ds_trials == 0) or (num_choice_cond_2_ds_trials == 0):
            if verbose:
                if conditions is not None:
                    print(f'No trials for one of the choices: {conditions[stim_cond_idx]}, skipping.')
                else:
                    print('No trials for one of the choices, skipping.')
            continue

        n_x_and_y = num_choice_cond_1_ds_trials + num_choice_cond_2_ds_trials
        n_x = num_choice_cond_1_ds_trials

        # generate shuffle matrix
        ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
        # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
        # check shuffle and original matrix are different


        # make the first column the original order
        # TODO: somehow assert fails if the line belowis placed
        # before the assert, which is weird...
        shuffle_matrix[:, 0] = np.arange(n_x)

        # check sum of each column is the same (ie. shuffling is within each column)
        column_sum = np.sum(ordered_matrix_shuffled, axis=0)
        assert np.all(column_sum == column_sum[0])

        # u_stat is equivalent to the 'n' in the matlab code
        u_stat_numerator = cal_mann_whit_numerator(
            x=choid_cond_1_data,
            y=choid_cond_2_data,
            shuffle_matrix=shuffle_matrix
        )

        total_possible_comparisons = num_choice_cond_1_ds_trials * num_choice_cond_2_ds_trials

        numerator_total += u_stat_numerator
        denominator_total += total_possible_comparisons

    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    # choice_probability = numerator_total / denominator_total

    # This is basically equivalent to calculating the percentile
    # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
    # choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
    # p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)

    return choice_probability, p_of_choice_probability

def cal_combined_conditions_choice_stim_numpy(cell_fr,
                                        aud_cond_per_trial,
                                        vis_cond_per_trial, 
                                        response_per_trial,
                                        unique_vis_cond,
                                        unique_aud_cond,
                                        unique_choice_cond,
                                        num_shuffle=2000,
                                        verbose=True,
                                        test_type='visLeftRight', # 'visLeftRight' or 'audLeftRight'
                                       ):

    """
    Calculates the combined conditions stimulus probability for a single neuron.

    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    # Select which stimulus condition to "marganlise out" by comparing the firing rate
    # by controlling for the other stimulus condition.
    if test_type == 'visLeftRight':
        marginal_stim_cond = unique_aud_cond
        cond_1_val = unique_vis_cond[0]
        cond_2_val = unique_vis_cond[1]
    elif test_type == 'audLeftRight':
        marginal_stim_cond = unique_vis_cond
        cond_1_val = unique_aud_cond[0]
        cond_2_val = unique_aud_cond[1]

    for other_stim_cond in marginal_stim_cond:

        for choice_cond in unique_choice_cond:

            if test_type == 'visLeftRight':

                cond_1_trial_idx = np.where(
                    (vis_cond_per_trial == cond_1_val) &
                    (aud_cond_per_trial == other_stim_cond) &
                    (response_per_trial == choice_cond)
                )[0]

                cond_2_trial_idx = np.where(
                    (vis_cond_per_trial == cond_2_val) &
                    (aud_cond_per_trial == other_stim_cond) &
                    (response_per_trial == choice_cond)
                )[0]

            elif test_type == 'audLeftRight':

                cond_1_trial_idx = np.where(
                    (vis_cond_per_trial == other_stim_cond) &
                    (aud_cond_per_trial == cond_1_val) &
                    (response_per_trial == choice_cond)
                )[0]

                cond_2_trial_idx = np.where(
                    (vis_cond_per_trial == other_stim_cond) &
                    (aud_cond_per_trial == cond_2_val) &
                    (response_per_trial == choice_cond)
                )[0]

            cond_1_fr = cell_fr[cond_1_trial_idx]
            cond_2_fr = cell_fr[cond_2_trial_idx]

            num_cond_1_ds_trials = len(cond_1_trial_idx)
            num_cond_2_ds_trials = len(cond_2_trial_idx)

            if (num_cond_1_ds_trials == 0) or (num_cond_2_ds_trials == 0):
                if verbose:
                    print('No trials for one of the stimulus conditions, skipping.')
                continue

            n_x_and_y = num_cond_1_ds_trials + num_cond_2_ds_trials
            n_x = num_cond_1_ds_trials

            # generate shuffle matrix
            ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
            # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
            ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
            shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
            # check shuffle and original matrix are different

            # make the first column the original order
            # TODO: somehow assert fails if the line below is placed
            # before the assert, which is weird...
            shuffle_matrix[:, 0] = np.arange(n_x)

            # check sum of each column is the same (ie. shuffling is within each column)
            column_sum = np.sum(ordered_matrix_shuffled, axis=0)
            assert np.all(column_sum == column_sum[0])

            # u_stat is equivalent to the 'n' in the matlab code
            u_stat_numerator = cal_mann_whit_numerator(
                x=cond_1_fr,
                y=cond_2_fr,
                shuffle_matrix=shuffle_matrix
            )

            total_possible_comparisons = num_cond_1_ds_trials * num_cond_2_ds_trials

            numerator_total += u_stat_numerator
            denominator_total += total_possible_comparisons


    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    return choice_probability, p_of_choice_probability
		

def calculate_cccp_ccsp(obj, 
                        until_movement = False, 
                        range_data=[0, 0.2], 
                        start_before_movement = 0.2,
                        delta_before_movement = 0.05):
	p_of_choice_probability = []
	aud_p_of_stim_probability = []
	vis_p_of_stim_probability = []
	trial_choice = obj.trials_formatted['choice'].values
	cccp_neuron_ids = []

	for key in obj.bombcell_keys:
		neuron = obj.formatted_data[key]
		spike_counts = []
		spike_counts_until_movement=[]
		if neuron == {}:
			p_of_choice_probability.append(np.nan)
			aud_p_of_stim_probability.append(np.nan)
			vis_p_of_stim_probability.append(np.nan)
			print(f"Skipping neuron {key} -- no spikes")
			continue
		cccp_neuron_ids.append(key)
		for trial_idx in range(len(neuron['spikes']['baseline'])):
			spike_times_relative = np.array(neuron['spikes']['baseline'][trial_idx] + neuron['spikes']['stimulus'][trial_idx])
			if until_movement:
				time_start = 0
				time_end = obj.spikes_formatted['start_time']['first_move'][trial_idx] - delta_before_movement
				if np.isnan(time_end):
					time_end = 0.2
     
				if time_end < 0:
					time_end = 0
					n_spike_per_trial = 0
				else:
					spike_times_relative_movement = spike_times_relative[(spike_times_relative >= time_end-start_before_movement) & (spike_times_relative <= time_end)]
					n_spike_per_trial = len(spike_times_relative_movement)/time_end
				spike_counts_until_movement.append(n_spike_per_trial)

			spike_times_relative = spike_times_relative[(spike_times_relative >= range_data[0]) & (spike_times_relative <= range_data[1])]
			n_spike_per_trial = len(spike_times_relative)
			spike_counts.append(n_spike_per_trial)
		spike_counts = np.array(spike_counts)
		if until_movement:
			spike_counts_until_movement = np.array(spike_counts_until_movement)
			counts = [spike_counts, spike_counts_until_movement]
		else:
			counts = [spike_counts, spike_counts]
  
		_, p_of_choice_prob = cal_combined_conditions_choice_prob_numpy(counts[0],
																			obj.trials_formatted['cond_id'],
																			trial_choice,
																			verbose = False)
  
		_, vis_p_of_stim_prob = cal_combined_conditions_choice_stim_numpy(counts[1],
																		obj.trials_formatted['aud_loc'].values,
																		obj.trials_formatted['vis_loc'].values,
																		trial_choice,
																		unique_vis_cond = ['l', 'r'],
																		unique_aud_cond = ['l', 'r'],
																		unique_choice_cond = [-1,0,1],
																		verbose=False,
																		test_type='visLeftRight')
		_, aud_p_of_stim_prob = cal_combined_conditions_choice_stim_numpy(counts[1],
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

	p_of_choice_probability = np.array(p_of_choice_probability)
	aud_p_of_stim_probability = np.array(aud_p_of_stim_probability)
	vis_p_of_stim_probability = np.array(vis_p_of_stim_probability)
 
	return cccp_neuron_ids, p_of_choice_probability, aud_p_of_stim_probability, vis_p_of_stim_probability


def return_ccxp_count_freq(ccxp):
	ccxp_values = np.array(ccxp) 
	ccxp_up = ccxp_values > 0.975
	ccxp_down = ccxp_values < 0.025
	ccxp_up_count = np.sum(ccxp_up, axis=1)
	ccxp_up_freq = ccxp_up_count/ccxp_up.shape[1]
	ccxp_down_count = np.sum(ccxp_down, axis=1)
	ccxp_down_freq = ccxp_down_count/ccxp_down.shape[1]
	return ccxp_values, ccxp_up_count, ccxp_up_freq, ccxp_down_count, ccxp_down_freq
