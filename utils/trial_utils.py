import numpy as np

def format_trials(obj):
    ''' 
    Formats the trials data for a given session. Adds columns for visual and auditory locations, and condition ID.
    
    Args:
        obj: Object containing trials in obj.all_event_data.
    Returns:
        pd.DataFrame: The formatted trials data.
    '''
    # Filter out invalid trials if only_validTrial is True
    if obj.only_validTrial:
        trials = obj.all_event_data[obj.all_event_data.is_validTrial]
    else:
        trials = obj.all_event_data
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
    
    # Define choices for aud_loc
    aud_choices = ['l', 'r']

    # Create the new columns
    trials['vis_loc'] = np.select(vis_conditions, vis_choices, default='o')
    trials['aud_loc'] = np.select(aud_conditions, aud_choices, default='c')
    
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
    trials['cond_id'] = trials.apply(
        lambda row: condition_mapping.get((row['aud_loc'], row['vis_loc'])), axis=1)
    return trials

def select_trials(obj, **criteria):
    """
    Filters trials based on provided keyword criteria. 
    It returns the filtered trials and the original trial indices.
    
    Args:
        obj: Object containing trials in obj.trials_formatted.
        **criteria: Keyword arguments such as blank_trial, visual_trial, aud_stim_loc, choice, etc.
    
    Returns:
        tuple: (filtered_trials, original_trial_indices)
    """
    trials = obj.trials_formatted.copy()
    
    # Mapping of criteria to lambda functions for filtering
    conditions = {
        'blank_trial': lambda t, val: t.is_blankTrial if val else ~t.is_blankTrial,
        'visual_trial': lambda t, val: t.is_visualTrial if val else ~t.is_visualTrial,
        'auditory_trial': lambda t, val: t.is_auditoryTrial if val else ~t.is_auditoryTrial,
        'coherent_trial': lambda t, val: t.is_coherentTrial if val else ~t.is_coherentTrial,
        'conflict_trial': lambda t, val: t.is_conflictTrial if val else ~t.is_conflictTrial,
        'vis_stim_loc': lambda t, val: t.vis_loc == ('l' if val in ['left', 'l'] else 'r' if val in ['right', 'r'] else 'o'),
        'aud_stim_loc': lambda t, val: t.aud_loc == ('l' if val in ['left', 'l'] else 'r' if val in ['right', 'r'] else 'c'),
        'choice': lambda t, val: t.choice == (0 if val in ['left', 'l', 0] else 1 if val in ['right', 'r', 1] else -1)
    }
    
    # Apply filtering based on provided criteria
    for key, value in criteria.items():
        if key in conditions and value is not None:
            trials = trials[conditions[key](trials, value)]
    
    trials_index = np.array(trials.index)
    trials.reset_index(drop=True, inplace=True)
    return trials, trials_index