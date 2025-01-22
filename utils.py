import pandas as pd
import numpy as np

from typing import Optional, Literal
from pinkrigs_tools.utils import ev_utils, spk_utils

def format_session(recordings, session_no = 0, expDef = 'multiSpaceWorld_checker_training', check_dominantModality=True):
    active_session  = recordings[recordings.expDef==expDef].iloc[session_no]

    session_date = active_session.expDate + '-' + str(active_session.expNum)
    ev = active_session.events._av_trials
    spikes = active_session.probe0.spikes
    clusters = active_session.probe0.clusters

    formatted_events = ev_utils.format_events(ev)
    formatted_cluster_data = spk_utils.format_cluster_data(clusters)
    
    if check_dominantModality:
        dominant_modality = determine_dominance(formatted_events)
    
    return formatted_events, formatted_cluster_data, spikes, dominant_modality, session_no, session_date

def determine_dominance_row(audL, feedback, choice):
    """
    Determines the dominant stimulus based on audL, feedback, and choice.

    Args:
        audL (int): 1 if audio is on the left, 0 if on the right.
        feedback (int): 1 if correct, -1 if incorrect.
        choice (int): 1 for right choice, -1 for left choice.

    Returns:
        str: 'aud' if audio is dominant, 'visual' otherwise.
    """
    return 'aud' if feedback * choice == (1 if audL == 0 else -1) else 'visual'

def determine_dominance(events: pd.DataFrame) -> Optional[str]:
    '''
    Determines the dominant stimulus based on audL, feedback, and choice.
    
    Args:
        events (pd.DataFrame): The events dataframe.
        
    Returns:
        str: 'aud' if audio is dominant, 'visual' otherwise.
    '''
    val_events=events[events.is_validTrial]
    conf_events = val_events[val_events.is_conflictTrial]
    conf_events = conf_events[conf_events.choice != 0]

    conf_events['dominant'] = conf_events.apply(lambda row: determine_dominance_row(row['audL'], row['feedback'], row['choice']), axis=1)

    if np.unique(conf_events.dominant).shape[0] > 1:
        print('There is a conflict')
    else:
        return np.unique(conf_events.dominant)[0]
    