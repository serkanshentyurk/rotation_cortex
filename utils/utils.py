import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from typing import Optional, Literal
from pinkrigs_tools.utils import ev_utils, spk_utils

def format_session(recordings, session_no = 0, expDef = 'multiSpaceWorld_checker_training'):
    '''
    Formats the data for a given session. It formats only 1 session.
    
    Args:
        recordings (pd.DataFrame): The recordings dataframe.
        session_no (int): The session number.
        expDef (str): The experiment definition.
        
        Returns:
            tuple: A tuple containing the formatted events, formatted cluster 1, formatted cluster 2, spikes 1, spikes 2, session number, session date, and animal ID.
    '''
    active_session = recordings[recordings.expDef==expDef].iloc[session_no]

    session_date = active_session.expDate + '-' + str(active_session.expNum)
    ev = active_session.events._av_trials
    spikes_0 = active_session.probe0.spikes
    clusters_0 = active_session.probe0.clusters
    spikes_1 = active_session.probe1.spikes
    clusters_1 = active_session.probe1.clusters

    formatted_events = ev_utils.format_events(ev)
    formatted_cluster_1 = spk_utils.format_cluster_data(clusters_0)
    formatted_cluster_2 = spk_utils.format_cluster_data(clusters_1)
    
    animal_id = recordings.expFolder[0][-18:-13]

    return formatted_events, formatted_cluster_1, formatted_cluster_2, spikes_0, spikes_1, session_no, session_date, animal_id

def merge_spikes(spikes_0, spikes_1):
    """Merge spikes from two probes into a single dictionary."""
    spikes = defaultdict(list)
    for d in (spikes_0, spikes_1):
        for key, value in d.items():
            spikes[key] = np.concatenate([spikes[key], value])
    return dict(spikes)
    
# def determine_dominance_row(audL, feedback, choice):
#     """
#     Determines the dominant stimulus based on audL, feedback, and choice.

#     Args:
#         audL (int): 1 if audio is on the left, 0 if on the right.
#         feedback (int): 1 if correct, -1 if incorrect.
#         choice (int): 1 for right choice, -1 for left choice.

#     Returns:
#         str: 'aud' if audio is dominant, 'visual' otherwise.
#     """
#     return 'aud' if feedback * choice == (1 if audL == 0 else -1) else 'visual'

# def determine_dominance(events: pd.DataFrame) -> Optional[str]:
#     '''
#     Determines the dominant stimulus based on audL, feedback, and choice.
    
#     Args:
#         events (pd.DataFrame): The events dataframe.
        
#     Returns:
#         str: 'aud' if audio is dominant, 'visual' otherwise.
#     '''
#     conf_events = events[events.is_conflictTrial]
#     conf_events = conf_events[conf_events.choice != 0]

#     conf_events['dominant'] = conf_events.apply(lambda row: determine_dominance_row(row['audL'], row['feedback'], row['choice']), axis=1)
    

# def normalize_p_values(p_values, min_p=1e-5, max_p=0.05):
#     """Normalize p-values between 0 and 1 (low p -> 1, high p -> 0)."""
#     norm_p = (np.log10(p_values) - np.log10(max_p)) / (np.log10(min_p) - np.log10(max_p))
#     norm_p = np.clip(norm_p, 0, 1)  # Ensure values stay within 0-1
    
#     return norm_p
