# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:52:20 2023

@author: KNebiolo
"""
import numpy as np
import pandas as pd

def noise_ratio (duration, freq_codes,epochs,study_tags):

    ''' function calculates the ratio of miscoded, pure noise detections, to matching frequency/code
    detections within the duration specified.

    In other words, what is the ratio of miscoded to correctly coded detections within the duration specified

    duration = moving window length in minutes
    data = current data file
    study_tags = list or list like object of study tags
    '''
    # identify miscodes
    miscode = np.isin(freq_codes, study_tags, invert = True)

    # bin everything into nearest 5 min time bin and count miscodes and total number of detections
    binned_epoch = epochs//duration
        
    # Now identify the number of unique freq-codes within each bin
    # Create a DataFrame from the arrays
    df = pd.DataFrame({'FreqCodes': freq_codes,'Epoch':epochs, 'Bins': binned_epoch, 'miscode': miscode})
    
    # Group by 'Bins' and count unique 'FreqCodes' in each bin
    miscodes_per_bin = df.groupby('Bins')['miscode'].sum()
    obs_per_bin = df.groupby('Bins')['miscode'].count()
    
    # Convert Series to DataFrame and reset index
    miscodes_per_bin_df = miscodes_per_bin.reset_index()
    obs_per_bin_df = obs_per_bin.reset_index()
    
    # Merge the aggregated data back to the original DataFrame
    # Assuming 'Bins' is the common column
    df = pd.merge(df, miscodes_per_bin_df, on='Bins', how='left', suffixes=('', '_miscodes_sum'))
    df = pd.merge(df, obs_per_bin_df, on='Bins', how='left', suffixes=('', '_obs_count'))
    
    df['noise_ratio'] = df.miscode_miscodes_sum / df.miscode_obs_count
    
    return df.noise_ratio.values.astype(np.float32)

def factors(n):

    ''''function to return primes used to quantify the least common multiplier
    see: http://stackoverflow.com/questions/16996217/prime-factorization-list for recipe'''
    pList = []
    for i in np.arange(1, n + 1):
        if n % i == 0:
            pList.append(i)
    return pList

def series_hit (lags, pulse_rate, mort_rate, status,):
    '''seriesHit is a function for returning whether or not a detection on a specific
    frequency/code is in series with the previous or next detection on that same
    frequency/code
    '''
    
    # determine if the lag is potentially in series with the correct pulse rate based on status
    series_hit = np.where(status == 'A', 
                          np.where(lags % pulse_rate == 0,
                                   1,
                                   0),
                          np.where(lags % mort_rate == 0,
                                   1,
                                   0)
                          )
    
def detection_history (epoch, pulse_rate, num_detects, num_channels, scan_time, channels, status = 'A'):
    '''
    Calculate detection history for multi-channel switching receivers.

    This function computes the detection history based on the epoch time, pulse rate, number of detections, 
    number of channels, scan time, and channels. It accounts for the sparseness of detection histories 
    in multi-channel switching receivers, acknowledging that bursts may not always align with scan windows.

    Parameters:
    - Epoch (array-like): Array of epoch times.
    - pulse_rate (int): Pulse rate of the tag.
    - num_detects (int): Number of detections to consider.
    - num_channels (int): Number of channels in the receiver.
    - scan_time (int): Time taken for one scan.
    - channels (int): Number of channels.
    - status (str, optional): Status of the detection. Defaults to 'A'.

    Returns:
    - det_hist_conc (list of str): Concatenated string representation of detection history.
    - det_hist (numpy.ndarray): Array representing the ratio of detections over the total number of detections.
    - cons_det (numpy.ndarray): Array indicating consecutive detections.
    - max_count (numpy.ndarray): Array of the longest contiguous sequence of detections for each epoch.

    Example:
    If the scan time is 2 seconds, there are two channels, and the Epoch is 100, the receiver would listen at:
    (-3) 87-89, (-2) 91-93, (-1) 95-97, (0) 100, (1) 103-105, (2) 107-109, (3) 111-113 seconds.

    A tag with a 3-second burst rate at an epoch of 100 would be heard at:
    100, 103, 106, 109, 112 - meaning at least 1 out of 5 bursts could be missed.

    Note:
    Detection histories from multi-channel switching receivers are expected to be sparse, as they are not always listening.
    This function helps in understanding and analyzing this sparseness.
    '''

    # create dictionaries that will hold the epoch shift and its lower and upper limits
    epoch_shift_dict = {}
    lower_limit_dict = {}
    upper_limit_dict = {}
    
    # create a dictionary with key a detection and the value a boolean array of length of Epoch array
    detection_history_dict = {}

    # build detection history ushing shifts
    if num_channels > 1:
        for i in np.arange(num_detects * -1 , num_detects + 1, 1):
            epoch_shift_dict[i] = np.round(pd.Series(epoch).shift(i * -1).to_numpy().astype(np.float32),6)
            lower_limit_dict[i] = np.where(scan_time > 2 * pulse_rate,
                                           epoch + (pulse_rate * i - 1),
                                           epoch + ((scan_time * channels * i) - 1))
            upper_limit_dict[i] = np.where(scan_time > 2 * pulse_rate, 
                                           epoch + (pulse_rate * i + 1),
                                           epoch + ((scan_time * channels * i) + 1))

    else:
        for i in np.arange(num_detects * -1 , num_detects + 1, 1):
            epoch_shift_dict[i] = pd.Series(epoch).shift(i * -1).to_numpy().astype(np.float32)
            lower_limit_dict[i] = epoch + (pulse_rate * i - 1)
            upper_limit_dict[i] = epoch + (pulse_rate * i + 1)

    for i in np.arange(num_detects * -1 , num_detects + 1, 1):
        if i == 0:
            detection_history_dict[i] = np.repeat('1',len(epoch))

        else:
            detection_history_dict[i] = np.where(np.logical_and(epoch_shift_dict[i] >= lower_limit_dict[i],
                                                                epoch_shift_dict[i] < upper_limit_dict[i]),
                                                 '1',
                                                 '0')
    
    # create an immediate detection history around the current record to assess consecutive detections
    cons_arr = np.column_stack((detection_history_dict[-1],detection_history_dict[0],detection_history_dict[1])).astype(np.float32)
    cons_arr_sums = np.sum(cons_arr, axis = 1)
    cons_det = np.where(cons_arr_sums > 1,1,0)

    # calculate hit ratio - return det_hist_conc, det_hist
    det_hist_length = np.arange(num_detects * -1 , num_detects + 1, 1).shape[0]
    det_hist_stack = np.column_stack([detection_history_dict[x] for x in np.arange(num_detects * -1, num_detects + 1, 1)])
    det_hist_conc = [''.join(row) for row in det_hist_stack]
    det_hist_float = det_hist_stack.astype(np.float32) 
    row_sums = np.sum(det_hist_float, axis = 1)
    det_hist = row_sums/det_hist_length

    # calculate consecutive record length
    max_count = np.zeros(epoch.shape)
    current_count = np.repeat(1,len(epoch.shape))
    for i in np.arange(num_detects * -1 , num_detects + 1, 1):
        current_col = detection_history_dict[i]
        current_count = np.where(current_col == '1', current_count + 1, 0)
        max_count = np.where(current_count > max_count,
                             current_count,
                             max_count) 
        
    return det_hist_conc, det_hist, cons_det, max_count

                             
