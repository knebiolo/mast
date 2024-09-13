# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:52:20 2023

@author: KNebiolo
"""
import numba as nb
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)
np.set_printoptions(suppress=True, precision=10)


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

def max_contiguous_sequence(arr):
    # Finds the maximum number of consecutive 1's in an array
    return max(map(len, ''.join(map(str, arr)).split('0')))

def detection_history(epoch, pulse_rate, num_detects, num_channels, scan_time):
    shifts = np.arange(-num_detects, num_detects + 1)
    
    # Create shifted epochs for each detection window
    shifted_epochs = pd.DataFrame({f'Shift_{shift}': pd.Series(epoch).shift(shift) for shift in shifts * -1}).to_numpy()
    
    # Expand arrays for vectorized operations
    epoch_expanded = np.tile(epoch[:, None], (1, len(shifts)))
    scan_time_expanded = np.tile(scan_time[:, None], (1, len(shifts)))
    num_channels_expanded = np.tile(num_channels[:, None], (1, len(shifts)))
    
    # Compute expected epochs based on conditions
    expected_epoch = np.where(
        num_channels_expanded == 1, 
        epoch_expanded + shifts * pulse_rate,
        np.where(scan_time_expanded > 2 * pulse_rate, 
                 epoch_expanded + shifts * pulse_rate,
                 epoch_expanded + shifts * scan_time_expanded * num_channels_expanded)
    )

    # Adjust window size relative to pulse rate
    window_size = np.where(num_channels == 1, 
                           np.where(pulse_rate > 10, 1, pulse_rate),
                           scan_time / 2.)
    
    window_size_expanded = np.tile(window_size[:, None], (1, len(shifts)))
    
    # Compute detection history with reduced memory overhead
    lower_limits = expected_epoch - window_size_expanded
    upper_limits = expected_epoch + window_size_expanded
    
    # Initialize detection history
    detection_history = np.zeros_like(expected_epoch, dtype=np.int32)
    
    # Check if any values in each row of shifted_epochs fall within the corresponding windows
    detection_history = np.any(
        (shifted_epochs[:, :, None] >= lower_limits[:, None, :]) & 
        (shifted_epochs[:, :, None] <= upper_limits[:, None, :]),
        axis=1
    ).astype(np.int32)
    
    # Ensure the current epoch detection is marked
    detection_history[:, num_detects] = 1

    # Calculate hit ratio
    hit_ratio = detection_history.sum(axis=1) / detection_history.shape[1]

    # Consecutive detections (1 if consecutive, otherwise 0)
    cons_det = (detection_history[:, 1:-1].sum(axis=1) > 1).astype(np.int32)

    # Max contiguous sequence (find the maximum number of consecutive 1's)
    max_count = np.array([max_contiguous_sequence(hist) for hist in detection_history])

    return detection_history, hit_ratio, cons_det, max_count
    


                             
