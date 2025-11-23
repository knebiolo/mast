# -*- coding: utf-8 -*-
"""
Predictor functions for radio telemetry classification and filtering.

This module provides statistical predictor functions used during Naive Bayes
classification to distinguish true fish detections from noise. Each predictor
calculates a feature that helps identify legitimate versus spurious detections.

Core Predictors
---------------
- **noise_ratio**: Ratio of miscoded to correctly-coded detections in time window
- **series_hit**: Whether detection is in-series with previous/next detection
- **detection_history**: Maximum contiguous sequence of expected detections
- **factors**: Prime factorization for pulse rate calculations

Classification Pipeline
-----------------------
These predictors are calculated during data import and used as features in the
Naive Bayes classifier. They help identify:
1. Miscoded tags from environmental noise
2. Out-of-series detections from spurious signals
3. Detection patterns inconsistent with known pulse rates

Typical Usage
-------------
>>> import pymast.predictors as predictors
>>> import numpy as np
>>> 
>>> # Calculate noise ratio for detections
>>> noise = predictors.noise_ratio(
...     duration=5.0,
...     freq_codes=freq_codes_array,
...     epochs=epochs_array,
...     study_tags=['166.380 7', '166.380 12']
... )
>>> 
>>> # Check detection history
>>> max_seq = predictors.detection_history(
...     epoch=epoch_array,
...     pulse_rate=pulse_rate_array,
...     num_detects=5,
...     num_channels=1,
...     scan_time=1.0
... )

Notes
-----
- Predictors assume VHF pulse-coded tags (frequency + code)
- Noise ratio uses 5-minute moving window by default
- Series hit checks for mortality rate changes (active vs expired tags)
- Detection history accounts for multi-channel scan patterns

See Also
--------
naive_bayes.train : Classifier training using these predictors
parsers : Data import where predictors are first calculated
"""

import numba as nb
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)
np.set_printoptions(suppress=True, precision=10)


def noise_ratio (duration, freq_codes,epochs,study_tags):
    """
    Calculate ratio of miscoded to total detections within moving time window.
    
    Identifies noise by comparing miscoded detections (freq_codes not in study_tags)
    to total detection count within specified duration. High noise ratio indicates
    environmental interference or receiver malfunction.
    
    Parameters
    ----------
    duration : float
        Moving window length in seconds (e.g., 300.0 for 5 minutes)
    freq_codes : array_like
        Array of detected frequency-code strings (e.g., ['166.380 7', '166.380 12'])
    epochs : array_like
        Array of detection timestamps (seconds since 1970-01-01)
    study_tags : list of str
        List of valid freq_code tags deployed in study
    
    Returns
    -------
    numpy.ndarray
        Array of noise ratios (float32) with same length as input arrays.
        Values range from 0 (no noise) to 1 (all noise).
    
    Notes
    -----
    - Bins epochs into duration-sized windows
    - Counts miscoded detections per bin (not in study_tags)
    - Calculates ratio: miscodes / total_detections
    - Ratio propagated to all detections in same bin
    
    Examples
    --------
    >>> import numpy as np
    >>> freq_codes = np.array(['166.380 7', '166.380 99', '166.380 7'])
    >>> epochs = np.array([1000.0, 1100.0, 1200.0])
    >>> study_tags = ['166.380 7', '166.380 12']
    >>> predictors.noise_ratio(300.0, freq_codes, epochs, study_tags)
    array([0.33333334, 0.33333334, 0.33333334], dtype=float32)
    
    See Also
    --------
    naive_bayes.train : Uses noise_ratio as classification feature
    """

    # function calculates the ratio of miscoded, pure noise detections, to matching frequency/code
    # detections within the duration specified.

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
    """
    Return all factors of integer n.
    
    Used to calculate least common multiplier for pulse rate calculations.
    Helps identify valid pulse intervals when multiple tags share similar rates.
    
    Parameters
    ----------
    n : int
        Integer to factorize
    
    Returns
    -------
    list of int
        All factors of n (including 1 and n)
    
    Examples
    --------
    >>> predictors.factors(12)
    [1, 2, 3, 4, 6, 12]
    
    Notes
    -----
    Simple brute-force factorization, not optimized for large numbers.
    See: http://stackoverflow.com/questions/16996217/prime-factorization-list
    """
    pList = []
    for i in np.arange(1, n + 1):
        if n % i == 0:
            pList.append(i)
    return pList


def series_hit (lags, pulse_rate, mort_rate, status,):
    """
    Check if detection lag matches expected pulse rate (in-series detection).
    
    Determines whether time difference to previous/next detection is consistent
    with tag's programmed pulse rate (active) or mortality rate (expired tag).
    
    Parameters
    ----------
    lags : array_like
        Time differences to previous detection (seconds)
    pulse_rate : array_like
        Programmed pulse rate for each tag (seconds)
    mort_rate : array_like
        Mortality pulse rate for each tag (seconds)
    status : array_like
        Tag status ('A' for active, other for expired/mortality)
    
    Returns
    -------
    numpy.ndarray
        Binary array: 1 if detection is in-series, 0 if out-of-series
    
    Notes
    -----
    - Active tags checked against pulse_rate
    - Expired tags checked against mort_rate
    - Uses modulo to check if lag is multiple of expected rate
    
    Examples
    --------
    >>> lags = np.array([5.0, 10.0, 7.5])
    >>> pulse_rate = np.array([5.0, 5.0, 5.0])
    >>> mort_rate = np.array([30.0, 30.0, 30.0])
    >>> status = np.array(['A', 'A', 'A'])
    >>> predictors.series_hit(lags, pulse_rate, mort_rate, status)
    array([1, 1, 0])
    
    See Also
    --------
    detection_history : More comprehensive in-series detection check
    """
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
    """
    Find maximum number of consecutive 1's in binary array.
    
    Helper function for detection_history to identify longest run of
    expected in-series detections.
    
    Parameters
    ----------
    arr : array_like
        Binary array (0s and 1s)
    
    Returns
    -------
    int
        Length of longest consecutive sequence of 1's
    
    Examples
    --------
    >>> arr = np.array([1, 1, 0, 1, 1, 1, 0, 1])
    >>> predictors.max_contiguous_sequence(arr)
    3
    """
    # Finds the maximum number of consecutive 1's in an array
    return max(map(len, ''.join(map(str, arr)).split('0')))


def detection_history(epoch, pulse_rate, num_detects, num_channels, scan_time):
    """
    Calculate maximum contiguous sequence of expected detections.
    
    Looks forward and backward in time to find longest run of detections
    that match expected pulse intervals. Accounts for multi-channel scanning
    patterns where pulse rate may exceed scan time.
    
    Parameters
    ----------
    epoch : array_like
        Detection timestamps (seconds since 1970-01-01)
    pulse_rate : array_like
        Programmed pulse rate for each tag (seconds)
    num_detects : int
        Number of detections to look forward/backward (window size)
    num_channels : array_like
        Number of receiver channels
    scan_time : array_like
        Scan duration per channel (seconds)
    
    Returns
    -------
    numpy.ndarray
        Array of maximum contiguous sequence lengths for each detection
    
    Notes
    -----
    - Creates detection window of size (2 * num_detects + 1)
    - Checks if adjacent detections match expected pulse intervals
    - Accounts for scan_time > pulse_rate (detection aliasing)
    - Uses vectorized operations for performance
    
    Examples
    --------
    >>> epochs = np.array([100, 105, 110, 115, 120])
    >>> pulse_rate = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    >>> num_channels = np.array([1, 1, 1, 1, 1])
    >>> scan_time = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> predictors.detection_history(epochs, pulse_rate, 2, num_channels, scan_time)
    array([5, 5, 5, 5, 5])
    
    See Also
    --------
    max_contiguous_sequence : Helper function for finding longest run
    series_hit : Simpler in-series detection check
    """
    shifts = np.arange(-num_detects, num_detects + 1)

    # Create shifted epochs for each detection window (NaNs for out-of-range shifts)
    shifted_df = pd.DataFrame({f'Shift_{s}': pd.Series(epoch).shift(s) for s in (-shifts)})
    shifted_epochs = shifted_df.to_numpy()

    # Expand arrays for vectorized operations
    m = len(shifts)
    epoch_expanded        = np.tile(epoch[:, None],        (1, m))
    scan_time_expanded    = np.tile(scan_time[:, None],    (1, m))
    num_channels_expanded = np.tile(num_channels[:, None], (1, m))

    # Expected epoch per shift
    expected_epoch = np.where(
        num_channels_expanded == 1,
        epoch_expanded + shifts * pulse_rate,
        np.where(
            scan_time_expanded > 2 * pulse_rate,
            epoch_expanded + shifts * pulse_rate,
            epoch_expanded + shifts * scan_time_expanded * num_channels_expanded
        )
    )

    # Window size relative to pulse rate
    window_size = np.where(
        num_channels == 1,
        np.where(pulse_rate > 10, 1, pulse_rate),
        scan_time / 2.0
    )
    window_size_expanded = np.tile(window_size[:, None], (1, m))

    # Elementwise window limits
    lower_limits = expected_epoch - window_size_expanded
    upper_limits = expected_epoch + window_size_expanded

    # *** FIX: elementwise compare, not all-to-all ***
    detection_history = (
        (shifted_epochs >= lower_limits) &
        (shifted_epochs <= upper_limits)
    ).astype(np.int32)

    # Ensure current epoch marked
    detection_history[:, num_detects] = 1

    # Metrics
    hit_ratio = detection_history.sum(axis=1) / detection_history.shape[1]
    cons_det  = (detection_history[:, 1:-1].sum(axis=1) > 1).astype(np.int32)

    # Max contiguous sequence of 1s (assumes you have max_contiguous_sequence)
    max_count = np.array([max_contiguous_sequence(hist) for hist in detection_history])

    return detection_history, hit_ratio, cons_det, max_count

    


                             
