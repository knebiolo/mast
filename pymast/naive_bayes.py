# -*- coding: utf-8 -*-
"""
Naive Bayes classifier for distinguishing true fish detections from noise.

This module implements a custom Naive Bayes classifier tailored for radio telemetry
data. It uses multiple predictor variables (signal power, lag differences, noise ratio,
etc.) to classify detections as true fish signals versus environmental noise or
spurious transmissions.

Classification Workflow
-----------------------
1. **Training**: Calculate priors and likelihoods from hand-labeled data
2. **Testing**: Apply trained classifier to unlabeled detections
3. **Binning**: Discretize continuous predictors for probability calculations
4. **Posterior**: Combine priors and likelihoods via Bayes' theorem
5. **Threshold**: Classify based on posterior ratio (adjustable threshold)

Predictor Variables
-------------------
- **hit_ratio**: Proportion of detections matching expected pulse intervals
- **power**: Signal strength (dB or raw power)
- **lag_diff**: Variability in time between consecutive detections
- **cons_length**: Maximum contiguous sequence of expected detections
- **noise_ratio**: Ratio of miscoded to total detections in time window

Typical Usage
-------------
>>> import pymast.naive_bayes as nb
>>> 
>>> # Calculate priors from labeled training data
>>> priors = nb.calculate_priors(labeled_truth_array)
>>> 
>>> # Calculate likelihoods for each predictor
>>> likelihood_true = nb.calculate_likelihood(
...     training_obs=training_power,
...     labeled_array=labeled_truth,
...     assumption=True,
...     classification_obs=test_power,
...     laplace=1
... )
>>> 
>>> # Calculate posterior and classify
>>> posterior_true = nb.calculate_posterior(
...     priors, evidence, likelihoods_dict, assumption=True
... )
>>> classifications = nb.classify_with_threshold(
...     posterior_true, posterior_false, threshold_ratio=1.0
... )

Notes
-----
- Uses Laplace smoothing (add-one) to handle unseen predictor values
- Predictors are binned into discrete categories before classification
- Threshold ratio allows precision/recall tradeoff (default: 1.0 = MAP)
- Assumes conditional independence between predictors (Naive Bayes assumption)

See Also
--------
predictors : Calculation of predictor variables
radio_project : Project management and data storage
"""

import numpy as np

def calculate_priors(labeled_array):
    """
    Calculate the prior probabilities for a binary classification problem.

    Parameters:
    labeled_array: A numpy array of boolean values (True/False) representing the truth value of detections.

    Returns:
    A tuple containing the prior probabilities (prior_true, prior_false).
    """
    total_count = labeled_array.size
    true_count = np.sum(labeled_array)
    false_count = total_count - true_count

    prior_true = true_count / total_count
    prior_false = false_count / total_count

    return (prior_true, prior_false)

def calculate_likelihood(training_obs, labeled_array, assumption, classification_obs, laplace=1):
    """
    Calculate the likelihood of each value in the classification dataset given a 
    truth value (True or False), based on training data with Laplace smoothing.

    Parameters:
    training_obs: A numpy array of observed values (binned data) from training data.
    labeled_array: A numpy array of boolean values (True/False) from training data, 
    representing the truth value.
    assumption: The truth value (True or False) for which to calculate the likelihood.
    classification_obs: A numpy array of observed values (binned data) from 
    classification dataset.
    laplace: The smoothing parameter (default is 1 for add-one smoothing).

    Returns:
    A 1D numpy array containing the likelihood of each observed value in the 
    lassification dataset given the assumption.
    """
    # Filter the training observation array based on the assumption
    filtered_obs = training_obs[labeled_array == assumption]

    # Count occurrences of each value in the filtered observation array
    unique_values, counts = np.unique(filtered_obs, return_counts=True)

    # Get the total number of unique values in the training observations
    total_unique_values = len(np.unique(training_obs))

    # Calculate the likelihood for each unique value with Laplace smoothing
    likelihoods = {val: (count + laplace) / (len(filtered_obs) + laplace * total_unique_values) for val, count in zip(unique_values, counts)}

    # Default likelihood for unseen values after smoothing
    default_likelihood = laplace / (len(filtered_obs) + laplace * total_unique_values)

    # Map each observation in the classification dataset to its corresponding likelihood
    likelihood_array = np.array([likelihoods.get(obs, default_likelihood) for obs in classification_obs])

    return likelihood_array

def calculate_evidence(observation_arrays):
    """
    Calculate the evidence for a Naive Bayes classifier with multiple predictors.

    Parameters:
    *observation_arrays: variable number of arrays, each representing binned data for a predictor.

    Returns:
    A 1D numpy array containing the evidence for each record.
    """

    # Calculate the marginal probabilities for each bin in each predictor
    marginal_probs = [np.bincount(obs_array) / len(obs_array) for obs_array in observation_arrays]

    # Calculate the evidence for each record
    evidence = np.ones(len(observation_arrays[0]))
    for obs_array, probs in zip(observation_arrays, marginal_probs):
        evidence *= probs[obs_array.astype(int)]

    return evidence

def calculate_posterior(priors, evidence, likelihoods_dict, assumption):
    """
    Calculate the posterior probabilities given observations.

    Parameters:
    priors: A 1D numpy array containing the prior probabilities for True and False.
    evidence: A 1D numpy array containing the evidence for each instance.
    likelihoods_dict: A nested dictionary of the structure {assumption:{predictor:likelihood array}}.
    assumption: The truth value (True or False) for which to calculate the posterior.

    Returns:
    A 1D numpy array containing the posterior probabilities for each instance.
    """
    # Initialize the posterior array with the prior probabilities
    posterior = np.full_like(evidence, priors[assumption])

    # Multiply by the likelihoods for each predictor
    for predictor, likelihood_array in likelihoods_dict[assumption].items():
        posterior *= likelihood_array

    # Divide by the evidence
    posterior /= evidence

    return posterior

def classify_with_threshold(posterior_true, posterior_false, threshold_ratio=1.0):
    """
    Classify instances based on posterior probabilities with an adjustable threshold.

    Parameters:
    posterior_true: A numpy array containing the posterior probabilities of being True.
    posterior_false: A numpy array containing the posterior probabilities of being False.
    threshold_ratio: The ratio of posterior_true to posterior_false required to 
                     classify an instance as True.
                     Default is 1.0, which corresponds to the standard MAP approach.

    Returns:
    A numpy array of classifications (True or False).
    """
    # Calculate the ratio of posterior_true to posterior_false
    ratio = posterior_true / posterior_false

    # Classify as True where the ratio exceeds the threshold, otherwise False
    classification = ratio >= threshold_ratio

    return classification

def bin_predictors(hit_ratio, power, lag_diff, cons_length, noise_ratio):
    """
    Bin continuous predictor variables into discrete categories for Naive Bayes.
    
    Converts continuous predictor values into discrete bins for probability
    calculations. Binning allows Naive Bayes to estimate likelihoods from
    limited training data.
    
    Parameters
    ----------
    hit_ratio : array_like
        Proportion of detections matching expected pulse intervals (0.0 to 1.0)
    power : array_like
        Signal power values (dB or raw)
    lag_diff : array_like
        Differences in lag times between consecutive detections (seconds)
    cons_length : array_like
        Maximum contiguous sequence of expected detections (1 to 11)
    noise_ratio : array_like
        Ratio of miscoded to total detections (0.0 to 1.0)
    
    Returns
    -------
    tuple of numpy.ndarray
        (hit_ratio_count, power_count, lag_count, con_len_count, noise_count)
        Each array contains bin indices for corresponding input values
    
    Notes
    -----
    Binning strategies:
    - hit_ratio: 11 bins from 0.0 to 1.0 (0.1 increments)
    - power: 10 dB bins from min to max (rounded to nearest 5 dB)
    - lag_diff: 20-second bins from -100 to 110 seconds
    - cons_length: 1-unit bins from 1 to 12
    - noise_ratio: 0.1 increment bins from 0.0 to 1.0
    
    Examples
    --------
    >>> hit_ratio = np.array([0.25, 0.75, 0.95])
    >>> power = np.array([100, 120, 140])
    >>> lag_diff = np.array([-10, 0, 10])
    >>> cons_length = np.array([3, 5, 8])
    >>> noise_ratio = np.array([0.05, 0.15, 0.25])
    >>> nb.bin_predictors(hit_ratio, power, lag_diff, cons_length, noise_ratio)
    (array([3, 8, 10]), array([1, 3, 5]), array([5, 6, 6]), array([3, 5, 8]), array([1, 2, 3]))
    
    See Also
    --------
    calculate_likelihood : Uses binned data for probability calculations
    """
    # bin numerical predictors for classification
    # define bins for analysis
    
    # hit ratio bins
    hit_ratio_bins =np.linspace(0,1.0,11)

    # plot signal power histograms by detection class
    min_power = power.min()//5 * 5
    max_power = power.max()//5 * 5
    try:
        power_bins = np.arange(min_power, max_power + 20, 10)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Error creating power bins: {e}. Check that power values are valid."
        ) from e

    # Lag Back Differences - how steady are detection lags?
    lag_bins =np.arange(-100,110,20)

    # Consecutive Record Length
    con_length_bins =np.arange(1,12,1)

    # Noise Ratio
    noise_bins =np.arange(0,1.1,0.1)
    
    # use np.digitize to bind
    hit_ratio_count = np.digitize(hit_ratio,  hit_ratio_bins)
    power_count = np.digitize(power,power_bins)
    lag_count = np.digitize(lag_diff, lag_bins)
    con_len_count = np.digitize(cons_length, con_length_bins)
    noise_count = np.digitize(noise_ratio, noise_bins)
    
    return hit_ratio_count, power_count, lag_count, con_len_count, noise_count
