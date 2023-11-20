# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:50:12 2023

@author: KNebiolo
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
    'bin numerical predictors for classification'
    # define bins for analysis
    
    # hit ratio bins
    hit_ratio_bins =np.linspace(0,1.0,11)

    # plot signal power histograms by detection class
    min_power = power.min()//5 * 5
    max_power = power.max()//5 * 5
    try:
        power_bins =np.arange(min_power,max_power+20,10)
    except:
        print ('fuck')

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