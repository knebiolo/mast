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

def calculate_likelihood(observation_array, labeled_array, assumption):
    """
    Calculate the likelihood of observations given a truth value (True or False).

    Parameters:
    observation_array: A numpy array of observed values (binned data).
    labeled_array: A numpy array of boolean values (True/False) representing the truth value.
    assumption: The truth value (True or False) for which to calculate the likelihood.

    Returns:
    A 1D numpy array containing the likelihood of each observed value given the assumption.
    """
    # Filter the observation array based on the assumption
    filtered_obs = observation_array[labeled_array == assumption]

    # Count occurrences of each value in the filtered observation array
    unique_values, counts = np.unique(filtered_obs, return_counts=True)

    # Calculate the likelihood for each unique value
    likelihoods = dict(zip(unique_values, counts / counts.sum()))

    # Map each observation to its corresponding likelihood
    likelihood_array = np.array([likelihoods.get(obs, 0) for obs in observation_array])

    return likelihood_array

def calculate_evidence(observation_arrays):
    """
    Calculate the evidence for a Naive Bayes classifier with multiple predictors.

    Parameters:
    *observation_arrays: variable number of arrays, each representing binned data for a predictor.

    Returns:
    A 1D numpy array containing the evidence for each record.
    """
    if len(observation_arrays) > 7:
        raise ValueError("This function supports up to 7 predictors.")

    # Calculate the marginal probabilities for each bin in each predictor
    marginal_probs = [np.bincount(obs_array) / len(obs_array) for obs_array in observation_arrays]

    # Calculate the evidence for each record
    evidence = np.ones(len(observation_arrays[0]))
    for obs_array, probs in zip(observation_arrays, marginal_probs):
        evidence *= probs[obs_array]

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
    posterior = np.full_like(evidence, prior[assumption])

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
    threshold_ratio: The ratio of posterior_true to posterior_false required to classify an instance as True.
                     Default is 1.0, which corresponds to the standard MAP approach.

    Returns:
    A numpy array of classifications (True or False).
    """
    # Calculate the ratio of posterior_true to posterior_false
    ratio = posterior_true / posterior_false

    # Classify as True where the ratio exceeds the threshold, otherwise False
    classification = ratio >= threshold_ratio

    return classification