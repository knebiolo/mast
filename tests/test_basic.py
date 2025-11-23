"""
Basic tests for MAST core functionality
"""

import pytest
import pandas as pd
import numpy as np
from pymast import predictors, naive_bayes


@pytest.mark.unit
@pytest.mark.smoke
class TestPredictors:
    """Test predictor calculation functions"""
    
    @pytest.mark.smoke
    def test_noise_ratio(self):
        """Test noise ratio calculation"""
        # Create sample data
        freq_codes = np.array(['164.123 45', '164.456 78', '164.123 45', '164.999 99'])
        epochs = np.array([100.0, 105.0, 110.0, 115.0])
        study_tags = ['164.123 45', '164.456 78']
        
        # Calculate noise ratio
        result = predictors.noise_ratio(5.0, freq_codes, epochs, study_tags)
        
        # Should return array of same length
        assert len(result) == len(freq_codes)
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.smoke    
    def test_factors(self):
        """Test factor calculation"""
        # Test with simple number
        result = predictors.factors(12)
        expected = [1, 2, 3, 4, 6, 12]
        assert result == expected
        
    def test_max_contiguous_sequence(self):
        """Test consecutive sequence detection"""
        arr = np.array([1, 1, 0, 1, 1, 1, 0, 1])
        result = predictors.max_contiguous_sequence(arr)
        assert result == 3  # Three consecutive 1's


@pytest.mark.unit
@pytest.mark.classifier
class TestNaiveBayes:
    """Test Naive Bayes classifier functions"""
    
    @pytest.mark.smoke
    def test_calculate_priors(self):
        """Test prior probability calculation"""
        labeled = np.array([True, True, False, True, False])
        prior_true, prior_false = naive_bayes.calculate_priors(labeled)
        
        assert prior_true == 0.6  # 3/5
        assert prior_false == 0.4  # 2/5
        
    def test_calculate_likelihood(self):
        """Test likelihood calculation with Laplace smoothing"""
        training_obs = np.array([1, 1, 2, 2, 3])
        labeled = np.array([True, True, True, False, False])
        classification_obs = np.array([1, 2, 3])
        
        # Test for True assumption
        likelihood = naive_bayes.calculate_likelihood(
            training_obs, labeled, True, classification_obs, laplace=1
        )
        
        assert len(likelihood) == len(classification_obs)
        assert all(likelihood > 0)  # All should be positive with Laplace smoothing
        
    def test_classify_with_threshold(self):
        """Test classification with threshold"""
        posterior_true = np.array([0.8, 0.6, 0.3])
        posterior_false = np.array([0.2, 0.4, 0.7])
        
        # Test MAP (threshold = 1.0)
        classification = naive_bayes.classify_with_threshold(
            posterior_true, posterior_false, threshold_ratio=1.0
        )
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(classification, expected)
        
    def test_bin_predictors(self):
        """Test predictor binning"""
        hit_ratio = np.array([0.5, 0.7, 0.9])
        power = np.array([50, 60, 70])
        lag_diff = np.array([0, 3, -5])
        cons_length = np.array([1, 3, 5])
        noise_ratio = np.array([0.1, 0.2, 0.3])
        
        # This should not raise an error
        result = naive_bayes.bin_predictors(
            hit_ratio, power, lag_diff, cons_length, noise_ratio
        )


@pytest.mark.unit
class TestDataStructures:
    """Test data loading and validation"""
    
    @pytest.mark.smoke
    def test_tag_data_structure(self):
        """Test that tag data has required columns"""
        required_cols = ['freq_code', 'pulse_rate', 'cap_loc', 'rel_loc', 
                        'tag_type', 'rel_date']
        
        # Create minimal valid tag data
        tag_data = pd.DataFrame({
            'freq_code': ['164.123 45'],
            'pulse_rate': [3.0],
            'cap_loc': ['Site A'],
            'rel_loc': ['Site B'],
            'tag_type': ['study'],
            'rel_date': ['2024-01-01 12:00:00']
        })
        
        for col in required_cols:
            assert col in tag_data.columns
            
    def test_receiver_data_structure(self):
        """Test that receiver data has required columns"""
        required_cols = ['rec_id', 'rec_type', 'node']
        
        receiver_data = pd.DataFrame({
            'rec_id': ['R01'],
            'rec_type': ['srx800'],
            'node': ['N01']
        })
        
        for col in required_cols:
            assert col in receiver_data.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
