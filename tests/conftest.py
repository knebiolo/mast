"""
Shared pytest fixtures for PyMAST tests

This module provides reusable test fixtures for:
- Sample tag/receiver data
- Temporary project directories
- Mock databases
- Common test data patterns
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime


@pytest.fixture
def sample_tags():
    """
    Sample tag data for testing
    
    Returns:
        pd.DataFrame: Tag list with standard fields
    """
    return pd.DataFrame({
        'freq_code': ['164.123 45', '164.456 78', '164.789 12'],
        'pulse_rate': [3.0, 5.0, 2.5],
        'tag_type': ['study', 'study', 'BEACON'],
        'rel_date': pd.to_datetime([
            '2024-01-01 08:00:00',
            '2024-01-02 09:30:00',
            '2024-01-01 00:00:00'
        ]),
        'rel_loc': ['Site A', 'Site B', 'Beacon Site'],
        'cap_loc': ['Site A', 'Site A', 'Beacon Site'],
        'length': [450, 480, np.nan],
        'weight': [1200, 1500, np.nan],
        'species': ['Atlantic Salmon', 'Atlantic Salmon', np.nan],
        'sex': ['M', 'F', np.nan]
    })


@pytest.fixture
def sample_receivers():
    """
    Sample receiver data for testing
    
    Returns:
        pd.DataFrame: Receiver list with standard fields
    """
    return pd.DataFrame({
        'rec_id': ['R01', 'R02', 'R03', 'R04'],
        'rec_type': ['srx800', 'srx800', 'orion', 'srx1200'],
        'node': ['N01', 'N02', 'N03', 'N04'],
        'name': ['Downstream', 'Midstream', 'Upstream', 'Tailrace']
    })


@pytest.fixture
def sample_nodes():
    """
    Sample node data for testing
    
    Returns:
        pd.DataFrame: Node locations with coordinates
    """
    return pd.DataFrame({
        'node': ['N01', 'N02', 'N03', 'N04'],
        'reach': ['Lower', 'Middle', 'Upper', 'Dam'],
        'X': [100, 200, 300, 400],
        'Y': [100, 150, 200, 250]
    })


@pytest.fixture
def sample_detections():
    """
    Sample detection data for testing
    
    Returns:
        pd.DataFrame: Raw detection records
    """
    base_time = pd.Timestamp('2024-01-01 10:00:00')
    
    return pd.DataFrame({
        'freq_code': ['164.123 45'] * 10 + ['164.456 78'] * 5,
        'epoch': [base_time + pd.Timedelta(seconds=i*10) for i in range(15)],
        'rec_id': ['R01'] * 5 + ['R02'] * 5 + ['R01'] * 5,
        'power': np.random.randint(150, 255, 15),
        'channels': [1] * 15,
        'scan_time': [2.5] * 15
    })


@pytest.fixture
def temp_project(tmp_path):
    """
    Create temporary project directory structure
    
    Args:
        tmp_path: pytest's temporary directory fixture
        
    Returns:
        Path: Project directory path
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create subdirectories
    (project_dir / "input").mkdir()
    (project_dir / "output").mkdir()
    (project_dir / "figures").mkdir()
    
    return project_dir


@pytest.fixture
def sample_hdf5_db(tmp_path):
    """
    Create sample HDF5 database for testing
    
    Args:
        tmp_path: pytest's temporary directory fixture
        
    Returns:
        Path: Database file path
    """
    db_path = tmp_path / "test_db.h5"
    
    with h5py.File(db_path, 'w') as f:
        # Create sample datasets
        grp = f.create_group('raw_data')
        
        # Sample detection data
        data = np.array([
            (b'164.123 45', 1704106800.0, b'R01', 200, 1, 2.5),
            (b'164.123 45', 1704106810.0, b'R01', 205, 1, 2.5),
            (b'164.456 78', 1704106820.0, b'R02', 180, 1, 2.5),
        ], dtype=[
            ('freq_code', 'S20'),
            ('epoch', 'f8'),
            ('rec_id', 'S10'),
            ('power', 'i4'),
            ('channels', 'i4'),
            ('scan_time', 'f4')
        ])
        
        grp.create_dataset('detections', data=data)
    
    return db_path


@pytest.fixture
def mock_training_data():
    """
    Mock training data for classifier testing
    
    Returns:
        tuple: (X_train, y_train) where X is features, y is labels
    """
    np.random.seed(42)  # Reproducible
    
    n_true = 100
    n_false = 50
    
    # True detections (higher hit_ratio, lower noise)
    true_data = pd.DataFrame({
        'hit_ratio': np.random.uniform(0.7, 1.0, n_true),
        'cons_length': np.random.randint(2, 10, n_true),
        'noise_ratio': np.random.uniform(0.0, 0.3, n_true),
        'power': np.random.randint(180, 255, n_true),
        'lag_diff': np.random.uniform(0, 0.5, n_true)
    })
    
    # False detections (lower hit_ratio, higher noise)
    false_data = pd.DataFrame({
        'hit_ratio': np.random.uniform(0.0, 0.5, n_false),
        'cons_length': np.random.randint(1, 3, n_false),
        'noise_ratio': np.random.uniform(0.5, 1.0, n_false),
        'power': np.random.randint(150, 200, n_false),
        'lag_diff': np.random.uniform(0.5, 2.0, n_false)
    })
    
    X = pd.concat([true_data, false_data], ignore_index=True)
    y = np.array([True] * n_true + [False] * n_false)
    
    return X, y


@pytest.fixture
def sample_overlap_graph():
    """
    Sample parent-child graph for overlap testing
    
    Returns:
        dict: Adjacency list representation of graph
    """
    return {
        'N01': ['N02', 'N03'],  # N01 is parent of N02, N03
        'N02': ['N04'],          # N02 is parent of N04
        'N03': [],               # N03 is leaf
        'N04': []                # N04 is leaf
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Path to test data directory
    
    Returns:
        Path: Test data directory
    """
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_bout_intervals():
    """
    Sample inter-detection intervals for bout testing
    
    Returns:
        np.ndarray: Intervals in seconds
    """
    # Simulate three processes:
    # - Continuous presence: 1-60 seconds
    # - Edge milling: 1-30 minutes  
    # - Departure/return: > 1 hour
    
    continuous = np.random.exponential(10, 100)  # Mean 10 sec
    milling = np.random.exponential(600, 50)     # Mean 10 min
    departure = np.random.exponential(7200, 20)  # Mean 2 hours
    
    return np.concatenate([continuous, milling, departure])


# Performance benchmark fixtures

@pytest.fixture
def large_detection_dataset():
    """
    Large dataset for performance testing
    
    Returns:
        pd.DataFrame: 10,000 detection records
    """
    n = 10000
    base_time = pd.Timestamp('2024-01-01 00:00:00')
    
    return pd.DataFrame({
        'freq_code': np.random.choice(['164.123 45', '164.456 78'], n),
        'epoch': [base_time + pd.Timedelta(seconds=i) for i in range(n)],
        'rec_id': np.random.choice(['R01', 'R02', 'R03'], n),
        'power': np.random.randint(150, 255, n),
        'channels': np.ones(n, dtype=int),
        'scan_time': np.full(n, 2.5)
    })


# Parameterized test data

@pytest.fixture(params=['srx600', 'srx800', 'srx1200', 'orion', 'ares'])
def receiver_type(request):
    """Parameterized receiver types for testing parsers"""
    return request.param


@pytest.fixture(params=[1, 2, 4])
def channel_count(request):
    """Parameterized channel counts"""
    return request.param


# Cleanup and validation

@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """
    Automatically cleanup temporary files after tests
    
    This fixture runs before and after each test
    """
    yield
    # Cleanup code runs after test
    # tmp_path automatically cleaned by pytest


@pytest.fixture
def assert_dataframe_equal():
    """
    Helper fixture for DataFrame comparison
    
    Returns:
        function: Assertion function for DataFrames
    """
    def _assert_equal(df1, df2, **kwargs):
        """Compare DataFrames with useful error messages"""
        pd.testing.assert_frame_equal(
            df1, df2,
            check_dtype=True,
            check_index_type=True,
            **kwargs
        )
    return _assert_equal
