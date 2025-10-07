"""
Input validation utilities for MAST
"""
import pandas as pd
import os

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_tag_data(tag_data):
    """
    Validate master tag table has required columns and correct data types.
    
    Parameters
    ----------
    tag_data : pandas.DataFrame
        Master tag table to validate
    
    Raises
    ------
    ValidationError
        If required columns are missing or data types are incorrect
    
    Returns
    -------
    bool
        True if validation passes
    """
    required_columns = {
        'freq_code': 'object',
        'pulse_rate': 'float',
        'tag_type': 'object',
        'rel_date': 'datetime64',
        'cap_loc': 'object',
        'rel_loc': 'object'
    }
    
    # Check required columns exist
    missing_cols = set(required_columns.keys()) - set(tag_data.columns)
    if missing_cols:
        raise ValidationError(
            f"Tag data missing required columns: {', '.join(missing_cols)}. "
            f"See docs/API_REFERENCE.md for required schema."
        )
    
    # Check for duplicates in freq_code
    if tag_data['freq_code'].duplicated().any():
        duplicates = tag_data[tag_data['freq_code'].duplicated()]['freq_code'].values
        raise ValidationError(
            f"Duplicate freq_codes found in tag_data: {', '.join(duplicates[:5])}. "
            f"Each freq_code must be unique."
        )
    
    # Check tag_type values
    valid_tag_types = ['study', 'BEACON', 'TEST']
    invalid_types = set(tag_data['tag_type'].unique()) - set(valid_tag_types)
    if invalid_types:
        raise ValidationError(
            f"Invalid tag_type values found: {', '.join(invalid_types)}. "
            f"Valid values: {', '.join(valid_tag_types)}"
        )
    
    return True

def validate_receiver_data(receiver_data):
    """
    Validate master receiver table has required columns.
    
    Parameters
    ----------
    receiver_data : pandas.DataFrame
        Master receiver table to validate
    
    Raises
    ------
    ValidationError
        If required columns are missing
    
    Returns
    -------
    bool
        True if validation passes
    """
    required_columns = ['rec_id', 'rec_type', 'node']
    
    missing_cols = set(required_columns) - set(receiver_data.columns)
    if missing_cols:
        raise ValidationError(
            f"Receiver data missing required columns: {', '.join(missing_cols)}. "
            f"See docs/API_REFERENCE.md for required schema."
        )
    
    # Check for duplicate rec_id
    if receiver_data['rec_id'].duplicated().any():
        duplicates = receiver_data[receiver_data['rec_id'].duplicated()]['rec_id'].values
        raise ValidationError(
            f"Duplicate rec_id found in receiver_data: {', '.join(duplicates)}. "
            f"Each rec_id must be unique."
        )
    
    # Check receiver types
    valid_rec_types = ['srx600', 'srx800', 'srx1200', 'orion', 'ares', 'VR2']
    invalid_types = set(receiver_data['rec_type'].unique()) - set(valid_rec_types)
    if invalid_types:
        raise ValidationError(
            f"Invalid rec_type values found: {', '.join(invalid_types)}. "
            f"Valid values: {', '.join(valid_rec_types)}"
        )
    
    return True

def validate_nodes_data(nodes_data):
    """
    Validate network nodes table has required columns.
    
    Parameters
    ----------
    nodes_data : pandas.DataFrame
        Network nodes table to validate
    
    Raises
    ------
    ValidationError
        If required columns are missing
    
    Returns
    -------
    bool
        True if validation passes
    """
    if nodes_data is None:
        return True  # Nodes are optional
    
    required_columns = ['node', 'X', 'Y']
    
    missing_cols = set(required_columns) - set(nodes_data.columns)
    if missing_cols:
        raise ValidationError(
            f"Nodes data missing required columns: {', '.join(missing_cols)}. "
            f"See docs/API_REFERENCE.md for required schema."
        )
    
    # Check for duplicate nodes
    if nodes_data['node'].duplicated().any():
        duplicates = nodes_data[nodes_data['node'].duplicated()]['node'].values
        raise ValidationError(
            f"Duplicate node IDs found: {', '.join(duplicates)}. "
            f"Each node must be unique."
        )
    
    return True

def validate_project_dir(project_dir):
    """
    Validate project directory path.
    
    Parameters
    ----------
    project_dir : str
        Path to project directory
    
    Raises
    ------
    ValidationError
        If path contains spaces or special characters
    
    Returns
    -------
    bool
        True if validation passes
    """
    # Check for spaces (recommended against but not fatal)
    if ' ' in project_dir:
        import warnings
        warnings.warn(
            "Project directory path contains spaces. "
            "This may cause issues on some systems. "
            "Consider using underscores instead."
        )
    
    # Check path length (Windows limitation)
    if len(project_dir) > 200:
        raise ValidationError(
            f"Project directory path is too long ({len(project_dir)} characters). "
            f"Maximum recommended length: 200 characters."
        )
    
    return True

def validate_file_exists(file_path, file_description="File"):
    """
    Check if a file exists and is readable.
    
    Parameters
    ----------
    file_path : str
        Path to file
    file_description : str
        Description of file for error message
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    PermissionError
        If file exists but isn't readable
    
    Returns
    -------
    bool
        True if file exists and is readable
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_description} not found: {file_path}"
        )
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(
            f"{file_description} exists but is not readable: {file_path}"
        )
    
    return True
