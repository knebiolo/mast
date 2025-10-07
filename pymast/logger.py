"""
Logging configuration for MAST
"""
import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for MAST project.
    
    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file. If None, logs only to console.
    
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    
    Examples
    --------
    >>> from pymast.logger import setup_logging
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.info("Starting analysis...")
    """
    # Create logger
    logger = logging.getLogger('pymast')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger
logger = logging.getLogger('pymast')
