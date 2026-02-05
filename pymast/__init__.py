# Import the submodules first that have no dependencies
from .predictors import *
from .overlap_removal import *
#from .table_merge import *
from .formatter import *

# Import utilities
from .logger import setup_logging, logger
from .validation import (
    validate_tag_data,
    validate_receiver_data,
    validate_nodes_data,
    validate_project_dir,
    validate_file_exists,
    ValidationError
)

# Import the naive_bayes submodule next
from .naive_bayes import *

# Import the overlap removal submodule
from .overlap_removal import *

# Import the parsers submodule, which depends on predictor
from .parsers import *

# Finally, import the radio_project class, which depends on parsers, predictor, and naive_bayes
from .radio_project import *

# Version
__version__ = '1.0.5'

# Define what's available when using "from pymast import *"
__all__ = [
    'radio_project',
    'bout',
    'overlap_reduction',
    'fish_history',
    'setup_logging',
    'logger',
    'validate_tag_data',
    'validate_receiver_data',
    'validate_nodes_data',
    'validate_project_dir',
    'validate_file_exists',
    'ValidationError',
]



