# Import the submodules first that have no dependencies
from .predictors import *
from .overlap_removal import *
from .table_merge import *
from .formatter import *

# Import the naive_bayes submodule next
from .naive_bayes import *

# Import the overlap removal submodule
from .overlap_removal import *

# Import the parsers submodule, which depends on predictor
from .parsers import *

# Finally, import the radio_project class, which depends on parsers, predictor, and naive_bayes
<<<<<<< Updated upstream:mast/__init__.py
from .radio_project import *
=======
from .radio_project import *

>>>>>>> Stashed changes:biotas_refactor/__init__.py
