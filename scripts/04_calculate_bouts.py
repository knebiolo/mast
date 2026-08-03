"""
Bout Calculation Script
Calculates presence bouts (continuous periods of detection) for each receiver.

@author: KNebiolo
"""

# import modules
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from pymast.radio_project import radio_project
import pymast
import pandas as pd

# Set up project
project_dir = r"C:\path\to\your\project"  # UPDATE THIS
db_name = 'Scotland_repacked'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# Create/load project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

#%% Calculate bouts
# Receiver to process
rec_id = 'R15'

# Create a bout object (DBSCAN runs during initialization)
bout_obj = pymast.bout(project, rec_id, eps_multiplier=5, lag_window=2)

# Write presence records
bout_obj.presence()

# Undo bouts if needed
# project.undo_bouts(rec_id)
