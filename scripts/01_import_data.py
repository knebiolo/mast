"""
Data Import Script
Imports raw telemetry data from receiver files into the project database.

@author: KNebiolo
"""

# import modules
import os
import sys
# Ensure the repository root (one level up from scripts/) is on sys.path so imports like
# `import pymast` work regardless of the absolute path or username. This is safer than a
# hard-coded path which can break on different machines or user accounts.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('DEBUG: sys.executable=', sys.executable)
print('DEBUG: repo_root=', repo_root)
print('DEBUG: sys.path[0]=', sys.path[0])
from pymast.radio_project import radio_project
import pandas as pd

# Set up project
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_12_04"
db_name = 'thompson_2025_v3'

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

#%% Import data
rec_id = 'R1699-3'
rec_type = 'PIT'
training_dir = os.path.join(project_dir,'Data','Training_Files')
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 1 #10.5        
channels = 1. #2
antenna_to_rec_dict = {1:'R0001',
                        2:'R0002',
                        3:'R0003',
                        4:'R0004',
                        5:'R0005'}


project.telem_data_import(rec_id,
                           rec_type,
                           training_dir,
                           db_dir,
                           scan_time,
                           channels,
                           None,
                           True)

# Undo import if needed
# project.undo_import(rec_id)
