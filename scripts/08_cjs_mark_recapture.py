"""
Cormack-Jolly-Seber Mark Recapture Model Script
Creates input files for Program MARK to estimate survival and detection probabilities.

@author: KNebiolo
"""

# import modules
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('DEBUG: sys.executable=', sys.executable)
print('DEBUG: repo_root=', repo_root)
print('DEBUG: sys.path[0]=', sys.path[0])
from pymast.radio_project import radio_project
from pymast import formatter
import pandas as pd

# Set up project
project_dir = r"C:\Users\Kevin.Nebiolo\Desktop\Scotland KPN"
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

#%% Create a Cormack-Jolly-Seber Mark Recapture model

# What is the output directory?
output_ws = os.path.join(project_dir,'Output')
model_name = "scotland_cjs"

# What is the Node to Recapture Site relationship - use Python dictionary
# Map multiple receivers to single recapture sites for CJS model
receiver_to_recap = {'R01':'R01','R02':'R01',    # Windham
                     'R03':'R02',                 # Forebay
                     'R04':'R03',                 # Fish lift exit
                     'R05':'R04',                 # Submerged bypass
                     'R06':'R05','R07':'R05',     # Surface bypass
                     'R08':'R06','R09':'R06',     # Spillway
                     'R10':'R07',                 # Tailrace
                     'R11':'R08',                 # Fish lift entrance
                     'R12':'R09','R13':'R09',     # Downstream gate
                     'R14':'R10','R15':'R10'}     # Occum

# Step 1: Create CJS data class - we only need to feed it the directory and file name of input data
cjs = formatter.cjs_data_prep(receiver_to_recap, project, species = None, initial_recap_release = False)
print("Step 1 Completed, Data Class Finished")

# Step 2: Create input file for MARK
cjs.input_file(model_name, output_ws)
cjs.inp.to_csv(os.path.join(output_ws, model_name + '.csv'), index = False)

print("Step 2 Completed, MARK Input file created")
print("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")
