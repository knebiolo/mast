"""
Time-to-Event Model Script
Creates competing risks time-to-event data and generates movement summaries.

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

#%% Create Time to Event Model
    
# What is the Node to State relationship - use Python dictionary
upstream_states = {'R15':1,'R14':1,    # occum
                 'R13':2,'R12':2,    # downstream gate
                 'R10':3,            # tailrace
                 'R11':4,            # fish lift entrance
                 'R08':5,'R09':5,    # spillway
                 'R06':6,'R07':6,    # surface bypass
                 'R05':7,            # submerged bypass
                 'R04':8,            # fish lift exit
                 'R03':9,            # forebay
                 'R01':10,'R02':10}  # windham

downstream_states = {'R15':1,'R14':1,# occum
                 'R13':2,'R12':2,    # downstream gate
                 'R10':2,            # tailrace
                 'R11':2,            # fish lift entrance
                 'R08':3,'R09':3,    # spillway
                 'R06':6,'R07':6,    # surface bypass
                 'R05':7,            # submerged bypass
                 'R04':9,            # fish lift exit
                 'R03':9,            # forebay
                 'R01':10,'R02':10}  # windham
                                   
# Step 1: Create time to event data class 
tte = formatter.time_to_event(upstream_states,
                              project,
                              initial_state_release = True,
                              last_presence_time0 = False,
                              cap_loc = None,
                              rel_loc = 'windham',
                              species = None,
                              rel_date = None,
                              recap_date = None)

# Optional: Filter state transitions using adjacency matrix
# upstream_adjacency_filter = [(9, 1),(9, 2),(9, 3),(9, 5),(9, 8),(9, 9),(9, 6),
#                               (8, 1),(8, 2),(8, 3),(8, 5),(8, 9),(8, 8),(8, 6),
#                                (6, 1),(6, 6),(6, 5),
#                                 (1, 8),(1, 9),(2, 9),(2, 8),(3, 8),(3, 9),(0, 8),(0, 9),(5, 9)]

# downstream_adjacency_filter = [(1, 6),(1, 7),(1, 8),(1, 9),
#                                (9, 1),(8, 1),(9, 2),(9, 3),
#                                (2, 9),(2, 7),(2, 6),(2, 10),(2, 3),
#                                (3, 2),(3, 9)]

# Step 2: Format data - without covariates
tte.data_prep(project)#, adjacency_filter = downstream_adjacency_filter)

# Step 3: Generate a summary
stats = tte.summary()

# Print dataframes of the Movement Summary, State Table, Recaptures
# Ensure Spyder prints every column
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

out = os.path.join(project_dir, "Output")
df_movement_summary = pd.read_csv(os.path.join(out, "movement_summary.csv"))
df_state_table      = pd.read_csv(os.path.join(out, "state_table.csv"))
df_recaptures       = pd.read_csv(os.path.join(out, "recaptures.csv"))

# Print to console
print("=== Movement Summary ===")
print(df_movement_summary, "\n")  

print("=== State Table ===")
print(df_state_table, "\n")

print("=== Recaptures (first 5 rows) ===")
print(df_recaptures.head(5), "\n")
