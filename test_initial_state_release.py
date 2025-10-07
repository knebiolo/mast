"""
Test script for initial_state_release = True mode
This should model movement from release (state 0) through all states
"""

# import modules
import os
import sys
sys.path.append(r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01\mast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01"
db_name = 'thompson_2025'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# create a project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

#%% create Time to Event Model with initial_state_release = True
    
# what is the Node to State relationship - use Python dictionary
states = {'R1696':1,
          'R1699-1':2,
          'R1699-2':3,
          'R1698':4,
          'R1699-3':5,
          'R1695':5,
          'R0004':6,
          'R0005':6,
          'R0001':7,
          'R0002':7,
          'R0003':8}

print("=== TESTING initial_state_release = True ===")
print("This should model movement from release (state 0) through all states\n")

# Step 1, create time to event data class with initial_state_release = True
tte = formatter.time_to_event(states,
                              project,
                              initial_state_release = True,  # Changed to True
                              last_presence_time0 = False,
                              cap_loc = None,
                              rel_loc = None,
                              species = None,
                              rel_date = None,
                              recap_date = None)

# Step 2, format data - without covariates
tte.data_prep(project)
# Step 3, generate a summary
stats = tte.summary()

#Print off dataframes of the Movement Summary, State Table, Tailrace Table
# ensure Spyder prints every column
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

out = os.path.join(project_dir, "Output")
df_movement_summary = pd.read_csv(os.path.join(out, "movement_summary.csv"))
df_state_table      = pd.read_csv(os.path.join(out, "state_table.csv"))

# --- print to console ---
print("=== Movement Summary (initial_state_release = True) ===")
print(df_movement_summary, "\n")  

print("=== State Table (initial_state_release = True) ===")
print(df_state_table, "\n")

print("=== Fish Count Comparison ===")
print(f"Total unique fish in this model: {len(tte.recap_data.freq_code.unique())}")
print(f"Fish that started from release (state 0): {len(tte.recap_data[tte.recap_data.state == 0].freq_code.unique())}")