"""
Debug script to trace exactly what happens during data_prep for state 0 transitions
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pandas as pd

# Set up project
project_dir = os.environ.get("PYMAST_DEBUG_PROJECT_DIR", r"C:\path\to\your\project")
if project_dir == r"C:\path\to\your\project":
    raise RuntimeError("Set PYMAST_DEBUG_PROJECT_DIR to run this debug script.")
db_name = 'thompson_2025'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

project = radio_project(project_dir, db_name, detection_count, duration, tag_data, receiver_data, nodes_data)

states = {'R1696':1, 'R1699-1':2, 'R1699-2':3, 'R1698':4, 'R1699-3':5, 'R1695':5, 'R0004':6, 'R0005':6, 'R0001':7, 'R0002':7, 'R0003':8}

print("=== STEP-BY-STEP DEBUG OF data_prep ===\n")

# Initialize TTE
tte = formatter.time_to_event(states, project, initial_state_release=True, 
                              last_presence_time0=False, cap_loc=None, rel_loc=None, 
                              species=None, rel_date=None, recap_date=None)

# Manually step through data_prep logic
tte.project = project
columns = ['freq_code','species', 'start_state', 'end_state', 'presence', 'time_stamp',
           'time_delta', 'first_obs', 'time_0', 'time_1', 'transition']

tte.master_state_table = pd.DataFrame()
tte.bucket_length = 15

# Sort data
tte.recap_data.sort_values(by=['freq_code', 'epoch'], ascending=True, inplace=True)

# Merge start times
tte.recap_data = tte.recap_data.merge(
    tte.start_times[['first_recapture']].reset_index(),
    on='freq_code',
    how='left')

print("After merging start_times:")
print(f"Records with state 0: {len(tte.recap_data[tte.recap_data.state == 0])}")

# Create prev_state
tte.recap_data['prev_state'] = tte.recap_data.groupby('freq_code')['state'].shift(1).fillna(0).astype(int)

print("After creating prev_state:")
print("Sample records showing prev_state and state:")
sample = tte.recap_data[['freq_code', 'epoch', 'state', 'prev_state', 'rec_id']].head(20)
print(sample)

# Find records where we transition TO state 0
transition_to_0 = tte.recap_data[tte.recap_data.state == 0]
print(f"\nRecords transitioning TO state 0: {len(transition_to_0)}")
print("These records and their prev_state:")
print(transition_to_0[['freq_code', 'epoch', 'state', 'prev_state', 'rec_id']].head())

# Check state changes
state_change_mask = tte.recap_data['state'] != tte.recap_data['prev_state']
state_changes_to_0 = tte.recap_data[state_change_mask & (tte.recap_data.state == 0)]
print(f"\nActual state CHANGES to state 0: {len(state_changes_to_0)}")
print("These should all be release records (first for each fish):")
print(state_changes_to_0[['freq_code', 'epoch', 'state', 'prev_state', 'rec_id']].head())

# Look for the problematic transitions (non-release records with state 0)
problematic = tte.recap_data[(tte.recap_data.state == 0) & (tte.recap_data.rec_id != 'rel')]
print(f"\n⚠️  Problematic state 0 records (not release): {len(problematic)}")
if len(problematic) > 0:
    print(problematic[['freq_code', 'epoch', 'state', 'prev_state', 'rec_id']].head())
