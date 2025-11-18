"""
Debug script to find where state 0 transitions are coming from
"""

import os
import sys
sys.path.append(r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01\mast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pandas as pd

# Set up project
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01"
db_name = 'thompson_2025'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

project = radio_project(project_dir, db_name, detection_count, duration, tag_data, receiver_data, nodes_data)

states = {'R1696':1, 'R1699-1':2, 'R1699-2':3, 'R1698':4, 'R1699-3':5, 'R1695':5, 'R0004':6, 'R0005':6, 'R0001':7, 'R0002':7, 'R0003':8}

print("=== DEBUGGING STATE 0 TRANSITIONS ===\n")

# Test full journey mode
tte = formatter.time_to_event(states, project, initial_state_release=True, 
                              last_presence_time0=False, cap_loc=None, rel_loc=None, 
                              species=None, rel_date=None, recap_date=None)

# Check the raw recap_data before data_prep
print("Raw recap_data after __init__:")
print(f"Total records: {len(tte.recap_data)}")
state_counts = tte.recap_data.groupby('state').size()
print("State distribution:")
print(state_counts)

print(f"\nRecords with state 0: {len(tte.recap_data[tte.recap_data.state == 0])}")
print("Sample state 0 records:")
state_0_records = tte.recap_data[tte.recap_data.state == 0]
print(state_0_records[['freq_code', 'epoch', 'time_stamp', 'state', 'rec_id']].head())

# Check for any state 0 records in the detection data (should be none)
detection_state_0 = tte.recap_data[(tte.recap_data.state == 0) & (tte.recap_data.rec_id != 'rel')]
print(f"\n⚠️  State 0 records that are NOT release records: {len(detection_state_0)}")
if len(detection_state_0) > 0:
    print("These should not exist!")
    print(detection_state_0[['freq_code', 'epoch', 'time_stamp', 'state', 'rec_id']].head())