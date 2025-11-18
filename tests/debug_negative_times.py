"""
Debug script to identify the source of negative time deltas in TTE model
"""

import os
import sys
sys.path.append(r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01\mast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pandas as pd
import numpy as np

# Set up project (same as your main script)
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01"
db_name = 'thompson_2025'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# Create project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

# Same states as your main script
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

print("=== DEBUGGING NEGATIVE TIME DELTAS ===\n")

# Step 1: Check the raw recapture data
print("1. Checking raw recapture data...")
query_parts = []
for key in states:
    query_parts.append(f"rec_id == '{key}'")   
qry = " | ".join(query_parts)

recap_data = pd.read_hdf(project.db, 'recaptures', where=qry)
recap_data = recap_data.merge(project.tags, left_on='freq_code', right_index=True, how='left')
recap_data = recap_data[recap_data.overlapping == 0]
recap_data['state'] = recap_data.rec_id.map(states)
recap_data.sort_values(by=['freq_code', 'epoch'], ascending=True, inplace=True)

print(f"Total recaptures: {len(recap_data)}")
print(f"Unique fish: {len(recap_data.freq_code.unique())}")
print(f"Date range: {recap_data.time_stamp.min()} to {recap_data.time_stamp.max()}")

# Step 2: Check release dates vs first recaptures
print("\n2. Checking release dates vs first recaptures...")
release_times = project.tags[['rel_date']].copy()
release_times['rel_date'] = pd.to_datetime(release_times.rel_date)
release_times['rel_epoch'] = (release_times['rel_date'] - pd.Timestamp('1970-01-01')).dt.total_seconds()

first_recaps = recap_data.groupby('freq_code')['epoch'].min().to_frame()
first_recaps.columns = ['first_recap_epoch']

comparison = release_times.join(first_recaps, how='inner')
comparison['time_diff'] = comparison['first_recap_epoch'] - comparison['rel_epoch']

print(f"Fish with first recapture before release: {(comparison['time_diff'] < 0).sum()}")
if (comparison['time_diff'] < 0).any():
    print("Problematic fish:")
    problematic = comparison[comparison['time_diff'] < 0]
    for fish, row in problematic.head(5).iterrows():
        print(f"  Fish {fish}: Release {row['rel_date']}, First recap at epoch {row['first_recap_epoch']}")

# Step 3: Check state 1 first detections
print("\n3. Checking state 1 first detections...")
state1_first = recap_data[recap_data.state == 1].groupby('freq_code')['epoch'].min().to_frame()
state1_first.columns = ['state1_first_epoch']

state1_comparison = release_times.join(state1_first, how='inner')
state1_comparison['time_diff'] = state1_comparison['state1_first_epoch'] - state1_comparison['rel_epoch']

print(f"Fish with state 1 detection before release: {(state1_comparison['time_diff'] < 0).sum()}")
if (state1_comparison['time_diff'] < 0).any():
    print("Problematic fish in state 1:")
    problematic = state1_comparison[state1_comparison['time_diff'] < 0]
    for fish, row in problematic.head(5).iterrows():
        print(f"  Fish {fish}: Release {row['rel_date']}, State 1 first at epoch {row['state1_first_epoch']}")

# Step 4: Check for out-of-order detections
print("\n4. Checking for out-of-order detections within fish...")
out_of_order_count = 0
for fish in recap_data.freq_code.unique():
    fish_data = recap_data[recap_data.freq_code == fish].sort_values('epoch')
    if not fish_data.epoch.is_monotonic_increasing:
        out_of_order_count += 1
        if out_of_order_count <= 3:  # Show first 3 examples
            print(f"  Fish {fish} has out-of-order detections:")
            print(fish_data[['time_stamp', 'epoch', 'rec_id', 'state']].head())

print(f"Total fish with out-of-order detections: {out_of_order_count}")

# Step 5: Sample the actual TTE calculation to see where negatives occur
print("\n5. Simulating TTE time_delta calculation...")
test_data = recap_data.copy()
test_data['prev_state'] = test_data.groupby('freq_code')['state'].shift(1).fillna(0).astype(int)

# Get start times (same logic as in formatter)
start_times = test_data[test_data.state == 1].groupby(['freq_code'])['epoch'].min().to_frame()
start_times.rename(columns={'epoch': 'first_recapture'}, inplace=True)

# Merge start times
test_data = test_data.merge(start_times.reset_index(), on='freq_code', how='left')

# Calculate time_0 and time_delta
test_data['time_0'] = test_data.groupby('freq_code')['epoch'].shift(1)
test_data['time_0'].fillna(test_data['first_recapture'], inplace=True)
test_data['time_delta'] = test_data['epoch'] - test_data['time_0']

# Find negative time deltas
negative_deltas = test_data[test_data['time_delta'] < 0]
print(f"Records with negative time_delta: {len(negative_deltas)}")

if len(negative_deltas) > 0:
    print("\nSample negative time_delta records:")
    print(negative_deltas[['freq_code', 'time_stamp', 'epoch', 'time_0', 'time_delta', 'rec_id', 'state']].head())
    
    # Show the context around these negative deltas
    for fish in negative_deltas.freq_code.unique()[:3]:
        print(f"\nFull sequence for fish {fish}:")
        fish_seq = test_data[test_data.freq_code == fish][['time_stamp', 'epoch', 'time_0', 'time_delta', 'rec_id', 'state']].sort_values('epoch')
        print(fish_seq)

print("\n=== DEBUG COMPLETE ===")