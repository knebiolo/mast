"""
Debug script to understand the state transition counting issues
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
db_name = os.environ.get("PYMAST_DEBUG_DB_NAME", "pymast_debug")
if project_dir == r"C:\path\to\your\project":
    raise RuntimeError("Set PYMAST_DEBUG_PROJECT_DIR to run this debug script.")

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

project = radio_project(project_dir, db_name, detection_count, duration, tag_data, receiver_data, nodes_data)

states = {'R1696':1, 'R1699-1':2, 'R1699-2':3, 'R1698':4, 'R1699-3':5, 'R1695':5, 'R0004':6, 'R0005':6, 'R0001':7, 'R0002':7, 'R0003':8}

print("=== DEBUGGING STATE TRANSITION COUNTING ===\n")

# Test both modes
for mode_name, initial_release in [("Competing Risks (State 1)", False), ("Full Journey (Release)", True)]:
    print(f"--- {mode_name} ---")
    
    tte = formatter.time_to_event(states, project, initial_state_release=initial_release, 
                                  last_presence_time0=False, cap_loc=None, rel_loc=None, 
                                  species=None, rel_date=None, recap_date=None)
    tte.data_prep(project)
    
    # Check the master state table for issues
    print(f"Total records in master_state_table: {len(tte.master_state_table)}")
    print(f"Unique fish: {len(tte.master_state_table.freq_code.unique())}")
    
    # Check for impossible transitions (X -> 0)
    impossible_transitions = tte.master_state_table[tte.master_state_table.end_state == 0]
    if len(impossible_transitions) > 0:
        print(f"⚠️  IMPOSSIBLE TRANSITIONS TO STATE 0: {len(impossible_transitions)}")
        print("Sample impossible transitions:")
        print(impossible_transitions[['freq_code', 'start_state', 'end_state']].head())
    
    # Detailed breakdown of state 1 fish (for competing risks mode)
    if not initial_release:
        state_1_fish = tte.master_state_table[tte.master_state_table.start_state == 0]
        print(f"Fish starting from state 0 (entry to model): {len(state_1_fish)}")
        
        from_state_1 = tte.master_state_table[tte.master_state_table.start_state == 1]
        print(f"Transitions FROM state 1: {len(from_state_1)}")
        print("Breakdown of transitions from state 1:")
        print(from_state_1.groupby('end_state').size())
        
        # Check for fish that appear to "end" without transitions
        all_fish = set(tte.master_state_table.freq_code.unique())
        fish_with_transitions = set(from_state_1.freq_code.unique())
        fish_without_transitions = all_fish - fish_with_transitions
        print(f"Fish that never transition from state 1: {len(fish_without_transitions)}")
        
        # Check last observations
        last_obs_per_fish = tte.master_state_table.groupby('freq_code').tail(1)
        last_states = last_obs_per_fish.groupby('end_state').size()
        print("Final states where fish end up:")
        print(last_states)
    
    print(f"State transition crosstab:")
    transition_table = pd.crosstab(tte.master_state_table['start_state'], tte.master_state_table['end_state'])
    print(transition_table)
    print("\n" + "="*60 + "\n")
