"""
Diagnostic for TTE orphan/double-count issues
Run in project environment; prints per-fish recap vs master counts and examples.
"""
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.radio_project import radio_project
from pymast import formatter
import pandas as pd

project_dir = r"C:\path\to\your\project"  # UPDATE THIS
db_name = 'thompson_2025_v3'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

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

print('Initializing TTE for species LL')
tte = formatter.time_to_event(states, project, initial_state_release=True, species='LL')
print('Running data_prep')
tte.data_prep(project)

SPECIES = 'LL'
recap = tte.recap_data.copy()
master = tte.master_state_table.copy()
recap_sp = recap[recap.species == SPECIES]
master_sp = master[master.species == SPECIES]

print('Recap rows (species):', len(recap_sp))
print('Master rows (species):', len(master_sp))

recap_counts = recap_sp.groupby('freq_code').size().rename('recap_n')
master_counts = master_sp.groupby('freq_code').size().rename('master_n')

counts = recap_counts.to_frame().join(master_counts, how='outer').fillna(0).astype(int)
print(counts.describe())

double_counted = counts[counts.master_n > counts.recap_n]
single_no_transition = counts[(counts.recap_n == 1) & (counts.master_n == 0)]

print('\nNumber of fish double-counted (master_n > recap_n):', len(double_counted))
print('Number of fish single-recapture with no transition:', len(single_no_transition))

print('\nSample double-counted fish:')
print(double_counted.head())

print('\nSample single-recap fish:')
print(single_no_transition.head())

# duplicates in master
dupes_mask = master.duplicated(subset=['freq_code','time_0','time_1','start_state','end_state'], keep=False)
print('\nTotal duplicate transition rows in master_state_table:', dupes_mask.sum())
if dupes_mask.sum() > 0:
    print(master[dupes_mask].head(20))

# print examples
for fid in list(double_counted.index[:5]):
    print('\n--- Example double-counted fish:', fid, '---')
    print('Recap rows:')
    print(recap_sp[recap_sp.freq_code==fid])
    print('Master rows:')
    print(master_sp[master_sp.freq_code==fid])

for fid in list(single_no_transition.index[:5]):
    print('\n--- Example single-recap fish:', fid, '---')
    print(recap_sp[recap_sp.freq_code==fid])

print('\nDiagnostic complete.')
