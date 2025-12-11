"""
Compare TTE master_state_table when filtering at init vs filtering after data_prep.
"""
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.radio_project import radio_project
from pymast import formatter
import pandas as pd

project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_12_04"
db_name = 'thompson_2025_v3'

tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

project = radio_project(project_dir, db_name, 5, 1, tag_data, receiver_data, nodes_data)

states = {'R1696':1,'R1699-1':2,'R1699-2':3,'R1698':4,'R1699-3':5,'R1695':5,'R0004':6,'R0005':6,'R0001':7,'R0002':7,'R0003':8}

# Workflow A: filter at init
tteA = formatter.time_to_event(states, project, initial_state_release=True, species='LL')
tteA.data_prep(project)
A = tteA.master_state_table.copy()
A['__source'] = 'A'

# Workflow B: no filter at init, filter master after
tteB = formatter.time_to_event(states, project, initial_state_release=True, species=None)
tteB.data_prep(project)
B = tteB.master_state_table.copy()
B_filtered = B[B.species == 'LL'].copy()
B_filtered['__source'] = 'B'

print('A rows:', len(A))
print('B_filtered rows:', len(B_filtered))

# Normalize columns for comparison
cmp_cols = ['freq_code','time_0','time_1','start_state','end_state','presence']
for df in (A, B_filtered):
    for c in ['time_0','time_1']:
        if c in df.columns:
            df[c] = df[c].astype('int64')
    for c in ['start_state','end_state','presence']:
        if c in df.columns:
            df[c] = df[c].astype('int64')

# rows in A not in B_filtered
A_not_B = pd.merge(A, B_filtered, on=cmp_cols, how='outer', indicator=True)
onlyA = A_not_B[A_not_B['_merge']=='left_only']
onlyB = A_not_B[A_not_B['_merge']=='right_only']

print('Rows only in A (count):', len(onlyA))
print('Rows only in B_filtered (count):', len(onlyB))

if len(onlyA) > 0:
    print('\nSample rows only in A:')
    print(onlyA.head()[cmp_cols])
if len(onlyB) > 0:
    print('\nSample rows only in B_filtered:')
    print(onlyB.head()[cmp_cols])

# Check per-fish differences
A_counts = A.groupby('freq_code').size().rename('A_n')
B_counts = B_filtered.groupby('freq_code').size().rename('B_n')
counts = A_counts.to_frame().join(B_counts, how='outer').fillna(0).astype(int)
counts['diff'] = counts['A_n'] - counts['B_n']
print('\nPer-fish differences (summary):')
print(counts['diff'].describe())

# List fish with non-zero diff
diff_fish = counts[counts['diff'] != 0]
print('\nNumber of fish with differing counts:', len(diff_fish))
if len(diff_fish) > 0:
    print(diff_fish.head())

print('\nDone')
