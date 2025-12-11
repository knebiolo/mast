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

fish_ids = ['3DD.003D959082','3DD.003E53996E']

# Workflow A
print('Building TTE A (species at init)')
tteA = formatter.time_to_event(states, project, initial_state_release=True, species='LL')
tteA.data_prep(project)
A = tteA.master_state_table.copy()

# Workflow B
print('Building TTE B (no species at init)')
tteB = formatter.time_to_event(states, project, initial_state_release=True, species=None)
tteB.data_prep(project)
B = tteB.master_state_table.copy()
Bf = B[B.species == 'LL'].copy()

for fid in fish_ids:
    print('\n--- Fish', fid, 'rows in A ---')
    print(A[A.freq_code == fid].sort_values(['time_0','time_1']).to_string(index=False))
    print('\n--- Fish', fid, 'rows in B_filtered ---')
    print(Bf[Bf.freq_code == fid].sort_values(['time_0','time_1']).to_string(index=False))

print('\nDone')
