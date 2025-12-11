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

print('Building TTE A (species at init)')
tteA = formatter.time_to_event(states, project, initial_state_release=True, species='LL')
tteA.data_prep(project)
A = tteA.master_state_table.copy()

print('Building TTE B (no species at init)')
tteB = formatter.time_to_event(states, project, initial_state_release=True, species=None)
tteB.data_prep(project)
B = tteB.master_state_table.copy()

for fid in fish_ids:
    print('\n=== Fish', fid, '===')
    print('\nA full rows:')
    print(A[A.freq_code==fid].to_string(index=False))
    print('\nB full rows:')
    print(B[B.freq_code==fid].to_string(index=False))
    print('\nB rows where start_state==0:')
    b0 = B[(B.freq_code==fid) & (B.start_state==0)]
    if b0.empty:
        print('  (none)')
    else:
        print(b0.to_string(index=False))
    print('\nSpecies values in B release rows (start_state==0):')
    if not b0.empty:
        print(b0['species'].unique())
    print('\nAny rows in B with start_state==0 and species==LL?')
    print(len(B[(B.freq_code==fid)&(B.start_state==0)&(B.species=='LL')]))

print('\nDone')
