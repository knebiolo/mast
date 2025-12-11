"""
Find rows in master_state_table with end_state == 0 and show corresponding recap rows.
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

print('Initializing TTE for species LL')
tte = formatter.time_to_event(states, project, initial_state_release=True, species='LL')
print('Running data_prep')
tte.data_prep(project)

master = tte.master_state_table
recap = tte.recap_data

zeros = master[master.end_state == 0]
print('Rows in master_state_table with end_state == 0:', len(zeros))
if len(zeros) > 0:
    print(zeros.head(30))

# For each zero row, show recap rows for that fish around the time
for idx, row in zeros.iterrows():
    fish = row['freq_code']
    t0 = int(row['time_0'])
    t1 = int(row['time_1'])
    print(f"\n--- Fish {fish} zero-row context ({t0} to {t1}) ---")
    sub = recap[recap.freq_code == fish].sort_values('epoch')
    # show recaps within +/- 1 day of the recorded interval
    window = sub[(sub.epoch >= t0 - 86400) & (sub.epoch <= t1 + 86400)]
    print(window[['freq_code','rec_id','epoch','time_stamp','state']].head(50))

print('\nDone')
