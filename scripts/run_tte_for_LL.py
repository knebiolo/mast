import os
import sys
import pandas as pd

# Ensure repo root is on sys.path so local `pymast` package imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.radio_project import radio_project
from pymast.formatter import time_to_event

# Configure paths (adjust if needed)
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_12_04"
db_name = 'thompson_2025_v2'
db_path = os.path.join(project_dir, f"{db_name}.h5")

# Load project CSVs
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

proj = radio_project(project_dir, db_name, detection_count=5, duration=1, tag_data=tag_data, receiver_data=receiver_data, nodes_data=nodes_data)
print('Project initialized. DB path:', proj.db)

# Build receiver->state map from receivers file; if there's a 'state' column, use it; else map rec_id->index
if 'state' in receiver_data.columns:
    receiver_to_state = dict(zip(receiver_data.rec_id.astype(str), receiver_data.state.astype(int)))
else:
    # fallback: sequential state numbers (not ideal)
    receiver_to_state = {rid: i+1 for i, rid in enumerate(receiver_data.rec_id.astype(str).unique())}

print('Receiver->state map sample:', list(receiver_to_state.items())[:10])

# Run TTE for species LL
tte = time_to_event(receiver_to_state, proj, initial_state_release=True)

# Run data_prep with species filter
print('Running data_prep for species LL...')
try:
    tte.data_prep(proj, species='LL')
except Exception as e:
    import traceback
    print('Error running tte.data_prep:', e)
    traceback.print_exc()
    raise

print('recap_data dtypes:')
print(tte.recap_data.dtypes)
print('recap_data head:')
print(tte.recap_data.head(20))

print('\nMaster state table head:')
print(tte.master_state_table.head(50))

print('\nTransition matrix:')
print(pd.crosstab(tte.master_state_table.start_state, tte.master_state_table.end_state))

print('\nDone')
