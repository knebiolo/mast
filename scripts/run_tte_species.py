"""
Run time_to_event for a specific species and print diagnostics.
Usage:
    python scripts\run_tte_species.py --project-dir <project_dir> --db-name <db_name> --species BULL
"""
import os
import sys
import argparse
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
from pymast.radio_project import radio_project
from pymast import formatter

parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', required=True)
parser.add_argument('--db-name', required=True)
parser.add_argument('--species', required=True)
parser.add_argument('--detection_count', type=int, default=5)
parser.add_argument('--duration', type=float, default=1.0)
args = parser.parse_args()

project_dir = args.project_dir
db_name = args.db_name
species = args.species

# read CSVs
tag_csv = os.path.join(project_dir, 'tblMasterTag.csv')
rec_csv = os.path.join(project_dir, 'tblMasterReceiver.csv')
nodes_csv = os.path.join(project_dir, 'tblNodes.csv')

if not (os.path.exists(tag_csv) and os.path.exists(rec_csv)):
    print('Missing required CSVs in project dir:', tag_csv, rec_csv)
    sys.exit(1)

print('Loading project CSVs...')
tag_data = pd.read_csv(tag_csv)
receiver_data = pd.read_csv(rec_csv)
try:
    nodes_data = pd.read_csv(nodes_csv)
except Exception:
    nodes_data = None

print('Creating radio_project object...')
proj = radio_project(project_dir, db_name, args.detection_count, args.duration, tag_data, receiver_data, nodes_data)

# Build a simple receiver->state mapping using rec_id ordering if not provided
receiver_to_state = {}
for i, rec in enumerate(receiver_data['rec_id'].unique(), start=1):
    receiver_to_state[rec] = i

print('Running time_to_event for species:', species)
tte = formatter.time_to_event(receiver_to_state, proj, initial_state_release=True)

try:
    tte.data_prep(proj, species=species)
    print('TTE recap_data dtypes:')
    if hasattr(tte, 'recap_data'):
        print(tte.recap_data.dtypes)
        print('recap rows:', len(tte.recap_data))
        print(tte.recap_data.head(20))
    else:
        print('No recap_data on tte object')

    print('\nMaster state table sample:')
    if hasattr(tte, 'master_state_table'):
        print('Master rows:', len(tte.master_state_table))
        print(tte.master_state_table.head(50))
        try:
            print('\nTransition crosstab:')
            print(pd.crosstab(tte.master_state_table.start_state, tte.master_state_table.end_state))
        except Exception as e:
            print('Could not create crosstab:', e)
    else:
        print('No master_state_table produced')
except Exception as e:
    print('Error running TTE:', e)
    import traceback
    traceback.print_exc()
