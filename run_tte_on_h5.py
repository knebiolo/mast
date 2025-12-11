#!/usr/bin/env python3
"""
Run Time-To-Event preparer against a supplied HDF5 and print `recap_data`.

Usage:
  python run_tte_on_h5.py <path_to_h5>

This script creates a minimal `project` object (with `db` and `tags`) and
instantiates `pymast.formatter.time_to_event` using the state mapping from
`scripts/07_time_to_event_model.py`. It prints diagnostics and writes
`tte_recap_data.csv` for inspection.
"""
import sys
import os
import traceback
import argparse
import pandas as pd
import numpy as np
from types import SimpleNamespace

def main(db_path):
    print(f"Using DB: {db_path}")
    if not os.path.exists(db_path):
        print(f"ERROR: HDF5 not found: {db_path}")
        return 2

    try:
        print("Listing HDF5 keys...")
        with pd.HDFStore(db_path,'r') as store:
            keys = store.keys()
        print("HDF5 keys:", keys)
    except Exception as e:
        print("Failed to list HDF5 keys:", e)
        traceback.print_exc()
        return 3

    # Try to load tags table from common locations
    tags = None
    for candidate in ['/project_setup/tags', 'project_setup/tags', '/project_setup/tags']:
        try:
            tags = pd.read_hdf(db_path, key=candidate)
            print(f"Loaded tags from {candidate}, {len(tags)} rows")
            break
        except Exception:
            continue

    if tags is None:
        print("Warning: Could not find tags table in HDF5; creating minimal tags table.")
        tags = pd.DataFrame(columns=['freq_code','rel_date','rel_loc','cap_loc','species'])

    # Create minimal project object expected by formatter
    project = SimpleNamespace()
    project.db = db_path
    project.tags = tags
    project.output_dir = os.path.join(os.path.dirname(db_path), 'Output')

    # Use the same states mapping as your script
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

    try:
        import pymast.formatter as formatter
        print('Imported pymast.formatter OK')
    except Exception as e:
        print('Failed to import pymast.formatter:', e)
        traceback.print_exc()
        return 4

    try:
        # Inspect raw recaptures table prior to formatter to see what's present
        print('\nAttempting to read /recaptures table (first 10 rows)')
        try:
            recaps = pd.read_hdf(db_path, key='recaptures')
            print('recaptures columns:', recaps.columns.tolist())
            print('recaptures sample:')
            print(recaps.head(10))
            print('unique rec_id values (sample 50):', pd.Series(recaps.rec_id.unique()).head(50).tolist())
        except Exception as e:
            print('Could not read recaptures directly with key "recaptures":', e)
            # try with leading slash
            try:
                recaps = pd.read_hdf(db_path, key='/recaptures')
                print('recaptures columns:', recaps.columns.tolist())
                print(recaps.head(10))
            except Exception as e2:
                print('Second attempt to read recaptures failed:', e2)
                recaps = None

        print('\nInstantiating time_to_event with initial_state_release=True')
        tte = formatter.time_to_event(states, project, initial_state_release=True,
                                       last_presence_time0=False, hit_ratio_filter=False)

        # Print recap_data diagnostics
        print('\n[TTE] recap_data shape and columns:')
        try:
            rd = tte.recap_data
            print('type:', type(rd))
            print('shape:', getattr(rd, 'shape', None))
            print('columns:', list(rd.columns))
            print('head:')
            print(rd.head(20))
            print('\nunique rec_id sample (first 50):', pd.Series(rd.rec_id.unique()).head(50).tolist())
            print('\nunique freq_code sample (first 50):', pd.Series(rd.freq_code.unique()).head(50).tolist())
            # Save to CSV for manual inspection
            out_csv = os.path.abspath('tte_recap_data.csv')
            rd.to_csv(out_csv, index=False)
            print(f'Wrote recap_data to {out_csv}')
        except Exception as e:
            print('Error examining tte.recap_data:', e)
            traceback.print_exc()

        # Also run data_prep to produce master_state_table
        print('\nRunning tte.data_prep(project)')
        try:
            tte.data_prep(project)
            print('master_state_table head:')
            print(getattr(tte, 'master_state_table', pd.DataFrame()).head(20))
            m_out = os.path.abspath('tte_master_state_table.csv')
            getattr(tte, 'master_state_table', pd.DataFrame()).to_csv(m_out, index=False)
            print(f'Wrote master_state_table to {m_out}')
        except Exception as e:
            print('Error running data_prep:', e)
            traceback.print_exc()

    except Exception as e:
        print('Unexpected error during TTE run:', e)
        traceback.print_exc()
        return 5

    print('\nDone')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TTE checks against an HDF5 database')
    parser.add_argument('db_path', help='Path to .h5 database')
    args = parser.parse_args()
    sys.exit(main(args.db_path))
"""
Runner to execute time_to_event.data_prep on a provided HDF5 database
Usage:
  python run_tte_on_h5.py "K:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\thompson_2025.h5" 3DD.003E53A7D4

This script mirrors the `07_time_to_event_model.py` approach but takes explicit db path and tag.
"""
import sys
import os
import ast
import traceback

if len(sys.argv) < 3:
    print("Usage: python run_tte_on_h5.py <path_to_h5> <tag_freq_code>")
    sys.exit(1)

db_path = sys.argv[1]
tag = sys.argv[2]

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print('Using Python executable:', sys.executable)
print('Database path:', db_path)
print('Tag:', tag)

from pymast.formatter import time_to_event
import pandas as pd
from types import SimpleNamespace

def extract_upstream_states(script_path):
    """Safely parse `upstream_states` dict from the provided script file."""
    if not os.path.exists(script_path):
        return None
    with open(script_path, 'r', encoding='utf-8') as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=script_path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'upstream_states':
                        return ast.literal_eval(node.value)
    except Exception:
        return None
    return None

try:
    # Read tags directly from HDF5
    tags = pd.read_hdf(db_path, key='tblMasterTag')
except Exception as e:
    print('Error reading tblMasterTag from HDF5:', e)
    raise

# Create a minimal project object expected by formatter
proj = SimpleNamespace()
proj.db = db_path
proj.project_dir = os.path.dirname(db_path)
proj.tags = tags
proj.output_dir = os.path.join(proj.project_dir, 'Output')

# Try to extract mapping from the user's script if present
script_path = os.path.join(repo_root, 'scripts', '07_time_to_event_model.py')
receiver_to_state = extract_upstream_states(script_path)
if receiver_to_state is None:
    print('Could not extract upstream_states from script; using minimal mapping: {"rel": 0}')
    receiver_to_state = {'rel': 0}

try:
    tte = time_to_event(receiver_to_state, proj, initial_state_release=True)
    master = tte.data_prep(proj, tag_list=[tag])
    # If data_prep returns a DataFrame, print it; some implementations set master_state_table instead
    if isinstance(master, pd.DataFrame):
        out_df = master
    else:
        out_df = tte.master_state_table

    print('\nmaster_state_table for tag:', tag)
    # Filter to the tag and print
    if 'freq_code' in out_df.columns:
        print(out_df[out_df['freq_code'] == tag].to_string(index=False))
    else:
        print(out_df.to_string(index=False))
except Exception:
    traceback.print_exc()
    raise
