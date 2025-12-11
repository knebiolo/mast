#!/usr/bin/env python3
"""
Migrate PIT epoch to int64 and optionally run time-to-event pipeline.

Usage:
  python scripts/migrate_pit_epoch_and_run_tte.py --h5 "K:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\thompson_2025.h5" \
      [--backup] [--run-tte --project-dir "K:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01"] [--tag TAG]

This script will:
- create a timestamped backup of the HDF5 (if --backup)
- read `/raw_data` from the HDF5, find rows where `rec_type` contains 'PIT' (or 'PIT_Multiple')
- recompute `epoch` as integer seconds using `time_stamp.astype('int64') // 10**9` and cast to int64
- write the table back (overwriting `/raw_data`).

If --run-tte is provided and the project directory contains the standard CSVs
(`tblMasterTag.csv`, `tblMasterReceiver.csv`, `tblNodes.csv`) the script will attempt to
instantiate your `radio_project` and run the `formatter.time_to_event` pipeline for a
single tag (provided with `--tag`, default `3DD.003E53A7D4`). The TTE run is optional.

IMPORTANT: This script overwrites `/raw_data` in the HDF5. Make a backup before running
if you are unsure. Use the --backup flag to create an automatic copy.
"""

import argparse
import os
import shutil
import datetime
import sys
import traceback

import pandas as pd


def backup_file(path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(path)[0]
    out = f"{base}.backup.{ts}.h5"
    print(f"Creating backup: {out}")
    shutil.copy2(path, out)
    return out


def migrate_epoch(h5_path):
    print('Opening HDF5:', h5_path)
    # Read full raw_data table (may be large).
    df = pd.read_hdf(h5_path, 'raw_data')
    print('raw_data rows:', len(df))

    # Identify PIT rows by rec_type or rec_id containing 'PIT'
    mask = df['rec_type'].astype(str).str.contains('PIT', case=False, na=False) | \
           df['rec_id'].astype(str).str.contains('PIT', case=False, na=False)

    n_pit = mask.sum()
    print(f'Found {n_pit} PIT rows to migrate')
    if n_pit == 0:
        print('No PIT rows found. No changes made.')
        return df

    # Ensure time_stamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.loc[mask, 'time_stamp']):
        print('Converting time_stamp to datetime for PIT rows')
        df.loc[mask, 'time_stamp'] = pd.to_datetime(df.loc[mask, 'time_stamp'], errors='coerce')
        n_na = df.loc[mask, 'time_stamp'].isna().sum()
        if n_na > 0:
            print(f'Warning: {n_na} PIT rows have invalid time_stamp after parsing')

    # Compute epoch as int64 seconds
    print('Computing epoch as int64 seconds')
    df.loc[mask, 'epoch'] = (df.loc[mask, 'time_stamp'].astype('int64') // 10**9).astype('int64')

    # Sanity sample
    print('Sample migrated PIT rows (first 5):')
    print(df.loc[mask, ['rec_id', 'freq_code', 'time_stamp', 'epoch']].head(5))

    # Write back to HDF5 (overwrite raw_data). Keep format='table' and data_columns=True
    print('Writing updated raw_data back to HDF5 (this will overwrite /raw_data)')
    with pd.HDFStore(h5_path, mode='a') as store:
        store.put('raw_data', df, format='table', data_columns=True)

    print('Write complete')
    return df


def run_tte_if_requested(project_dir, h5_path, tag):
    try:
        # Import local project classes
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from pymast.radio_project import radio_project
        from pymast import formatter

        # Try to read required CSVs
        tag_csv = os.path.join(project_dir, 'tblMasterTag.csv')
        rec_csv = os.path.join(project_dir, 'tblMasterReceiver.csv')
        nodes_csv = os.path.join(project_dir, 'tblNodes.csv')

        if not (os.path.exists(tag_csv) and os.path.exists(rec_csv) and os.path.exists(nodes_csv)):
            print('One or more project CSVs missing in project_dir; cannot instantiate radio_project.')
            print('Expected:', tag_csv, rec_csv, nodes_csv)
            return

        tag_data = pd.read_csv(tag_csv)
        receiver_data = pd.read_csv(rec_csv)
        nodes_data = pd.read_csv(nodes_csv)

        # Use defaults similar to your script
        detection_count = 5
        duration = 1

        print('Creating radio_project object...')
        project = radio_project(project_dir, os.path.splitext(os.path.basename(h5_path))[0], detection_count, duration, tag_data, receiver_data, nodes_data)

        # States mapping from your example script
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

        tte = formatter.time_to_event(states,
                                      project,
                                      initial_state_release=True,
                                      last_presence_time0=False,
                                      hit_ratio_filter=False,
                                      cap_loc=None,
                                      rel_loc=None,
                                      species=None,
                                      rel_date=None,
                                      recap_date=None)

        print('Running tte.data_prep for tag:', tag)
        # call the data_prep using the single-tag signature used in current code
        try:
            tte.data_prep(project, tag)
        except TypeError:
            # fallback if older signature expects tag_list kwarg
            try:
                tte.data_prep(project, tag_list=[tag])
            except Exception:
                print('tte.data_prep failed with both call signatures')
                raise

        print('tte.recap_data sample:')
        if hasattr(tte, 'recap_data') and getattr(tte, 'recap_data') is not None:
            try:
                print(tte.recap_data.dtypes)
                print(tte.recap_data.head(50))
                print('tte.recap_data rows:', len(tte.recap_data))
            except Exception:
                print('tte.recap_data not available or not a DataFrame')
        else:
            print('tte.recap_data not present on tte object')

    except Exception as e:
        print('Error running TTE:', e)
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Migrate PIT epoch to int64 and optionally run TTE')
    parser.add_argument('--h5', dest='h5_path', required=True, help='Path to project HDF5 file')
    parser.add_argument('--backup', action='store_true', help='Create a timestamped backup of the HDF5 before modifying')
    parser.add_argument('--run-tte', action='store_true', help='After migration, attempt to run TTE (requires project_dir CSVs)')
    parser.add_argument('--project-dir', help='Project dir containing tblMasterTag.csv etc. Required for --run-tte')
    parser.add_argument('--tag', default='3DD.003E53A7D4', help='Tag to run TTE for when using --run-tte')

    args = parser.parse_args()

    h5_path = args.h5_path
    if not os.path.exists(h5_path):
        print('HDF5 file not found:', h5_path)
        return

    if args.backup:
        backup_file(h5_path)

    try:
        df = migrate_epoch(h5_path)
    except Exception as e:
        print('Migration failed:', e)
        traceback.print_exc()
        return

    if args.run_tte:
        if not args.project_dir:
            print('Missing --project-dir required for --run-tte')
            return
        run_tte_if_requested(args.project_dir, h5_path, args.tag)


if __name__ == '__main__':
    main()
