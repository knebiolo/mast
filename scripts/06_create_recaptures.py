"""
Recaptures Table Script
Creates the final recaptures table by merging classified detections with presence and overlap data.

@author: KNebiolo
"""

# import modules
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from pymast.radio_project import radio_project
import pandas as pd

# Set up project
project_dir = r"C:\path\to\your\project"  # UPDATE THIS
db_name = 'thompson_2025_v3'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# Create/load project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

#%% Create recaptures table
# Undo recaptures if needed (clears old table)
project.undo_recaptures()

project.make_recaptures_table(export = True,  pit_study = True)

# Check for orphan tags (tags present in recaptures but missing from master tag list)
orphans = project.orphan_tags(return_rows=False)
if not orphans:
    print('No orphan tags found.')
else:
    out_csv = os.path.join(project_dir, 'orphans.csv')
    pd.Series(orphans, name='freq_code').to_csv(out_csv, index=False)
    print(f"Orphans found: {len(orphans)} — saved to {out_csv}")
