import os,sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.radio_project import radio_project
import pandas as pd

project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_12_04"
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))
proj = radio_project(project_dir,'thompson_2025_v3',5,1,tag_data,receiver_data,nodes_data)

print('tags columns:', proj.tags.columns.tolist())
print('\nNumber of tag rows:', len(proj.tags))

fids = ['3DD.003D959082','3DD.003E53996E']
for fid in fids:
    print('\n--- Searching for', fid, '---')
    found = False
    for col in proj.tags.columns:
        try:
            matches = proj.tags[proj.tags[col] == fid]
            if not matches.empty:
                print('Column match:', col)
                print(matches.to_string())
                found = True
        except Exception:
            pass
    if not found:
        print('No exact column matches. Checking substring in pit_id...')
        if 'pit_id' in proj.tags.columns:
            subset = proj.tags[proj.tags['pit_id'].str.contains(fid.split('.')[-1], na=False)]
            if not subset.empty:
                print('Substring matches in pit_id:')
                print(subset.to_string())
                found = True
    if not found:
        print('No matches found for', fid)
print('\nDone')
