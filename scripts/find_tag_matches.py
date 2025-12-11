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

fids = ['3DD.003D959082','3DD.003E53996E']

print('pit_id sample (first 10):')
if 'pit_id' in proj.tags.columns:
    print(proj.tags['pit_id'].head(10).astype(str).to_list())

for fid in fids:
    print('\nLooking for exact match of', fid)
    found = False
    for col in ['freq_code','pit_id'] + [c for c in proj.tags.columns if c not in ['freq_code','pit_id']]:
        if col not in proj.tags.columns:
            continue
        try:
            matches = proj.tags[proj.tags[col].astype(str) == fid]
            if not matches.empty:
                print('Found in column', col)
                print(matches.to_string())
                found = True
        except Exception as e:
            pass
    if not found:
        print('No exact match, trying substring search in pit_id')
        if 'pit_id' in proj.tags.columns:
            subset = proj.tags[proj.tags['pit_id'].astype(str).str.contains(fid.split('.')[-1], na=False)]
            if not subset.empty:
                print('Substring match in pit_id:')
                print(subset.to_string())
                found = True
    if not found:
        print('No matches found for', fid)

print('\nDone')
