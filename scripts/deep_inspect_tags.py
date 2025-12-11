import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.radio_project import RadioProject
import pandas as pd
import re

proj = RadioProject()
print('Loaded project OK')
try:
    tag_table = proj.tags
except Exception as e:
    print('Error accessing project.tags:', e)
    tag_table = None

if tag_table is None:
    print('No tag table available')
    raise SystemExit(1)

print('\nColumns:', list(tag_table.columns))
print('\nDtypes:')
print(tag_table.dtypes)

print('\nfreq_code sample (first 20):')
if 'freq_code' in tag_table.columns:
    print(tag_table['freq_code'].head(20).astype(str).to_list())
else:
    print('No freq_code column present')

print('\nTag table head (first 50 rows):')
print(tag_table.head(50).to_string())

search_ids = ['3DD.003D959082','3DD.003E53996E']

for fish in search_ids:
    print('\nSearching for', fish)
    found_any = False
    if 'freq_code' in tag_table.columns:
        # exact
        mask_exact = tag_table['freq_code'].astype(str) == str(fish)
        if mask_exact.any():
            print(f"Exact match in 'freq_code': {mask_exact.sum()} rows; sample:\n", tag_table.loc[mask_exact].head(5).to_string())
            found_any = True
        # substring
        mask_sub = tag_table['freq_code'].astype(str).str.contains(re.escape(str(fish)), na=False)
        if mask_sub.any():
            print(f"Substring match in 'freq_code': {mask_sub.sum()} rows; sample:\n", tag_table.loc[mask_sub].head(5).to_string())
            found_any = True
        # normalized (digits-only)
        fc_norm = tag_table['freq_code'].astype(str).apply(lambda x: re.sub(r'[^0-9a-zA-Z]', '', x).lower())
        fish_norm = re.sub(r'[^0-9a-zA-Z]', '', str(fish)).lower()
        mask_norm = fc_norm == fish_norm
        if mask_norm.any():
            print(f"Normalized match in 'freq_code' for '{fish_norm}': {mask_norm.sum()} rows; sample:\n", tag_table.loc[mask_norm].head(5).to_string())
            found_any = True
    else:
        print("No 'freq_code' column to search")

    if not found_any:
        print('No matches found for', fish)

print('\nDone')
