# -*- coding: utf-8 -*-

"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
import dask.dataframe as dd
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\mast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"K:\Jobs\3671\014\Analysis\Submersible_Data\enm"
db_name = 'Thompson_Data_05012025.h5'
tag_dat = 'tblMasterTag.csv'

# identify receiver and fish you wish to investigate
rec_id = 'R04'
freq_code = '148.300 45'

rec_dat = dd.read_hdf(os.path.join(project_dir,db_name), key='classified')

# Filter for specific rec_id and convert to pandas DataFrame
rec_dat = rec_dat[rec_dat['rec_id'] == rec_id].compute()
print(f"Length of rec_dat after filtering by receiver ID: {len(rec_dat)}")


# Convert 'timestamp' column to datetime
rec_dat['time_stamp'] = pd.to_datetime(rec_dat['time_stamp'])

# Merge with release dates to filter out data before release
tags = pd.read_csv(os.path.join(project_dir,tag_dat))
rec_dat = rec_dat.merge(tags, left_on='freq_code', right_index=True)
rec_dat = rec_dat[rec_dat['time_stamp'] >= rec_dat['rel_date']]
print(f"Length of rec_dat after merging with release dates: {len(rec_dat)}")

# Reset index to avoid ambiguity between index and column labels
if 'freq_code' in rec_dat.columns and 'freq_code' in rec_dat.index.names:
    rec_dat = rec_dat.reset_index(drop=True)

# Filter by latest iteration and valid test
idxmax_values = rec_dat.groupby(['freq_code', 'rec_id'])['iter'].idxmax()
rec_dat = rec_dat.loc[idxmax_values]
print (f"Length of rec_dat after filtering by receiver ID: {len(rec_dat)}")