# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:16:11 2024

@author: Kevin.Nebiolo
"""

import pandas as pd
import os

# Path to the HDF5 file
project_dir = r"C:\Users\knebiolo\Desktop\York Haven"
db_name = 'york_haven_3.h5'
h5_file = os.path.join(project_dir,db_name)  # Replace with the path to your HDF5 file

# The key (dataset) in the HDF5 file to update
key = 'presence'  # Replace with the actual key in your HDF5 file

# Load the existing data from the HDF5 file
df = pd.read_hdf(h5_file, key)

# Add the 'power' column if it doesn't already exist
if 'power' not in df.columns:
    df['power'] = 0  # Set a default value of 0 for the 'power' column
    print(f"Added 'power' column with default value 0.")
else:
    print("'power' column already exists, no changes made.")

# Save the updated DataFrame back to the HDF5 file
df.to_hdf(h5_file, key, mode='a', format='table', data_columns=True)
print(f"Updated dataset '{key}' saved to '{h5_file}'.")
