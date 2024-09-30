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
h5_file = os.path.join(project_dir,db_name) # Replace with your actual HDF5 file path

# The key (dataset) in the HDF5 file to update
key = 'presence'  # Replace with the actual key in your HDF5 file

# Define the new minimum size for the 'freq_code' column
a = 20  # Adjust this value based on your needs

# Load the existing data from the HDF5 file
df = pd.read_hdf(h5_file, key)

# Save the DataFrame back to the HDF5 file, with an updated size for 'freq_code'
df.to_hdf(h5_file, key, mode='a', format='table', data_columns=True,
          min_itemsize={'freq_code': a})