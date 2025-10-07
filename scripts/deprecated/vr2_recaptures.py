# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:41:18 2022

@author: KNebiolo

Intent: create tblRecaptures with VR2 data
"""

# import modules
import os
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas")
import pandas as pd
import sqlite3
import biotas
import warnings
warnings.filterwarnings('ignore')

# what is the project directory?
proj_dir = r'C:\Users\knebiolo\Desktop\vr2_test'
# what did you call the database?
dbName = 'vr2_test.db'

# create a tblRecaptures
recaptures = biotas.vr2_recaps(os.path.join(proj_dir,'Data',dbName))

# write to sql
conn = sqlite3.connect(os.path.join(proj_dir,'Data',dbName))
c = conn.cursor()
recaptures.to_sql('tblRecaptures',con = conn,index = False, if_exists = 'append', chunksize = 1000)
c.close()
