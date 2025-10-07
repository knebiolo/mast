# import modules
import time
import os
import sqlite3
import pandas as pd
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas")
import biotas
import warnings
warnings.filterwarnings('ignore')

#%% Part 1: Set Script Parameters and Workspace


# what is the site/receiver ID?
site = 'VR2Tx-487374'
# what is the receiver type?
recType = 'vr2'
# what is the project directory?
proj_dir = r'C:\Users\knebiolo\Desktop\vr2_test'
# what did you call the database?
dbName = 'vr2_test.db'
# antenna to location, default project set up 1 Antenna, 1 Location, 1 Receiver
ant_to_rec_dict = {'1':site}

# set up workspaces
file_dir = os.path.join(proj_dir,'Data','Training_Files')
files = os.listdir(file_dir)
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
print ("There are %s files to iterate through"%(len(files)))

#%% Part 2: Import Site Data and Train Alogrithm
tS = time.time()

# Import Data, if the receiver does not switch between antennas scanTime and channels = 1.
# If the receiver switches, scanTime and channels must match study values
biotas.telemDataImport(site,
                        recType,
                        file_dir,
                        projectDB)

print ("Raw data imported, proceed to training")
