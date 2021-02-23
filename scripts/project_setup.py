import sys
sys.path.append(r'D:\Manuscript\CT_River_2015') # Enter the path for where the BIOTAS program lives
import biotas
proj_dir = r'D:\Manuscript\CT_River_2015'
dbName = 'ctr_2015_v3.db'
biotas.createTrainDB(proj_dir, dbName)  # create project database
