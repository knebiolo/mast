import sys
sys.path.append(r'E:\Manuscript\CT_River_2015') # Enter the path for where the BIOTAS program lives
import biotas
proj_dir = r'E:\Manuscript\CT_River_2015'                   
dbName = 'ctr_2015_v2.db' 
biotas.createTrainDB(proj_dir, dbName)  # create project database
