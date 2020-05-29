# import modules
import time
import os
import sqlite3
import pandas as pd
import biotas
import warnings
warnings.filterwarnings('ignore')
# set script parameters
site = 'T33'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'E:\Manuscript\CT_River_2015'             # what is the project directory?
dbName = 'ctr_2015.db'                                                       # whad did you call the database?
scanTime = 1.0
channels = 1
ant_to_rec_dict = {1:'T33'}                                                    # if orion receiver switching between antennas add more to dictionary

# set up workspaces     
file_dir = os.path.join(proj_dir,'Data','Training_Files')
files = os.listdir(file_dir)
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
print ("There are %s files to iterate through"%(len(files)))
tS = time.time()                                                             

# if you are using a Lotek receiver or orion that does not employ switching use:                                                          
biotas.telemDataImport(site,recType,file_dir,projectDB) 

# if orion recievers use swtiching use:
#biotas.telemDataImport(site,recType,file_dir,projectDB, switch = True, scanTime = scanTime, channels = channels, ant_to_rec_dict = ant_to_rec_dict) 

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND TagType IS NOT 'Beacon' AND TagType IS NOT 'Test';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.unique()
    c.close()
    print ("Finished importing data and indexing database, there are %s fish to iterate through" %(len(histories)))
    print ("Creating training objects for every fish at site %s"%(site))   
    # create a training data object for each fish and calculate training parameters.  
    for i in histories:
        train_dat = biotas.training_data(i,site,projectDB,scratch_dir)
        biotas.calc_train_params_map(train_dat)  
        print ("training parameters quantified for tag %s"%(i))

print ("Telemetry Parameters Quantified, appending data to project database")
biotas.trainDatAppend(scratch_dir,projectDB)    
print ("process took %s to compile"%(round(time.time() - tS,3)))
