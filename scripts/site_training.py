# import modules
import time
import os
import sqlite3
import pandas as pd
import biotas.biotas as biotas
import warnings
warnings.filterwarnings('ignore')
# set script parameters

site = 'T27'                                                                   # what is the site/receiver ID?
recType = 'lotek'                                                              # what is the receiver type?
proj_dir = r'D:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                         # what did you call the database?
ant_to_rec_dict = {1:site}                                                    # if orion receiver switching between antennas add more to dictionary

# set up workspaces     
file_dir = os.path.join(proj_dir,'Data','Training_Files')
files = os.listdir(file_dir)
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
print ("There are %s files to iterate through"%(len(files)))
tS = time.time()

# if you are using a  receiver does not employ switching use:
biotas.telemDataImport(site,recType,file_dir,projectDB)

# if recievers use swtiching use - ***note hard coded values for scanTime and channels***:
#biotas.telemDataImport(site,recType,file_dir,projectDB, scanTime = 2, channels = 2, ant_to_rec_dict = ant_to_rec_dict)

print ("Raw data imported, proceed to training")

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql ='''SELECT tblRaw.FreqCode FROM tblRaw 
            LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode 
            WHERE recID == '%s' 
            AND TagType IS NOT 'Beacon' 
            AND TagType IS NOT 'Test';'''%(ant_to_rec_dict[i])
    histories = pd.read_sql_query(sql,con = conn).FreqCode.unique()

    c.close()
    print ("There are %s fish to iterate through" %(len(histories)))
    print ("Creating training objects for every fish at site %s"%(site))
    # create a training data object for each fish and calculate training parameters.
    for j in histories:
        if j != '149.800 -266' and j != '149.720 -258':
            train_dat = biotas.training_data(j,ant_to_rec_dict[i],projectDB,scratch_dir)
            biotas.calc_train_params_map(train_dat)
            print ("training parameters quantified for tag %s"%(j))
    print ("Telemetry Parameters Quantified, appending data to project database")
    biotas.trainDatAppend(scratch_dir,projectDB)
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    train_stats = biotas.training_results(recType,projectDB,figure_ws,ant_to_rec_dict[i])
    train_stats.train_stats()

