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
site = 2
# what is the receiver type?
recType = 'srx800'
# what is the project directory?
proj_dir = r'C:\Users\knebiolo\Desktop\Nuyakuk BIOTAS'
# what did you call the database?
dbName = 'nuya_test.db'
# antenna to location, default project set up 1 Antenna, 1 Location, 1 Receiver
ant_to_rec_dict = {'A0':site}

# set up workspaces
file_dir = os.path.join(proj_dir,'Data','Training_Files')
files = os.listdir(file_dir)
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
print ("There are %s files to iterate through"%(len(files)))

#%% Part 2: Import Site Data and Train Alogrithm
tS = time.time()

#Import Data, if the receiver does not switch between antennas scanTime and channels = 1.
#If the receiver switches, scanTime and channels must match study values

biotas.telemDataImport(site,
                        recType,
                        file_dir,
                        projectDB,
                        scanTime = 1,
                        channels = 1,
                        ant_to_rec_dict = ant_to_rec_dict)

print ("Raw data imported, proceed to training")

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL
    # this SQL statement allows us to train on study tags
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
    

    # create a training data object for each fish and train naive Bayes.
    for j in histories:
        train_dat = biotas.training_data(j,
                                         ant_to_rec_dict[i],
                                         projectDB,
                                         scratch_dir)
        
        biotas.calc_train_params_map(train_dat)
        print ("training parameters quantified for tag %s"%(j))
    print ("Telemetry Parameters Quantified, appending data to project database")
    # append data and summarize
    biotas.trainDatAppend(scratch_dir,projectDB)
    train_stats = biotas.training_results(recType,
                                          projectDB,
                                          figure_ws,
                                          ant_to_rec_dict[i])
    train_stats.train_stats()

print ("process took %s to compile"%(round(time.time() - tS,3)))