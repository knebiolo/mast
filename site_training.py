import multiprocessing as mp
import time
import os
import sqlite3
import pandas as pd
import abtas
import warnings
warnings.filterwarnings('ignore')
'''train data - we can run function in serial or multiprocessing over cores in a
computer.  this script uses multiprocessing, which is why we call the process 
behind the main statement.'''
if __name__ == "__main__":
    # set script parameters
    site = 't04'                                                               # what is the site/receiver ID?
    recType = 'orion'                                                          # what is the receiver type?
    proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                   # what is the project directory?
    dbName = 'ultrasound_2018_test.db'                                         # what is the name of the project database    
    file_dir = os.path.join(proj_dir,'Data','Training_Files')
    files = os.listdir(file_dir)
    projectDB = os.path.join(proj_dir,'Data',dbName)
    scratch_dir = os.path.join(proj_dir,'Output','Scratch')
    print ("There are %s files to iterate through"%(len(files)))
    tS = time.time()                                                           # what time is it? for script performanceg    
    abtas.telemDataImport(site,recType,file_dir, projectDB)                    # import data automatically using built in text parsers, programming is FUN! THIS IS RAW DATA!    
    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT DISTINCT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND TagType IS NOT 'Study';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.values
    c.close()
    print ("Finished importing data and indexing database, there are %s fish to iterate through" %(len(histories)))
    print ("Creating training objects for every fish at site %s"%(site))    
    # create list of training data objects to iterate over with a Pool multiprocess
    iters = []
    for i in histories:
        iters.append(abtas.training_data(i,site,projectDB,scratch_dir))
    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")
    pool = mp.Pool(processes = 3)                                               # the number of processes equals the number of processors you have
    pool.map(abtas.calc_train_params_map, iters)                                # map the parameter functions over each training data object      
    print ("Telemetry Parameters Quantified, appending data to project database")
    abtas.trainDatAppend(scratch_dir,projectDB)    
    print ("process took %s to compile"%(round(time.time() - tS,3)))