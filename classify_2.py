 # Module contains all of the objects and required for analysis of telemetry data

# import modules required for function dependencies
import multiprocessing as mp
import time
import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import abtas

'''train data - we can run function in serial or multiprocessing over cores in a
computer.  this script uses multiprocessing, which is why we call the process 
behind the main statement.'''
if __name__ == "__main__":
    tS = time.time()
    #set script parameters
    site = 't10Hol'                                                            # what is the site/receiver ID?
    recType = 'orion'                                                          # what is the receiver type?
    project_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                # what is the raw data directory
    dbName = 'ultrasound_2018_test.db'                                         # what is the name of the project database
    
    # create worskspaces - you haven't changed the directory have you?                                              
    trainingDB = os.path.join(project_dir,'Data',dbName)
    outputWS = os.path.join(project_dir,'Output')                              # we are getting time out error and database locks - so let's write to disk for now 
    outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now 
    figure_ws = os.path.join(outputWS,'Figures')
    workFiles = os.path.join(project_dir,'Data','TrainingFiles')
    projectDB = os.path.join(project_dir,'Data',dbName)

    # list fields used in likelihood classification, must be from this list:
    # ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
    fields = ['conRecLength','hitRatio','power','lagDiff']

    files = os.listdir(workFiles)
    print ("There are %s files to iterate through"%(len(files)))
    tS = time.time()                                                            # what time is it? for script performanceg
    
    abtas.telemDataImport(site,recType,workFiles,projectDB) 
    
    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT DISTINCT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND TagType == 'Study';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.values
    c.close()

    print ("There are %s fish to iterate through at site %s" %(len(histories),site))
    
    # create list of training data objects to iterate over with a Pool multiprocess
    iters = []
    for i in histories:
        iters.append(abtas.classify_data(i,site,fields,projectDB,outputScratch,training = trainingDB))

    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")

    pool = mp.Pool(processes = 8)                                               # the number of processes equals the number of processors you have
    pool.map(abtas.calc_class_params_map, iters)                                # map the parameter functions over each training data object 
     
    print ("Telemetry Parameters Quantified, appending data to project database")
    abtas.classDatAppend(site,outputScratch,projectDB)

   
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    # generate summary statistics for classification by receiver type
    class_stats = abtas.classification_results(recType,projectDB,figure_ws,site)
    class_stats.classify_stats()