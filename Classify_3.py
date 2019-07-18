# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 10:04:21 2018
Code to allow us to use actual study tags as beacons---work in progress
@author: tcastrosantos
"""

# import modules required for function dependencies
import multiprocessing as mp
import time
import os
import sqlite3
import pandas as pd
import abtas
import warnings
warnings.filterwarnings('ignore')
print ("Modules Imported")

'''train data - we can run function in serial or multiprocessing over cores in a
computer.  this script uses multiprocessing, which is why we call the process 
behind the main statement.'''
if __name__ == "__main__":
    tS = time.time()
    #set script parameters
    site = '205'                                                               # what is the site/receiver ID?
    class_iter= 2 #Enter the iteration number here--start at 2
    recType = 'orion'                                                          # what is the receiver type?
    proj_dir = r"D:\a\Projects\Eel Telemetry\Shetucket"                        # what is the raw data directory
    dbName = 'Shetucket.db'                                                    # what is the name of the project database
    
    outputWS = os.path.join(proj_dir,'Output')                                 # we are getting time out error and database locks - so let's write to disk for now 
    outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now 
    workFiles = os.path.join(proj_dir,'Data','TrainingFiles')
    projectDB = os.path.join(proj_dir,'Data',dbName)
    figure_ws = os.path.join(proj_dir,'Output,','Figures')
    
    # list fields used in likelihood classification, must be from this list:
    # ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
    fields = ['conRecLength','hitRatio','lagDiff']#I have struck power from the list because our beacons were very powerful---maybe put back if we use KN's training data
    # Do we want to use an informed prior?
    prior = True

    print ("Set Up Complete, Creating Histories")

    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT DISTINCT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND \
        TagType == 'Study' AND tblRaw.FreqCode IS NOT '164.480 25' AND tblRaw.FreqCode IS NOT '164.480 26' AND tblRaw.FreqCode IS NOT '164.290 100';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.values
    c.close()

    print ("There are %s fish to iterate through at site %s" %(len(histories),site))
    # create list of training data objects to iterate over with a Pool multiprocess
    iters = []
    for i in histories:
        iters.append(abtas.classify_data(i,site,fields,projectDB,outputScratch,informed_prior = prior,class_iter=class_iter))

    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")

    pool = mp.Pool(processes = 10)                                               # the number of processes equals the number of processors you have
    pool.map(abtas.calc_class_params_map, iters)                                # map the parameter functions over each training data object 
    print ("Predictors values calculated, proceeding to classification")
     
    abtas.classDatAppend(site,outputScratch,projectDB,class_iter)

    print ("process took %s to compile"%(round(time.time() - tS,3)))
    # get data you just classified, run some statistics and make some plots
    class_stats = abtas.classification_results(recType,projectDB,figure_ws,site)
    class_stats.classify_stats()    

