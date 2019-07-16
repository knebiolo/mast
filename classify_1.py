import multiprocessing as mp
import time
import os
import sqlite3
import pandas as pd
import abtas
import warnings
warnings.filterwarnings('ignore')
'''We can run the classification function in serial or parallel over cores.  
This script uses multiprocessing, which is why we call the process behind the 
main statement.'''
if __name__ == "__main__":
    tS = time.time()
    #set script parameters
    site = 't04'                                                               # what is the site/receiver ID?
    recType = 'orion'                                                          # what is the receiver type?
    project_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                # what is the raw data directory
    dbName = 'ultrasound_2018_test.db'                                         # what is the name of the project database
    # directory creations
    scratch_ws = os.path.join(project_dir,'Output','Scratch')  
    figure_ws = os.path.join(project_dir,'Output','Figures')                
    working_files = os.path.join(project_dir,'Data','TrainingFiles')
    projectDB = os.path.join(project_dir,'Data',dbName)
    # list fields used in likelihood classification, must be from this list:
    # ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
    fields = ['conRecLength','hitRatio','power','lagDiff']                     # enter the fields you wish to classify on from list above
    # Do we want to use an informed prior?
    prior = True                                                               # enter whether or not you wish to use an informed prior, if not a 50/50 split is used and the classifier behaves like Maximum Likelihood                                                         
    print ("Set Up Complete, Creating Histories")
    # get the fish to iterate over with SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT DISTINCT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND TagType == 'Study';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.values
    c.close()
    print ("There are %s fish to iterate through at site %s" %(len(histories),site))
    # create list of training data objects to iterate over with a Pool multiprocess
    iters = []
    for i in histories:
        iters.append(abtas.classify_data(i,site,fields,projectDB,scratch_ws,informed_prior = prior))
    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")
    pool = mp.Pool(processes = 3)                                               # the number of processes equals the number of processors you have - 1
    pool.map(abtas.calc_class_params_map, iters)                                # map the parameter functions over each training data object 
    print ("Predictors values calculated, proceeding to classification")
    abtas.classDatAppend(site, scratch_ws, projectDB)
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    # generate summary statistics for classification by receiver type
    class_stats = abtas.classification_results(recType,projectDB,figure_ws,site)
    class_stats.classify_stats()