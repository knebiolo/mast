# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 10:04:21 2018
Code to allow us to use actual study tags as beacons---work in progress
@author: tcastrosantos

Edited 08-12-2019 by KPN - develop script for general use 
"""

# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import abtas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()
#set script parameters
class_iter= 3 #Enter the iteration number here--start at 2
site = 'T04'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'\\EGRET\Condor\Jobs\1210\005\Calcs\Studies\3_3_19\Manuscript'                   # what is the project directory?
dbName = 'algorithm_manuscript.db'                                                   # what is the name of the project database?

# directory creations
outputWS = os.path.join(proj_dir,'Output')                                 # we are getting time out error and database locks - so let's write to disk for now 
outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now 
workFiles = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')
# list fields used in likelihood classification, must be from this list:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','hitRatio','lagDiff','power','noiseRatio']
# Do we want to use an informed prior?
prior = True
print ("Set Up Complete, Creating Histories")
# get the fish to iterate through using SQL 
conn = sqlite3.connect(projectDB)
c = conn.cursor()
sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
histories = pd.read_sql(sql,con = conn)
tags = pd.read_sql("SELECT FreqCode FROM tblMasterTag WHERE TagType == 'Study'", con = conn)
histories = histories.merge(right = tags, left_on = 'FreqCode', right_on = 'FreqCode').FreqCode.unique()
c.close()
histories = histories.tolist()

print ("There are %s fish to iterate through at site %s" %(len(histories),site))
print ("This will take a while")
print ("Grab a coffee, call your mother.")
# create list of training data objects to iterate over with a Pool multiprocess
iters = []
for i in histories:
    iters.append(abtas.classify_data(i,site,fields,projectDB,outputScratch,informed_prior = prior,reclass_iter=class_iter))
    print ('History object created for fish %s'%(i))
print ("History objects created, proceed to classification")
for i in iters:
    abtas.calc_class_params_map(i)
    print ('classified detections for fish %s'%(i))
print ("Detections classified!") 
abtas.classDatAppend(site,outputScratch,projectDB,reclass_iter = class_iter)
print ("process took %s to compile"%(round(time.time() - tS,3)))
# get data you just classified, run some statistics and make some plots
class_stats = abtas.classification_results(recType,projectDB,figure_ws,site,reclass_iter = class_iter)
class_stats.classify_stats()    

