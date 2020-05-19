# -*- coding: utf-8 -*-
"""
classification stats
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
site = 'T21'                                                               # what is the site/receiver ID?
class_iter= 2 #Enter the iteration number here--start at 2
recType = 'orion'                                                          # what is the receiver type?
proj_dir = r'J:\Jobs\1210\005\Calcs\Studies\3_3_19\2019'                        # what is the project directory?
dbName = 'ultrasound_2019.db'                                                    # what is the name of the project database?
# directory creations
outputWS = os.path.join(proj_dir,'Output')                                 # we are getting time out error and database locks - so let's write to disk for now 
outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now 
workFiles = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')
class_stats = abtas.classification_results(recType,projectDB,figure_ws,site)#,reclass_iter = class_iter)
class_stats.classify_stats()    

