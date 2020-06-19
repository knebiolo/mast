# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import biotas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()
#set script parameters
class_iter= 2 #Enter the iteration number here--start at 2
site = 'T22'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'E:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015.db'                                                         # whad did you call the database?

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
tags = pd.read_sql("SELECT FreqCode, TagType FROM tblMasterTag WHERE TagType == 'Study'", con = conn)
histories = histories.merge(right = tags, left_on = 'FreqCode', right_on = 'FreqCode')
histories = histories[histories.TagType == 'Study'].FreqCode.unique()
c.close()
histories = histories.tolist()

print ("There are %s fish to iterate through at site %s" %(len(histories),site))
# create list of training data objects to iterate over with a Pool multiprocess

# create training data for this round of classification
train = biotas.create_training_data(site,projectDB)

for i in histories:
    class_dat =biotas.classify_data(i,site,fields,projectDB,outputScratch,train,informed_prior = prior,reclass_iter=class_iter)
    biotas.calc_class_params_map(class_dat)
    print ('classified detections for fish %s'%(i))
print ("Detections classified!") 
biotas.classDatAppend(site,outputScratch,projectDB,reclass_iter = class_iter)
print ("process took %s to compile"%(round(time.time() - tS,3)))
# get data you just classified, run some statistics and make some plots

del train, class_dat

class_stats = biotas.classification_results(recType,projectDB,figure_ws,site,reclass_iter = class_iter)
class_stats.classify_stats()    

