# import moduces
import time
import os
import sqlite3
import pandas as pd
import abtas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()
#set script parameters
site = 'T04'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'\\EGRET\Condor\Jobs\1210\005\Calcs\Studies\3_3_19\Manuscript'                   # what is the project directory?
dbName = 'algorithm_manuscript.db'                                                   # what is the name of the project database?

# directory creations
outputWS = os.path.join(proj_dir,'Output') 
scratch_ws = os.path.join(proj_dir,'Output','Scratch')  
figure_ws = os.path.join(proj_dir,'Output','Figures')                
working_files = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)
# list fields used in likelihood classification, must be from this list:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','hitRatio','power','lagDiff', 'seriesHit','noiseRatio']             # enter the fields you wish to classify on from list above
# Do we want to use an informed prior?
prior = True                                                                   # enter whether or not you wish to use an informed prior, if not a 50/50 split is used and the classifier behaves like Maximum Likelihood                                                         
print ("Set Up Complete, Creating Histories")
# get the fish to iterate over with SQL 
conn = sqlite3.connect(projectDB)
c = conn.cursor()
sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
histories = pd.read_sql(sql,con = conn)
tags = pd.read_sql("SELECT FreqCode FROM tblMasterTag WHERE TagType == 'Study'", con = conn)
histories = histories.merge(right = tags, left_on = 'FreqCode', right_on = 'FreqCode').FreqCode.unique()
c.close()
print ("There are %s fish to iterate through at site %s" %(len(histories),site))
print ("This will take a while")
print ("Grab a coffee, call your mother.") 
# create list of training data objects to iterate over with a Pool multiprocess
iters = []
for i in histories:
    iters.append(abtas.classify_data(i,site,fields,projectDB,scratch_ws,informed_prior = prior))
print ("Finished creating history objects")
for i in iters:
    abtas.calc_class_params_map(i)   
print ("Detections classified!")
abtas.classDatAppend(site, scratch_ws, projectDB)
print ("process took %s to compile"%(round(time.time() - tS,3)))
# generate summary statistics for classification by receiver type
figure_ws = os.path.join(outputWS, 'Figures')
class_stats = abtas.classification_results(recType,projectDB,figure_ws,site)
class_stats.classify_stats()