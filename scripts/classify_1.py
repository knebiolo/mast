# import modules
import time
import os
import sqlite3
import pandas as pd
import biotas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()

#set script parameters
site = 'T22'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'E:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015.db'                                                         # whad did you call the database?

# directory creations
outputWS = os.path.join(proj_dir,'Output') 
scratch_ws = os.path.join(proj_dir,'Output','Scratch')  
figure_ws = os.path.join(proj_dir,'Output','Figures')                
working_files = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)

# list fields used in likelihood classification, must be from this list:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','hitRatio','noiseRatio','power','lagDiff']            # enter the fields you wish to classify on from list above

# Do we want to use an informed prior?
prior = True                                                                   # enter whether or not you wish to use an informed prior, if not a 50/50 split is used and the classifier behaves like Maximum Likelihood                                                         
print ("Set Up Complete, Creating Histories")

# get the fish to iterate over with SQL 
conn = sqlite3.connect(projectDB)
c = conn.cursor()
sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
histories = pd.read_sql(sql,con = conn)
tags = pd.read_sql("SELECT FreqCode, TagType FROM tblMasterTag WHERE TagType == 'Study'", con = conn)
histories = histories.merge(right = tags, left_on = 'FreqCode', right_on = 'FreqCode')
histories = histories[histories.TagType == 'Study'].FreqCode.unique()
c.close()
print ("There are %s fish to iterate through at site %s" %(len(histories),site))

# create training data for this round of classification
train = biotas.create_training_data(site,projectDB)

# create a training object and classify each specimen
for i in histories:
    class_dat = biotas.classify_data(i,site,fields,projectDB,scratch_ws,train,informed_prior = prior)
    print ("Classification dataset created")
    biotas.calc_class_params_map(class_dat)   
    print ("Fish %s classified"%(i))
print ("Detections classified!")
biotas.classDatAppend(site, scratch_ws, projectDB)

print ("process took %s to compile"%(round(time.time() - tS,3)))
class_stats = biotas.classification_results(recType,projectDB,figure_ws,site)
class_stats.classify_stats()