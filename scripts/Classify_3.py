# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import biotas

#%% Part 1: Set Up Script Parameters
tS = time.time()

#set script parameters
class_iter= 2 #Enter the iteration number here--start at 2
# what is the site/receiver ID?
site = 'T4'   
# what is the receiver type?                                                                
recType = 'orion' 
# what is the project directory?                                                             
proj_dir = r'J:\1871\196\Calcs\BIOTAS'  
# whad did you call the database?                                    
dbName = 'pepperell_am_eel.db'                                                           

# create directories with OS tools
outputWS = os.path.join(proj_dir,'Output')                                  
outputScratch = os.path.join(outputWS,'Scratch')                           
workFiles = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','hitRatio','lagDiff','power']

# Do we want to use an informed prior?
prior = False

print ("Set Up Complete, Creating Histories")

#%% Part 2: Classify Detections using A-La-Carte Likelihood and Summarize

# get the fish to iterate through using SQL 
conn = sqlite3.connect(projectDB)
c = conn.cursor()
sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
histories = pd.read_sql(sql,con = conn)
tags = pd.read_sql('''SELECT FreqCode, TagType 
                   FROM tblMasterTag 
                   WHERE TagType == 'Study' ''', con = conn)
histories = histories.merge(right = tags, 
                            left_on = 'FreqCode', 
                            right_on = 'FreqCode')
histories = histories[histories.TagType == 'Study'].FreqCode.unique()
c.close()
histories = histories.tolist()

print ("There are %s fish to iterate through at site %s" %(len(histories),site))

# create training data for this round of classification
train = biotas.create_training_data(site,projectDB, rec_list = [site])

counter = 0
for i in histories:
    counter = counter + 1
    class_dat =biotas.classify_data(i,
                                    site,
                                    fields,
                                    projectDB,
                                    outputScratch,
                                    train,
                                    informed_prior = prior,
                                    reclass_iter=class_iter)
    biotas.classify(class_dat)
    print ('classified detections for fish %s, %s percent complete'%(i,round(counter/len(histories),2)))
print ("Detections classified!") 
biotas.classDatAppend(site,outputScratch,projectDB,reclass_iter = class_iter)
print ("process took %s to compile"%(round(time.time() - tS,3)))

# get data you just classified, run some statistics and make some plots
del train, class_dat

class_stats = biotas.classification_results(recType,
                                            projectDB,
                                            figure_ws,
                                            rec_list=[site],
                                            reclass_iter = class_iter)
class_stats.classify_stats()    

