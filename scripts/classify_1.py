# import modules
import time
import os
import sqlite3
import pandas as pd
import biotas
tS = time.time()

#%% Part 1: Set Up Script Parameters and Create Diretories

# what is the site/receiver ID?
site = 'T27'    
# what is the receiver type?                                                               
recType = 'lotek'  
# what is the project directory?                                                            
proj_dir = r'D:\Manuscript\CT_River_2015'   
# whad did you call the database?                                   
dbName = 'ctr_2015_v2.db'                                                         

# create directories using OS tools
outputWS = os.path.join(proj_dir,'Output') 
scratch_ws = os.path.join(proj_dir,'Output','Scratch')  
figure_ws = os.path.join(proj_dir,'Output','Figures')                
working_files = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['power','lagDiff','hitRatio']            

# Do we want to use an informed prior?
prior = True       
print ("Set Up Complete, Creating Histories")

#%% Part 2: Classify Detections using A-La-Carte likelihood and Summarize  

# get the fish to iterate over with SQL 
conn = sqlite3.connect(projectDB)
c = conn.cursor()
sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
histories = pd.read_sql(sql,con = conn)
tags = pd.read_sql('''SELECT FreqCode, TagType 
                   FROM tblMasterTag 
                   WHERE TagType == 'Study' ''', con = conn)
histories = histories.merge(right = tags, 
                            how = 'left',
                            left_on = 'FreqCode', 
                            right_on = 'FreqCode')
histories = histories[histories.TagType == 'Study'].FreqCode.unique()
c.close()
print ("There are %s fish to iterate through at site %s" %(len(histories),site))

# create training data for this round of classification
train = biotas.create_training_data(site,
                                    projectDB,
                                    rec_list = [site])

# classify data from each tag
for i in histories:
    class_dat = biotas.classify_data(i,
                                     site,
                                     fields,
                                     projectDB,
                                     scratch_ws,
                                     training_data=train,
                                     informed_prior = prior)
    biotas.calc_class_params_map(class_dat)   
    print ("Fish %s classified"%(i))
print ("Detections classified!")
biotas.classDatAppend(site, scratch_ws, projectDB)

print ("process took %s to compile"%(round(time.time() - tS,3)))
class_stats = biotas.classification_results(recType,
                                            projectDB,
                                            figure_ws,
                                            rec_list=[site])
class_stats.classify_stats()