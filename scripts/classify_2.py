# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import biotas

#set script parameters
site = 'T11'                                                                   # what is the site/receiver ID?
recType = 'lotek'                                                              # what is the receiver type?
proj_dir = r'J:\920\162\Calcs\Data'                    # what is the project directory?
dbName = 'Holyoke_2019_1.db'                                                   # what is the name of the project database?
t_DBName = 'ultrasound_2018_copy.db'                                                # what is the name of the training database?  We assume it is in the same directory

# optional orion parameters if receivers used switching
scanTime = 2.0
channels = 2
# even if you aren't using switching, fill in this dictionary with the antenna to reciever ID relationship
ant_to_rec_dict = {'1':'T11'}

# create worskspaces - you haven't changed the directory have you?                                              
trainingDB = os.path.join(proj_dir,'Data',t_DBName)
outputWS = os.path.join(proj_dir,'Output')                                     # we are getting time out error and database locks - so let's write to disk for now 
outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now 
figure_ws = os.path.join(outputWS,'Figures')
workFiles = os.path.join(proj_dir,'Data','Training_Files')
projectDB = os.path.join(proj_dir,'Data',dbName)

# list fields used in likelihood classification, must be from this list:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['power']
files = os.listdir(workFiles)
print ("There are %s files to iterate through"%(len(files)))
tS = time.time()  

# if you are using lotek receivers or orion receivers that do not employ switching:                                                          
abtas.telemDataImport(site,recType,workFiles,projectDB) 

# if you are using orion recievers that use swtiching:
#abtas.telemDataImport(site,recType,workFiles,projectDB, switch = True, scanTime = scanTime, channels = channels, ant_to_rec_dict = ant_to_rec_dict) 

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL
    site = ant_to_rec_dict[i]
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
    for j in histories:
        iters.append(abtas.classify_data(j,ant_to_rec_dict[i],fields,projectDB,outputScratch,training = trainingDB))
    print ("Finished creating history objects")
    for k in iters:
        abtas.calc_class_params_map(k)     
    print ("Detections classified!")
    abtas.classDatAppend(site,outputScratch,projectDB)   
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    # generate summary statistics for classification by receiver type
    class_stats = abtas.classification_results(recType,projectDB,figure_ws,ant_to_rec_dict[i])
    class_stats.classify_stats()
    
