import sys
sys.path.append(r"C:\a\Projects\Winooski\2020\Data\BIOTAS\biotas")

# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import biotas

#set script parameters
site = '102'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'C:\a\Projects\Winooski\2020\Data'                    # what is the project directory?
dbName = 'Winooski_2020_102.db'                                                   # what is the name of the project database?
t_DBName = 'Winooski_102_Trainer.db'                                                # what is the name of the training database?  We assume it is in the same directory

# optional orion parameters if receivers used switching
scanTime = 2.0
channels = 1
# even if you aren't using switching, fill in this dictionary with the antenna to reciever ID relationship
ant_to_rec_dict = {1:'102'}

# create worskspaces - you haven't changed the directory have you?
trainingDB = os.path.join(proj_dir,'Data',t_DBName)

outputWS = os.path.join(proj_dir,'Output')                                     # we are getting time out error and database locks - so let's write to disk for now
outputScratch = os.path.join(outputWS,'Scratch')                           # we are getting time out error and database locks - so let's write to disk for now
figure_ws = os.path.join(outputWS,'Figures')
workFiles = os.path.join(proj_dir,'Data','Training_Files')
projectDB = os.path.join(proj_dir,'Data',dbName)

# list fields used in likelihood classification, must be from this list:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
files = os.listdir(workFiles)
print ("There are %s files to iterate through"%(len(files)))
tS = time.time()

# if you are using receivers that do not employ antenna switching:
biotas.telemDataImport(site,recType,workFiles,projectDB)

# if you are using recievers that use swtiching:
#biotas.telemDataImport(site,recType,workFiles,projectDB, scanTime = scanTime, channels = channels, ant_to_rec_dict = ant_to_rec_dict)

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

    conn = sqlite3.connect(trainingDB)
    c = conn.cursor()
    sql = "SELECT * FROM tblTrain WHERE recID == '%s';"%(site)
    tblTrainDF = pd.read_sql(sql,con = conn)
    c.close()


    iters = []
    for j in histories:
        iters.append(biotas.classify_data(j,ant_to_rec_dict[i],fields,projectDB,outputScratch,training_data=tblTrainDF,training = trainingDB))
    print ("Finished creating history objects")
    for k in iters:
        biotas.calc_class_params_map(k)
    print ("Detections classified!")
    biotas.classDatAppend(site,outputScratch,projectDB)
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    # generate summary statistics for classification by receiver type
    class_stats = biotas.classification_results(recType,projectDB,figure_ws,rec_list=[ant_to_rec_dict[i]])
    class_stats.classify_stats()

