# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import biotas

#%% Part 1: Set Up Script Parameters

# what is the site/receiver ID?
site = '102' 
# what is the site/reciever ID in the training database? 
t_site = '102'
# what is the receiver type?                                                                 
recType = 'orion'
# what is the project directory?                                                              
proj_dir = r'C:\a\Projects\Winooski\2020\Data'                    
# what is the name of the project database?
dbName = 'Winooski_2020_102.db'                                                   
# what is the name of the training database?  We assume it is in the same directory
t_DBName = 'Winooski_102_Trainer.db'                                                

# optional orion parameters if receivers used switching
scanTime = 2.0
channels = 1
# Dictionary that stores the Antenna to Receiver relationship
ant_to_rec_dict = {1:'102'}

# create worskspaces - you haven't changed the directory have you?
trainingDB = os.path.join(proj_dir,'Data',t_DBName)

# create some directories using OS tools
outputWS = os.path.join(proj_dir,'Output')                                     
outputScratch = os.path.join(outputWS,'Scratch')                           
figure_ws = os.path.join(outputWS,'Figures')
workFiles = os.path.join(proj_dir,'Data','Training_Files')
projectDB = os.path.join(proj_dir,'Data',dbName)

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['power','lagDiff','hitRatio']            

# Do we want to use an informed prior?
prior = True       
print ("Set Up Complete, Creating Histories")

tS = time.time()

#%% Part 2: Classify Detections using A-La-Carte Likelihood and Summarize
# We first need to import data 
biotas.telemDataImport(site,
                       recType,
                       workFiles,
                       projectDB,
                       scanTime = scanTime,
                       channels = channels,
                       ant_to_rec_dict = ant_to_rec_dict)

# Now we create a training dataset using the separate training database
# it is also possible to pass a list of receivers 
train = biotas.create_training_data(t_site,
                                    trainingDB)

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL
    site = ant_to_rec_dict[i]
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT FreqCode FROM tblRaw WHERE recID == '%s';"%(site)
    histories = pd.read_sql(sql,con = conn)
    tags = pd.read_sql("SELECT FreqCode FROM tblMasterTag WHERE TagType == 'Study'", con = conn)
    histories = histories.merge(right = tags, 
                                left_on = 'FreqCode', 
                                right_on = 'FreqCode').FreqCode.unique()
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

    for j in histories:
        class_dat = biotas.classify_data(i,
                                         site,
                                         fields,
                                         projectDB,
                                         outputScratch,
                                         training_data=train,
                                         informed_prior = prior)
        biotas.calc_class_params_map(class_dat)   
        print ("Fish %s classified"%(i))
    print ("Detections classified!")
    biotas.classDatAppend(site,outputScratch,projectDB)
    
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    
    # generate summary statistics for classification by receiver type
    class_stats = biotas.classification_results(recType,
                                                projectDB,
                                                figure_ws,
                                                rec_list=[ant_to_rec_dict[i]])
    class_stats.classify_stats()

