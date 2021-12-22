# BIOTAS

BIO-Telemetry Analysis Software (BIOTAS) for use in removing false positive and overlap detections from radio telemetry projects.

# Introduction
BIOTAS was developed by Kleinschmidt and the USGS to assist researchers with cleaning large-scale radio telemetry datasets.  The software and README were developed with novice Python users in mind.  The sample scripts provided within the read me represent the current best practices.  The end user just has to copy, paste, and edit the appropriate lines into their favorite Python IDE.  It's that simple.  Future research and development will see the creation of a Python Notebook designed to guide the end user through data cleaning steps.  If followed correctly, the resulting dataset represents a standardized, repeatable and defensible product.  Each of the scripts carries out a task one may have when analyzing radio telemetry data, including: creating a standardized project database, importing raw data directly from receiver downloads, identifying and removing false positive detections, cross validating and assessing the quality of training data, producing an array of standard project statistics (no. recaptured at receiver, etc.), removing overlap between neighboring receivers with larger detection zones, and producing data appropriate for analysis with Competing Risks and Mark Recapture (Cormack Jolly Seber and Live Recapture Dead Recover) methods.  In future iterations of the software, we will replace the functional sample scripts with a Jupyter Notebook to guide users through training, classification, and data preparation.

Radio telemetry projects create vast quantities of data, especially those that employ in-water beacon tags.  Therefore, it is advised to have at least 10 GB of hard disk space to store raw, intermediate, and final data products, and at least 8 GB of RAM.  To handle this data, the software creates a SQLite project database.  SQLite is an in-process library that implements a self-contained, server-less, zero configuration, transactional SQL database engine (SQLite, 2017).  More importantly, SQLite can handle simultaneous reads, so it is well-suited for write-once, read-many largescale data analysis projects that employ parallel processing across multiple cores.  To view the SQLite project database download either: [sqlite browser](http://sqlitebrowser.org/) or [sqlite studio](https://sqlitestudio.pl/index.rvt) .  Both SQLite viewers can execute SQL statements, meaning you can create a query and export to csv file from within the database viewer.  They are not that different from Microsoft Access, users familiar with databases will have an easy transition into SQLite.

Starting in 2019, Kleinschmidt Associates (primary developer) will begin hosting the code on Atlassian Bitbucket.  With internal and external collaborators, version control has become an issue and the software is no longer standard among users.  Bitbucket is a web-based version control repository hosting service, which uses the Git system of version control.  It is the preferred system for developers collaborating on proprietary code, because users must be invited to contribute.  The Git distributed version control system tracks and manages changes in source code during development.  This is important when more than 1 user of the open source software adapts it to use on their project.  

The software is written in Python 3.7.x and uses dependencies outside of the standard packages.  Please make sure you have the following modules installed and updated when running telemetry analysis: Numpy, Pandas, Networkx, Matplotlib, Sqlite3, Statsmodels, and SciKit Learn.  The software also uses a number of standard packages including: Time, Math, Os, Datetime, Operator, and Collections.  

The example scripts found in the Read Me will guide the end user through a coplete radio telemetry project.  However, you could import BIOTAS into your own proprietary scripts and data management routines.  These scripts are examples only.

# Project Set Up
The simple 4-line script “project_setup.py” will create a standard file structure and project database in a directory of your choosing.  **It is recommended that the directory does not contain any spaces or special characters.**  For example, if our study was of fish migration in the Connecticut River our initial directory could appear as (if saved to your desktop):
> C:\Users\UserName\Desktop\Connecticut_River_Study

When a directory has been created, insert it into the script in line 2 (highlighted below).  Then, edit line 3 to name the database.  **It is recommended to avoid using spaces in the name of the database.**  Once lines 2 and 4 have been edited, run the script.  

projct_setup.py example:
```
import biotas
proj_dir = r'D:\Manuscript\CT_River_2015'
dbName = 'ctr_2015_v3.db'
biotas.createTrainDB(proj_dir, dbName)  
```

Before proceeding to the next step, investigate the folder structure that was created.  The folders should appear as:

* Project_Name
    * Data *for storing raw data and project set up files*
	    * Training Files *stores raw receiver files for the station you are currently working up*
	* Output *for storing figures, modeling files, and scratch data*
	    * Scratch *holds intermediate files that are managed by the software* - **never delete the contents of this folder**
		* Figures *holds figures exported by the software*

**Do not alter the structure of the directory.** Both the sample scripts provided and BIOTAS expects that the directory is structured exactly as created.

# Initializing the Project Database
Once the project directory has been created, we will initialize the project database by declaring parameters and importing formatted data files that describe the tags and receivers used in the study.  If one of the objectives of your study is to analyze movement or you have close-spaced receivers and overlap is an issue, then we will also need to create a Node table.  This table describes the telemetry network and helps us visualize the relationships between receivers.  Please save initial database files to the ‘Data’ folder (see directory structure above if you forgot where it was) as comma delimited text files (.csv).  

## Master Tag Table
The tag table must match the following schema and be saved as a comma delimited text file.  Field names must appear exactly as in the first column and data types must match.  Please save the file to the ‘Data’ folder and call it ‘tblMasterTag.csv’.  

| Field      | Data Type |                                      Comment                             |
|------------|-----------|--------------------------------------------------------------------------|
|FreqCode    |String     |(required) combination of radio frequency and tag code.  **must be unique**|
|PIT_ID      |Integer    |(not required) if tagged with RFID tag indicate ID                        |
|PulseRate   |Float      |(required) seconds between tag pulses                                     |
|MortRate    |Float      |(not required) if equipped, seconds between tag pulses if fish has died   |
|CapLoc      |String     |(required) capture location of fish                                       |
|RelLoc      |String     |(required) release location of fish                                       |
|TagType     |String     |(required) either 'Study' or 'Beacon                                      |
|Length      |Integer    |(not required) - mm                                                       |
|Sex         |String     |(not required) - either 'M' or 'F'                                        |
|RelDate     |DateTime   |(required) - Date and time of release                                     |

## Master Receiver Table
The receiver file must contain the following fields and should be saved as a comma delimited text file.  Please see the schema below.  As with the master tag file, please save the master receiver file to the ‘Data’ folder.    A good name for this file is “tblMasterReceiver.csv”.  

| Field      | Data Type |                                      Comment                             |
|------------|-----------|--------------------------------------------------------------------------|
|Name        |String     |(not required) - common name of reciever location,  e.g. 'Norse Farms'    |
|RecType     |String     |(required) - acceptable inputs are either **'lotek'** or **'orion'**      |
|recID       |String     |(required) - alphanumeric ID for receiver, e.g. 'T01'                     |
|Node        |String     |(required) - alphanumeric ID for network node, e.g. 'S01'.                |

## Telemetry network
If one of your objectives is to analyze how fish move through a study area, you will need to create a table that describes relationships between receivers (single receivers or groupings) by identifying the logical pathways that exist between them.  Figure 1 depicts a telemetry network of a recent project completed by Kleinschmidt Associates.  Each point on the picture is either a single telemetry receiver or group of receivers.  These points are known as nodes and represent telemetered river reaches.  The receiver-to-node relationship is mapped in the master receiver table with the ‘Node’ column.  The lines, or edges in Figure 1 depict the relationships between nodes.  Some are double headed while others are one way.  Some edges are one way because it is impossible for a fish to swim up through a hydroelectric turbine.  This type of graph is known as a directed acyclic graph.  For now, we only need to identify the nodes and give them arbitrary XY coordinates.  

![Example Telemetry Network](https://i.ibb.co/zNtHCwS/telem-network-sm.png)

Like the tag and receiver tables, the node table will be saved as a comma delimited text file in the ‘Data’ folder.  The XY coordinate data is meant to produce a schematic of the telemetry study (Figure 1).  **We do not recommend the use of actual XY coordinate data (latitude, longitude) because relationships between near-adjacent nodes may be hard to view and aquatic networks are often sinuous.**  The node table has the following schema below: 

| Field      | Data Type |                                      Comment                             |
|------------|-----------|--------------------------------------------------------------------------|
|Node        |String     |(required) - alphanumeric ID for network node **must be unique**          |
|Reach       |String     |(not required) - common name of reach monitored by node, e.g. 'Cabot Tailrace'|
|X           |Integer    |(required) - arbitrary X coordinate point                                 |
|Y           |Integer    |(required) - arbitrary Y coordinate point                                 |

Note: There may be more than 1 receiver associated with a node.  For example, a fishway may have two entrances, but for the purposes of the study you only have to know if a fish has entered the fishway.  It is logical to group them into a single network node.  Doing so will simplify movement modeling.  The receiver to node relationship is developed in the master receiver table with the node column.  IDs must match between columns for relationships to work.

Once the initial data files have been created and stored in the ‘Data’ folder, we will need to import them into the project database.  We will complete this task with the “project_db_ini.py” script (see below).  You will need to follow these steps after pasting the script into your favorite IDE:

1.	Update the project directory (proj_dir)  
2.	Update the project database name (dbName)
3.	Update the number of detections (+/-) in the Proximate Detection History (det).  **The default is 5.**  
4.	Update the duration used when calculating the noise ratio (duration).  **The default is 1 minute.**  
5.	If you are not assessing movement, and do not have a node table, then comment out any lines dealing with nodes 

example project_db_ini.py:
```
import os
import biotas
import pandas as pd

# declare project directory
proj_dir = r'E:\Manuscript\CT_River_2015'  
# delcare database name
dbName = 'ctr_2015_v2.db'
# create directories using OS tools
data_dir = os.path.join(proj_dir,'Data')                                       
db_dir = os.path.join(proj_dir,'Data',dbName)  
# number of detections (+/-) in the PDH                                
det = 5 
# duration used in noise ratio calculation 
duration = 1   

# import data to Python
tblMasterTag = pd.read_csv(os.path.join(data_dir,'tblMasterTag.csv'))
tblMasterReceiver = pd.read_csv(os.path.join(data_dir,'tblMasterReceiver.csv'))
tblNodes = pd.read_csv(os.path.join(data_dir,'tblNodes.csv'))    

# write data to SQLite
biotas.studyDataImport(tblMasterTag,db_dir,'tblMasterTag')
print ('tblMasterTag imported')
biotas.studyDataImport(tblMasterReceiver,db_dir,'tblMasterReceiver')
print ('tblMasterReceiver imported')
biotas.studyDataImport(tblNodes,db_dir,'tblNodes')                             
print ('tblNodes imported')                                                    
biotas.setAlgorithmParameters(det,duration,db_dir)
print ('tblAlgParams data entry complete, begin importing data and training')
```

# False Positive Removal
Radio telemetry receivers record four types of detections based upon their binary nature; true positives, true negatives, false positives and false negatives (Beeman and Perry, 2012). True positives and true negatives are valid data points that indicate the presence or absence of a tagged fish. A false positive is a detection of a fish’s presence when it is not there, while a false negative is a non-detection of a fish that is there. False negatives arise from a variety of causes including insufficient detection areas, collisions between transmitters, interference from ambient noise, or weak signals (Beeman & Perry, 2012). Inclusion of false negatives may negatively bias statistics as there is no way to know if a fish’s absence from a receiver was because it truly wasn’t there or if it was not recaptured by the receiver. While the probability of false negatives can be quantified from sample data as the probability of detection, quantifying the rate of false positives (type I error) is more problematic (Beeman & Perry, 2012). Inclusion of false positives in a dataset can bias study results in two ways: they can favor survivability through a project by including fish that weren’t there or increase measures of delay when a fish has already passed. There are no statistical approaches that can reduce bias associated with false positives, therefore they must be identified and removed *a priori*. ABTAS identifies and removes false positive detections with a Naïve Bayes classifier and removes recaptures resulting from overlapping detections zones with an algorithm inspired by nested Russian dolls. 

Specifically, Bayes Rule calculates the posterior probability, or the probability of a hypothesis occurring given some information about its present state, and is written with P(θ_i |x_j); where θ_i is the hypothesis (true or false positive) and x_j is observed data. Formally, Bayes Rule is expressed as:

![Bayes](https://i.ibb.co/DCngysX/bayes.png)

Where (x_j│θ_i ) is referred to as the likelihood of the j^thdata occurring given the hypothesis (θ_i); P(θ_i ) is the prior probability of the i^thhypothesis (θ); and P(x_j ) is the marginal likelihood or evidence. In most applications, including this one, the marginal likelihood is ignored as it has no effect on the relative magnitudes of the posterior probability (Stone, 2013). Therefore, there is no need to waste computational effort by calculating the joint probability. We can state that the posterior probability is approximately equal to the prior probability times the likelihood.

The prior probability is estimated by looking at how often each class (true or false positive) occurs in the training dataset.  The likelihood is estimated from the histograms of each predictor in the training dataset given each hypothesis (true or false positive) (Marsland, 2009). For simplicity, continuous variables were discretized into bins.  

In most circumstances, the data (x) are usually vectors of feature values or predictor variables with n levels (x_n). As the dimensionality of x increases, the amount of data within each bin of the histogram shrinks, and it becomes difficult to estimate the posterior probability without more training data (Marsland, 2009).  To solve this, the Naïve Bayes classifier makes a simplifying assumption, that the predictor variables are conditionally independent of each other given the classification (Marsland, 2009). Therefore, the probability of getting a particular string of predictor variable feature values is equal to the product of multiplying all of the individual probabilities together (Marsland, 2009). The likelihood is given with:

![Likelihood](https://i.ibb.co/rQ24Yks/likelihood.png)

Where n is equal to the number of features or predictor variables in x and θ_i is the hypothesis (either true or false positive). The classifier rule for Naïve Bayes is to select the detection class θ_i for which the following computation is maximized: 

![MAP](https://i.ibb.co/Kjhhr2m/map.png)

The detection class θ_jwith the maximum posterior probability classifies every line of data belonging to a study tag into one of two classes: true or false positive. This is known as the maximum a posteriori or MAP hypothesis (Marsland, 2009). 

A Naïve Bayes classifier is nothing more than a database application designed to keep track of which feature gives evidence to which class (Richert & Pedro-Coelho, 2013). However, there are circumstances where a particular feature variable level does not occur for a given detection class in the feature dataset (e.g., false positive detection with very high power and many consecutive hits in series), meaning that the likelihood of that feature given a detection class is zero. Therefore, when multiplied together, the posterior probability is zero, and uninformative. To counteract this, the Naïve Bayes classifier uses add-one smoothing, which simply adds 1 to all histogram counts (Richert & Pedro-Coelho, 2013). The underlying assumption here is that even if the feature value was not seen in the training dataset for a particular detection class, the resultant likelihood is close to zero, which allows for an informative posterior.

To calculate the likelihoods, we need to develop a training dataset that consists of known true and false positive detections. By sacrificing real tags and placing them at strategic locations in the water column throughout the study area, beacon tags give the algorithm information on what a known true positive detection looks like. On the other hand, known false positive detections are generated by the telemetry receivers themselves, and consist of detections coded toward tags that were not present in the list of tags released for the study. 

Following the completion of the study, several predictor features were calculated for each received line of data. Predictor features include a detection history of pulses, the consecutive record hit length, hit ratio, miscode ratio, consecutive detection, detection in series, power, the lag time (first order difference) between successive detections, and the second order difference in time between detections. The detection history consists of a string of 1s and 0s that look forwards and backwards in time from the current detection in series, and identifies whether or not a pulse from that particular tag was detected. For example, if a particular tag had a 3-second burst rate, the algorithm will look forwards and backwards in time 3 seconds, query the entire dataset, and then return 1 if it was detected or 0 if it was not. The algorithm looks forwards and backwards for a user-defined set of detection intervals (det parameter). Consecutive detection length and hit ratio are derived from this detection history. Consecutive detection length simply counts the highest number of detections in series, while hit ratio is the ratio of the count of heard detections to the length of the detection history string below.

| Detection History | Consecutive Record Length |       Hit Ratio        |
|-------------------|---------------------------|------------------------|
|0101010            |1                          |3/7                     |
|0011100            |3                          |3/7                     |

Note that both detection history events are considerably different, but they have the same hit ratio. The hit ratio counts the number of correctly assigned detections to the total number of detections within a user-defined set of time. The hypothesis behind this predictor stipulates that a detection is more likely to be true when there are less missed detections.  A sparse detection history is usually associated with false positive detections. Consecutive detections and detections in series are binary in nature. For consecutive detection to return as true, either the previous detection or next detection must occur within the next pulse (i.e., 3-second interval). Detections in series allow the previous or next detection to occur at intervals greater than the first pulse; however, recaptures need to be in series. For example, if the pulse rate is 3 seconds and the next consecutive detection was missed, series hit would return true if the next recorded transmission occurred on the 6th or 9th second. In other words, the pulse rate must be a factor of the difference in time between the present detection and next detection for a series hit to return true. Power, is hypothesized to be higher for true detections than false positives.   

The lag time between consecutive detections is a powerful predictor.  Detections are more likely to be true if the previous detection occurred exactly 1 pulse width prior.  For example, if a tag had a 3 second pulse width and was detected at 11:59:06 and 11:59:09, the lag would be 3.  False positive detections are associated with large and random lag times between detections.  The second order difference in times between detections is even more powerful.  For example, if a tag with a 3 second pulse width was detected at 11:59:06, 11:59:09, and 11:59:12; the second order difference is 0.  Repeated detections at constant intervals is nearly always associated with true positive detections.  If your project happens to switch between receivers or channels, the first order difference may be larger than the nominal pulse width, but the second order difference for true-positive detections would still be 0.

## Create Training Data
The first part of the Naïve Bayes classifier develops training data, which is created from in-water beacon tag recaptures and known false positive detections (i.e. detections from tags not in master tag list).  **We can only train (and classify) a single site at a time.**  Simply copy and paste the raw data files from a single site into the “Training_Files” directory and follow the steps below to create training data.

 To train the classifier with data collected at this site, copy the following script into your favorite IDE, and follow these steps.
 
1.	Update 'site', this must match with a single receiver (‘recID’) in the master receiver table.  
2.	Update 'recType' with the receive type.  Your current options are either ‘lotek’ or ‘orion’.  
3.	Update 'proj_dir' to identify the project directory, this is the same directory you created in step 1.  
4.	Update 'dbName' with the name of the project database.  
5.	Whether or not the receiver switched between antennas, edit the antenna to receiver dictionary 'ant_to_rec_dict'.
6.	Update the scanTime and channel arguments in the biotas.telemDataImport function to match study parameters.  Note the default value for both arguments is 1

```
# import modules
import time
import os
import sqlite3
import pandas as pd

#%%
# Part 1: Set Script Parameters and Workspaces

# what is the site/receiver ID?
site = 'T27'                                                                   
# what is the receiver type?
recType = 'lotek'                                                              
# what is the project directory?
proj_dir = r'D:\Manuscript\CT_River_2015'                                      
# what did you call the database?
dbName = 'ctr_2015_v2.db'                                                         
# antenna to location, default project set up 1 Antenna, 1 Location, 1 Receiver 
ant_to_rec_dict = {1:site}                                                    

# set up workspaces using OS tools    
file_dir = os.path.join(proj_dir,'Data','Training_Files')
files = os.listdir(file_dir)
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
print ("There are %s files to iterate through"%(len(files)))

#%%
# Part 2: Import Site Data and Train Alogrithm 
tS = time.time()

# Import Data, if the receiver does not switch between antennas scanTime and channels = 1.
# If the receiver switches, scanTime and channels must match study values 
biotas.telemDataImport(site,
                       recType,
                       file_dir,
                       projectDB,
                       scanTime = 1,
                       channels = 1,
                       ant_to_rec_dict = ant_to_rec_dict)

print ("Raw data imported, proceed to training")

for i in ant_to_rec_dict:
    # get the fish to iterate through using SQL
    # this SQL statement allows us to train on study tags
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql ='''SELECT tblRaw.FreqCode FROM tblRaw 
            LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode 
            WHERE recID == '%s' 
            AND TagType IS NOT 'Beacon' 
            AND TagType IS NOT 'Test';'''%(ant_to_rec_dict[i])
    histories = pd.read_sql_query(sql,con = conn).FreqCode.unique()
    c.close()
    
    print ("There are %s fish to iterate through" %(len(histories)))
    print ("Creating training objects for every fish at site %s"%(site))
    
    # create a training data object for each fish and train naive Bayes.
    for j in histories:
        train_dat = biotas.training_data(j,
                                         ant_to_rec_dict[i],
                                         projectDB,
                                         scratch_dir)
        biotas.calc_train_params_map(train_dat)
        print ("training parameters quantified for tag %s"%(j))
    print ("Telemetry Parameters Quantified, appending data to project database")
    # append data and summarize
    biotas.trainDatAppend(scratch_dir,projectDB)
    train_stats = biotas.training_results(recType,
                                          projectDB,
                                          figure_ws,
                                          ant_to_rec_dict[i])
    train_stats.train_stats()
    
print ("process took %s to compile"%(round(time.time() - tS,3)))
```


There are multiple strategies for training data.  In some studies, practioners may sacrifice study tags and place them in strategic locations to simulate what a fish would look like.  This training data provides excellent information on what a known true positive detection looks like.  However, some studies do not have the project budget available to sacrifice study tags. In this case, the study trains on study tags and assumes all records from study tags are true in the first pass.  You will note the SQL statement makes this disctinction.  To train on beacon tags instead, update line SQL to:

```
    sql = '''SELECT tblRaw.FreqCode 
    	     FROM tblRaw 
    	     LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode 
    	     WHERE recID == '%s' AND TagType IS NOT 'Study' AND TagType IS NOT 'Test';'''%(site)
```

## Cross-Validate Training Data

**If the end user creates their own training data, it is advised to assess the accuracy of the training data prior to classification.** BIOTAS uses a k-fold cross validation procedure to assess accuracy. The cross-validation procedure randomly assigns each row of data to one of n folds.  Then, the procedure iterates over each fold, classifying data belonging to the current fold, while the remaining rows serve as training data. Then, the classifications were compared against the known states of the training data and compiled in a cross validation table (Table 5).  BIOTAS calculates the positive predictive value (PPV), negative predictive value (NPV), sensitivity (TPR), and specificity (SPC).  For convention below, TP = true positive, TN = true negative, FP = false positive, FN = false negative.

The sensitivity is the probability of the algorithm classifying a record as true positive correctly, and is given with:

>TPR= TP/(TP+FN)

The specificity is the probability of correctly identifying a false positive, and is given with: 

>SPC= TN/(TN+FP)

The PPV and NPV are the proportions of positive and negative results that are true positive and true negative results respectively.  PPV is given with:

>PPV= TP/(TP + FP) 

while NPV is given with:

>NPV= TN/(TN + FN)

Since this algorithm is concerned with identifying and removing false positives, of special interest to researchers are the SPC and NPV.  A low SPC means that the algorithm is not able to discriminate between true and false positive detections.  The lower the SPC, the more false-positive detections are expected in the classified dataset, and the more time is required for manual false positive removal.  A low NPV means that the algorithm is rejecting too many good detections, thus creating instances of false negatives.  In other words, the algorithm is too strict and rejects too many good detections.  A low NPV rate could lead to poor GOF with mark-recapture modeling.  

To run the cross-validation procedure, open ‘cross_validation.py’ and update the following lines:

1.	Update 'recType' with either **lotek** or **orion** receivers. 
2.	Update 'proj_dir' with the project directory
3.	Update 'dbName' with the name of the database 
4.	Update the number of folds 'k'
5.	Update A-La-Carte Likelihood 'fields'
6.	Update the optional receiver list argument 'rec_list' in the biotas.cross_validated function call, if rec_list argument is null all recievers of a type are used.

```
import time
import os
import numpy as np
import biotas

#%%
# Part 1: Set up Cross Validation Script Parameters

# what is the receiver type?
recType = 'orion' 
# what is the project directory?                                                         
proj_dir = r'D:\Manuscript\CT_River_2015'
# what did you call the database?             
dbName = 'ctr_2015_v2.db' 
# use OS tools to create directories                                                      
figureWS = os.path.join(proj_dir,'Output','Figures')    
projectDB = os.path.join(proj_dir,'Data',dbName)
# How many folds in your cross validation?
k = 10

#%%
# Part 2: Cross Validate and Assess Model 
t0 = time.time()

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']

# create cross validated data object
cross = biotas.cross_validated(k,
                               recType,
                               fields,
                               projectDB, 
                               train_on = 'Study',
                               rec_list = ['T03','T06','T24'])
print ("Created a cross validated data object")

# perform the cross validation method
for i in np.arange(0,k,1):
    cross.fold(i)
    print ("Classified fold %s"%i)
# print the summary
cross.summary()
print ("process took %s to compile"%(round(time.time() - t0,3)))
```

## False Positive Classification
There are three classification methods available; the first uses the training data you just created, while the second uses someone else’s training data, and the third method reclassifies an already classified dataset.  

### Classify 1: classifying with data trained on current project.  

Copy and paste the following script into your favorite IDE and update the following lines: 
1.	Update the receiver ID ('site') 
2.	Update the receiver type ('recType') – input can either be **lotek** or **orion** 
3.	Update input workspace ('proj_dir') 
4.	Update project database name ('dbName') 
5.	Update the A-La-Carte likelihood fileds ('fields')
6.	Update whether or not to use an informed 'prior', default = True

example classify_1.py:
```
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
```
### Classify 2: Classify using training data from another project

In some circumstances, the end user may not have beacon tags that they can sacrifice or enough known false positive detections.  In these cases the end-user can use training data from a previous project or other researcher.  The script for this method is nearly identical to first, with the exception of an extra argument in the function call on line 40.  This argument identifies a separate training database.  

Update these variables and run the following script to Classify using someone else's training data.  

1.	Identify the receiver ID ('site') 
2.	Identify the sites ('t_site') in the training database you would like to use for training data
3.	Identify the receiver type ('recType') – input can either be **lotek** or **orion** 
4.	Identify input workspace ('proj_dir') 
5.	Identify project database name ('dbName') 
6.	Identify the training database name ('t_dbName') - **name of database that contains the training dataset - should be in Data folder**
7.	Identify the A-La-Carte likelihood fiends ('fields') - **must be list-like object**
8.	If reciever switches between antennas  update scane time for each antenna ('scanTime') **default = 1**
9.	If reciever switches between antennas, update with the number of antennas ('channels') **default = 1**
10.	If receiver switches between antennas, indicate all antennas in the antenna to receiver dictionary ('ant_to_rec_dict')
11.	Update whether or not to use an informed prior ('prior') **default = True**

```
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
```
### Classify 3: Iteratively reclassify a site until convergence.

The third classification method reclassifies data until there are no more false positives to remove and/or the log posterior ratio no longer improves.  The classify 3 script and workflow is below:

1.	Update the receiver ID ('site')
2.	Enter the classification iteration number ('class_iter') **start at 2**
3.	Update the receiver type ('recType') – input can either be **lotek** or **orion** 
4.	Update input workspace ('proj_dir') 
5.	Update project database name ('dbName') 
6.	Update the A-La-Carte likelihood fields used in the classification ('fields')
7.	Update informed prior ('prior') **default True**

```
# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import biotas

#%% Part 1: Set Up Script Parameters
tS = time.time()

#set script parameters
class_iter= 4 #Enter the iteration number here--start at 2
# what is the site/receiver ID?
site = 'T14'   
# what is the receiver type?                                                                
recType = 'lotek' 
# what is the project directory?                                                             
proj_dir = r'D:\Manuscript\CT_River_2015'  
# whad did you call the database?                                    
dbName = 'ctr_2015_v2.db'                                                           

# create directories with OS tools
outputWS = os.path.join(proj_dir,'Output')                                  
outputScratch = os.path.join(outputWS,'Scratch')                           
workFiles = os.path.join(proj_dir,'Data','TrainingFiles')
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','hitRatio','lagDiff','power','noiseRatio']

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
    biotas.calc_class_params_map(class_dat)
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


```

## Overlap Removal
The Naïve Bayes Classifier will identify and remove false positive detections.  However, if two receivers are placed near each other, a tag may be heard on more than one at the same time.  When assessing movement, overlap is problematic because the fish cannot be in the two different places at the same time.  BIOTAS employs an overlap reduction method inspired by nested Russian dolls.  The detection ranges on antennas vary greatly, but it was assumed that the regions increase in size from stripped coaxial cable up to large aerial Yagis. An algorithm inspired by nested-Russian Dolls was developed to reduce overlap and discretize positions in time and space within the telemetry network. If a fish can be placed at a receiver with a limited detection zone (stripped coaxial cables or dipole), then it can be removed from the overlapping detection zone (Yagi) if it is also recaptured there. 

Fish will often visit a limited range antenna for a certain amount of time, then leave that detection zone only to return sometime later. This behavior is commonly referred to as a “bout” in the ecological literature (Sibly, Nott, and Fletcher, 1990). Following Sibly, Nott and Fletcher’s method (1990), BIOTAS fits a three-process broken-stick model, which is a piecewise-linear regression with two knots (k=2). The function first calculates the lag between detections for each fish within each discrete detection zone. Then, it bins the lag into 10-second intervals, and counts the number of times a lag-interval occurs within each bin. After log-transforming the counts, BIOTAS fits a three-process broken-stick model using a brute-force procedure that tests every bout-length combination with an ordinary least squares regression. The best piecewise-model is the one that minimizes the total residual error (sum of squares). 

If a bout describes an entire visit to a detection zone, the lag between detections determines where a fish is in its bout.  The lags help us determine if the fish present, or if the fish is milling in and out at edge of the detection zone, or if it has left the zone completely only to return some time later.  The knots of the best piecewise regression determine the length of each bout process. 

The first bout process describes a continuous string of detections indicative of a fish being continuously present, the second bout process describes milling behavior at the edge of a detection zone where lags between detections may be 20 – 30 seconds or more, and the third bout process describes the lags between detections where a fish leaves one detection zone completely for another only to come back sometime later. 

After deriving the bout criteria for each discrete telemetry location, presences were enumerated. BIOTAS assumes that a fish has left a detection zone only to come back at the start of the third process. Therefore, the second knot of the broken-stick model describes the time between detections that signify a new presence at this location. If the lag between detections is equal to or greater than this duration, a fish has left the telemetry location only to return much later. The bout process iterates over every detection, for every fish, at every receiver, applies this logic to each lag time, and then enumerates and describes presences at each location with start and end time statistics. Depending upon the number of fish in the study and receivers identified, the algorithm may take a long time to complete.  

To run ‘bouts.py’ follow these steps:
1.	Update line 11 with the project directory
2.	Update line 12 with the database name
3.	Update line 18 with the nodes of interest (typically the entire network)

example bouts.py:
```
# import modules
import os
import warnings
import biotas
warnings.filterwarnings('ignore')
# set up script parameters
project_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                         # what is the name of the project database
inputWS = os.path.join(project_dir,'Data')                             
scratchWS = os.path.join(project_dir,'Output','Scratch')
figureWS = os.path.join(project_dir,'Output','Figures')
projectDB = os.path.join(inputWS,dbName)
# which node do you care about?
nodes = ['S00','S01','S02','S03','S04','S05','S06','S09','S10','S11','S12','S13']
bout_len_dict = dict()
for i in nodes:   
    # Step 1, initialize the bout class object
    bout = biotas.bout(i,projectDB,lag_window = 15, time_limit = 3600)
    # Step 2: get fishes - we only care about test = 1 and hitRatio > 0.3
    fishes = bout.fishes    
    print ("Got a list of tags at this node")
    print ("Start Calculating Bout Length for node %s"%(i))
    # Step 3, find the knot if by site - 
    boutLength_s = bout.broken_stick_3behavior(figureWS) 
    bout_len_dict[i] = boutLength_s
    # step 4 enumerate presence at this receiver for each fish
    for j in fishes:
        # Step 5i, enumerate presence at this receiver
        boutLength_s = bout.broken_stick_fish(j)
        print ("Fish %s had a bout length of %s seconds at node %s"%(j,boutLength_s,i))
        bout.presence(j,boutLength_s,projectDB,scratchWS)      
    print ("Completed bout length calculation and enumerating presences for node %s"%(i))    
    # clean up your mess
    del bout
# manage all of that data, there's a lot of it!
biotas.manage_node_presence_data(scratchWS,projectDB)
```

After describing presences at each receiver (time of entrance, time of exit) it is possible to reduce the overlap between receivers that traditionally plague statistical assessments of movement. If we envision overlapping detection zones as a series of nested-Russian Dolls, we can develop a hierarchical data structure that describes these relationships. If a fish is present in a nested antenna while also present in the overlapping antenna, we can remove coincident detections in the overlapping antenna and reduce bias in our statistical models. This hierarchical data structure is known as a directed graph, where nodes are detection zones and the edges describe the hierarchical relationships among them. For this assessment, edges were directed from a larger detection zone towards a smaller. Edges identify the successive neighbors (smaller detection zones) of each parent node (larger detection zone) and are expressed as a tuple (‘parent’: ‘child’). 

Each node on a telemetry network may consist of one or more receivers (Figure 2). We described the hierarchical relationships between nested receivers with a directed graph depicted in Figure 8. Here, the edges between nodes indicate successors, or nodes with successively smaller detection zones. On this figure, we can see what nodes overlap and what nodes are overlapped.  For example, node S01 overlaps S05, S06, S07, S08 and S09, while node S06 only overlaps node S08.  

![Overlap Relationships](https://i.ibb.co/m5kY2PL/overlap.png)

The Russian Doll algorithm iterates over each detection at each node in Figure 8. Then, the algorithm iterates over each presence at each successor node and asks a simple question: Was the fish detected at the child node while it was also detected at the parent node? If the answer is yes, then the detection at the parent node overlaps the detection at the child node. Thus, the algorithm is nothing more than an iterative search over a directed graph that applies a simple Boolean logic statement. However, it is very powerful in its ability to simplify movement and place fish in discrete spatial locations at discrete points in time. 

To run the Russian-Doll, open up overlap.py and follow these steps:
1.	Update Line 11 with the project directory
2.	Update line 12 with the database name
3.	Update line 17 with the nodes that overlap or are overlapped
4.	Update the successor relationships in line 19 as a list of tuples (‘parent node’, ’child node’) – these describe the edge relationship in the directed graph pictured in Figure 8 

example overlap.py
```
'''Script Intent: The Intent of this script is to identify overlapping records 
to remove redundant detections and to positively place a fish.'''
# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
import time
tS = time.time()
# set up script parameters
project_dir = r'C:\Users\Alex Malvezzi\Desktop'
dbName = 'ultrasound_2019.db'          
outputWS = os.path.join(project_dir,'Output','Scratch')
figureWS = os.path.join(project_dir,'Output','Figures')
projectDB = os.path.join(project_dir,'Data',dbName)
# which node do you care about?
nodes = ['S01','S02','S12','S13','S14','S15','S16','S17','S18','S19','S21','S22','S23','S24']
edges = [('S02','S01'),
         ('S02','S01'),
         ('S13','S14'),('S13','S15'),('S13','S12'),('S13','S16'),('S13','S17'),
         ('S14','S15'),('S14','S16'),('S14','S17'),
         ('S16','S13'),('S16','S15'),('S16','S22'),('S16','S18'),('S16','S19'),('S16','S17'),('S16','S14'),('S16','S23'),('S16','S24'),('S16','S21'),
         ('S17','S18'),('S17','S19'),('S17','S21'),('S17','S13'),('S17','S24'),('S17','S14'),('S17','S22'),('S17','S23'),
         ('S18','S16'),('S18','S17'),
         ('S21','S16'),('S21','S17'),
         ('S22','S16'),
         ('S19','S18'),
         ('S22','S21'),('S22','S16'),('S22','S17'),
         ('S23','S16'),('S23','S17'),
         ('S24','S16'),('S24','S17')]
# Step 1, create an overlap object
print ("Start creating overlap data objects - 1 per node")
for i in nodes:
    overlap_dat = biotas.overlap_reduction(i,nodes,edges,projectDB,outputWS,figureWS))
    biotas.russian_doll(overlap_dat)
    print ("Completed overlap analysis for node %s"%(i))

print ("Overlap analysis complete proceed to data management")
# Step 3, Manage Data   
biotas.manage_node_overlap_data(outputWS,projectDB)
print ("Overlap Removal Process complete, took %s seconds to compile"%(round(time.time() - tS,4)))
del iters
```

## Fish History
ABTAS has the ability to look at the movement of a specific fish through the telemetry network over time.  This function, known as a fish history, is critical for false positive removal as it identifies remaining false positive detections and overlap between adjacent receivers.  

To run the fish_history.py script follow these steps:
1.	Update line 8 with your project directory
2.	Update line 9 with the name of your project’s database
3.	Update line 11 with the list of receivers you wish to include, this defaults to all receivers in the study.
4.	Update line 13 with specific fish history options. For an unfiltered history pass filtered = False and overlapping = False.  For a history with Naïve Bayes false positives removed pass filtered = True and overlapping = False.  For a history with Naïve Bayes false positives and overlapping detections removed pass filtered = True and overlapping = True.  
5.	Update line 16 with the FreqCode of a specific fish.

example fish_history.py:
```
'''Module creates a 3d plot to view fish history''' 
# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir =  r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                             # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['t01','t02','t03O','t03L','t04','t05','t06','t07','t08','t09','t10','t11','t13']
# Step 1, create fish history object
fishHistory = biotas.fish_history(projectDB,filtered = False, overlapping = False, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '150.500 160'
fishHistory.fish_plot(fish)
```

The script will produce an interactive 3d plot than you can rotate and examine.  The following figure shows the fish history for a single fish migrating through the telemetry network depicted in Figure 2.  Note that the X and Y coordinates are the arbitrary coordinates provided in the Node table (see: Telemetry network), and the Z coordinates are in hours since first detection.  When all false positives are removed, it is possible to view how a fish moved through a telemetry network, and how long it spent at each node.  Note that the false positive algorithm removed detections at the onset of the study (middle panel), which reduced the time spent within the study area from over 400 hours to 200 hours.  The Naïve Bayes also removed considerable cross chatter and back and forth movements between major river reaches.  Then, the overlap removal algorithm removed considerable overlap between nodes S05, S08, S07 and S08.  This movement appeared as abrupt back and forth movement (i.e. cross-chatter) between overlapping receivers.  From here, you can identify errant detections by hand and classify them as false positive using SQL methods in your project database.

![Data Removal Steps](https://i.ibb.co/q0Nzpc7/reduction.png)

# The Big Merge
The final data management step prior to statistical formatting merges classified data and removes overlapping detection.  To perform this operation, run The Big Merge.

Follow these steps and run the following script:

1. Declare the project directory (line 6).
2. Declare the project database name

Note:  the big merge has a pre-release filter that removes detections from study tags before their release date.  

```
# import modules
import biotas
import os

# declare workspaces
project_dir = r'J:\920\162\Calcs\Data'
dbName = 'Holyoke_2019'          
outputWS = os.path.join(project_dir,'Data')
projectDB = os.path.join(project_dir,'Data',dbName)

# the big merge
biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True)
```

# Statistical Formatting
BIOTAS has functions that can format datasets appropriate for statistical assessment with Competing Risks and Mark Recapture methods.  Prior to running the formatting functions, you must merge the receiver specific recapture tables into a recaptures table consisting to valid detections only with the ABTAS function ‘the_big_merge’.  Arguments for the function are simple, just the project directory and database name are required.  The final recaptures table can be exported for use in your own analysis.  

## Cormack-Jolly-Seber
Mark-recapture survival analysis is typically used to assess passage effectiveness of fish ladders (Beeman and Perry, 2012). It has also been used to assess upstream movement of fish in river systems along the Northeastern United States (Kleinschmidt 2015, 2017, 2018, 2019).  Use of the term “survival” is standard for mark-recapture analysis, which is predominantly used to assess the actual survival of marked animals over time. However, survival in this context simply means successful passage, it does not convey mortality.  Therefore, we will use probability of arrival rather than survival to reduce confusion.  To estimate arrival parameters with radio telemetry under natural or anthropogenic conditions, one must follow individually marked animals through time (Lebreton et al., 1992). However, it is rarely possible to follow all individuals of an initial sample over time (Lebreton et al., 1992) as is evident by varying recapture rates at each telemetry receiver location. Open population mark-recapture models allow for change (emigration and mortality) during the course of a study (Armstrup, McDonald, and Manly, 2005) and can incorporate imperfect recapture histories. The Cormack-Jolly-Seber (CJS) model is based solely on recaptures of marked animals and provides estimates of arrival and capture probabilities only (Armstrup, McDonald, and Manly, 2005).  

To create a dataset suitable for analysis with CJS modeling, you will first need to develop a receiver-to-recapture-occasion relationship.  CJS modeling is simple; the output is the probability of arriving at the next upstream or downstream recapture occasion.  The receiver-to-recapture-occasion relationship simply groups nodes together into logical recapture occasions like upstream-downstream or forebay-tailrace.  CJS recapture occasions can be envisioned as gates within a river system and may represent potential bottlenecks.  

The input to the CJS function is a dictionary that describes a receiver to recapture occasion relationship and list of receivers used in the analysis.  Some receives may represent offshoots from the main river reach and are not used in the analysis.  Some recapture occasions may be a single receiver, while others are up of a group of receivers. 

To run the CJS_data_prep.py function, follow these steps:

1.	Update line 7 with your project directory
2.	Update line 8 with your project database name
3.	Update line 12 with a name for your model.
4.	Update line 14 with your receiver to recapture occasion relationship
5.	Update line 15 with a list of receivers used in the analysis. 

The output of the function is a dataset (.ISP) appropriate for import with MARK or RMARK and produces a 1 in the recapture occasion column if it was recaptured and a 0 if it wasn’t.  

example CJS_data_prep.py:
```
# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                             # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Ouptut')
modelName = "MontagueToSpillway"
# what is the Node to State relationship - use Python dictionary 
receiver_to_recap = {'t01':'R0','t02':'R0','t03O':'R1','t03L':'R1','t04':'R2',
                     't05':'R2','t06':'R2','t07':'R2','t08':'R2','t09':'R3',
                     't10':'R4','t11':'R4','t13':'R5','t12':'R6'}
recList = ['t01','t02','t03O','t03L','t04','t05','t06','t07','t08','t09','t10',
           't11','t12','t13']
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = biotas.cjs_data_prep(recList,receiver_to_recap,projectDB)
print ("Step 1 Completed, Data Class Finished")
# Step 2, Create input file for MARK
cjs.input_file(modelName,outputWS)
print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")
```

## Competing Risks Assessment
A multi-state model is used to understand situations where a tagged animal transitions from one state to the next (Therneau, Crowson, & Atkinson, 2016). A standard survival curve (Kaplan-Meier) can be thought of as a simple multi-state model with two states (alive and dead) and one transition between those two states (Therneau, Crowson, & Atkinson, 2016). For the purpose of radio telemetry projects, these two states can be staging and passing. Competing risks generalize the standard survival analysis of a single endpoint (as described above) into an investigation of multiple first event types (Beyersmann, Allignol, & Schumacher, 2011). Competing risks are the simplest multi-state model, where events are envisioned as transitions between states (Beyersmann, Allignol, & Schumacher, 2011). For competing risks, there is a common initial state for all models (Beyersmann, Allignol, & Schumacher, 2011). For example, with the assessment of time to move either upstream or downstream of the ultrasound array, the common initial state is within the array. When fish move upstream or downstream of the array, they enter an absorbing state. The baseline hazard is measured with the Nelson-Aalen cause specific cumulative incidence function. One can think of the hazard as the probability of experiencing an event (passage) within the next time unit conditional on still being in the initial state (Beyersmann, Allignol, & Schumacher, 2011). 

ABTAS produces counting process style datasets for use in a Competing Risks assessment with the survival package in R.  The time-to-event class object has methods that can produce datasets suitable for construction of Nelson-Aalen cumulative incidence functions and with Cox Proportional-Hazards (CoxPH) regression.  The major difference between the two is that the CoxPH output expands the time series between the start of a trial and the event time is into equal interval bins so that it can be joined with time-dependent covariates in your analysis software of choice.  The default time series interval is 15 minutes.  However, the end user can pass the optional argument ‘bucket_length_min’ to line 21. If 20 is passed, the function will expand time into 20 minute segments and the end user can join each row to a covariate on the ‘FlowPeriod’ column.  
Follow these steps to prepare your data for competing risks analysis:

1.	Update line 7 with the project directory
2.	Update line 8 with the project database name
3.	Update line 15 with the node to state dictionary.  This dictionary groups nodes together into states for Competing Risks multi-state modeling.  
4.	Update line 16 with the list of receivers used in your analysis
5.	If your time-dependent covariates occur on intervals other than 15 minutes pass the following argument to line 21, ‘bucket_length_min = n’ where n is the number of minutes in covariate bin.  

```
# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                            
dbName = 'ultrasound_2018_test.db'                                                   
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output')
o_fileName_cov = "bypass_detail_cov.csv"
o_fileName = "bypass_detail.csv"
# what is the Node to State relationship - use Python dictionary 
node_to_state = {'S01':1,'S02':1,'S03':1,'S04':1,'S05':1,'S06':1,'S07':1,'S08':1,'S09':2,'S10':3,'S11':3,'S12':4}
recList = ['t01','t02','t03O','t03L','t05','t06','t07','t08','t09','t10','t11','t13']
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
tte = biotas.time_to_event(recList,(node_to_state),projectDB)
print ("Step 1 Complete, Data Class Finished")
# Step 2, format data - with covariates
tte.data_prep(os.path.join(outputWS,o_fileName_cov), time_dependent_covariates = True)
print ("Time to Event data formatted for time dependent covariates")
# Step 3, format data - without covariates
tte.data_prep(os.path.join(outputWS,o_fileName))
print ("Time to Event data formated without time dependent covariates")
# Step 4, generate a summary
tte.summary()
print ("Data formatting complete, proceed to R for Time to Event Modeling")
```

# Bibliography

1.  Armstrup, S. C., McDonald, T. L., & Manly, B. F. (2010). Handbook of Capture-Recapture Analysis. Princeton University Press.
2.  Beeman, J. W., & Perry, R. W. (2012). Bias from False-Positive Detections and Strategies for their Removal in Studies Using Telemetry. In N. S. Adams, J. W. Beeman, & J. H. Eiler (Eds.). American Fisheries Society.
3.  Beyersmann, J., Allignol, A., & Schumacher, M. (2011). Competing Risks and Multistate Models with R. New York, NY: Springer.
4.  Lebreton, J.-D., Burnham, K. P., Clobert, J., & Anderson, D. R. (1992). Modeling survival and testing biological hypotheses using marked animals: a unified approach with case studies. Ecological monographs, 62, 67-118.
5.  Marsland, S. (2009). Machine Learning: An Algorithmic Perspective. Boca, Raton, Florida: CRC Press, Taylor and Francis Group.
6.  Richert, W., & Pedro-Coelho, L. (2013). Building Machine-Learning Systems with Python. Birmingham, UK: Packt Publishing.
7.  Sibly, R. M., Nott, H. M., & Fletcher, D. J. (1990). Splitting Behavior into bouts. Animal Behavior.
8.  Stone, J. V. (2013). Bayes ́Rule: A Tutorial Introduction to Bayesian Analysis. Lexington, KY: Sebtel Press.
9.  Therneau, T., Crowson, C., & Atkinson, E. (2016, October). Multi-state models and competing risks. Multi-state models and competing risks. Retrieved from https://cran.r-project.org/web/packages/survival/vignettes/compete.pdf

# Licensing
MIT License

Copyright (c) 2020 Kevin Patrick Nebiolo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.














