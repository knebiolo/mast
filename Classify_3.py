# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 10:04:21 2018
Code to allow us to use actual study tags as beacons---work in progress
@author: tcastrosantos
"""


# import modules required for function dependencies
import multiprocessing as mp
import time
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r"D:\a\Projects\Eel Telemetry\Shetucket\Program")
import telemetryAnalysis_5_multiprocessing as telem
import warnings
warnings.filterwarnings('ignore')
print ("Modules Imported")

'''train data - we can run function in serial or multiprocessing over cores in a
computer.  this script uses multiprocessing, which is why we call the process 
behind the main statement.'''
if __name__ == "__main__":
    tS = time.time()
    #set script parameters
    site = '205'                                                                 # what is the site/receiver ID?
    class_iter= 3 #Enter the iteration number here--start at 2
    recType = 'orion'                                                          # what is the receiver type?
    inputWS = r"D:\a\Projects\Eel Telemetry\Shetucket\Data"                             # what is the raw data directory
    dbName = 'Shetucket.db'                                                    # what is the name of the project database
    trainingDB = r"D:\a\Projects\Eel Telemetry\Shetucket\Data\Shetucket.db"
    outputWS = r"D:\a\Projects\Eel Telemetry\Shetucket\Output"                          # we are getting time out error and database locks - so let's write to disk for now 
    outputScratch = r"D:\a\Projects\Eel Telemetry\Shetucket\Output\Scratch"  # we are getting time out error and database locks - so let's write to disk for now 
    workFiles = os.path.join(inputWS,'TrainingFiles')
    projectDB = os.path.join(inputWS,dbName)
    #Now we bring the content from the database into pandas to create a new dataframe
    conn = sqlite3.connect(projectDB, timeout=30.0)
    c = conn.cursor()       
    if class_iter<=2:
        traindf = pd.read_sql("select * from tblTrain",con=conn)#This will read in tblTrain and create a pandas dataframe
        classdf = pd.read_sql("select test, FreqCode,Power,lagB,lagBdiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s"%(site),con=conn)
    else:
        traindf = pd.read_sql("select * from tblTrain%s"%(class_iter-1),con=conn)#This will read in tblTrain and create a pandas dataframe        
        classdf = pd.read_sql("select test, FreqCode,Power,lagB,lagBdiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s_%s"%(site,class_iter-1),con=conn)
    traindf = traindf[traindf.Detection==0]#Get rid of the 'beacons' from the previous round--we will replace these with tags that scored 'true' on the previous classification step
    #classdf = pd.read_sql("select test, FreqCode,Power,lagB,lagBdiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s"%(site),con=conn)
    classdf = classdf[classdf.test==1]    
    classdf['Channels']=np.repeat(1,len(classdf))#This is because there is a field in the tblTrain that has 'Channel'...for us this is always 1 at this time
    classdf.rename(columns={"test":"Detection","fishCount":"FishCount","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
    #
    #Channels,Detection,FreqCode,Power,lagB,lagBdiff,FishCount,conRecLength,miss_to_hit,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,Seconds,fileName,recID,recType,ScanTime
    traindf.columns
    classdf.columns

    #Next we append the classdf to the traindf
    traindf=traindf.append(classdf)
    traindf.to_sql('tblTrain%s'%(class_iter),index=False,con=conn)#we might want to allow for further iterations

    #Now we get back to the same code as Clasify1
    
# list fields used in likelihood classification, must be from this list:
    # ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
    #fields = ['conRecLength','hitRatio','power','lagDiff']
    fields = ['conRecLength','hitRatio','lagDiff']#I have struck power from the list because our beacons were very powerful---maybe put back if we use KN's training data
    # Do we want to use an informed prior?
    prior = True
    
    print ("Set Up Complete, Creating Histories")
    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT DISTINCT tblRaw.FreqCode FROM tblRaw LEFT JOIN tblMasterTag ON tblRaw.FreqCode = tblMasterTag.FreqCode WHERE recID == '%s' AND \
        TagType == 'Study' AND tblRaw.FreqCode IS NOT '164.480 25' AND tblRaw.FreqCode IS NOT '164.480 26' AND tblRaw.FreqCode IS NOT '164.290 100';"%(site)
    histories = pd.read_sql_query(sql,con = conn).FreqCode.values
    c.close()

    print ("There are %s fish to iterate through at site %s" %(len(histories),site))
    
    # create list of training data objects to iterate over with a Pool multiprocess
    iters = []
    for i in histories:
        iters.append(telem.classify_data(i,site,fields,projectDB,outputScratch,informed_prior = prior,class_iter=class_iter))

    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")

    pool = mp.Pool(processes = 10)                                               # the number of processes equals the number of processors you have
    pool.map(telem.calc_class_params_map, iters)                                # map the parameter functions over each training data object 
    print ("Predictors values calculated, proceeding to classification")
     
    telem.classDatAppend(site,outputScratch,projectDB,class_iter)
    
     
    print ("process took %s to compile"%(round(time.time() - tS,3)))
    
    # get data you just classified, run some statistics and make some plots
    # get the fish to iterate through using SQL 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    sql = "SELECT * FROM tblClassify_%s_%s"%(site,class_iter)
    dat = pd.read_sql(sql,con = conn)
    c.close()
    
    dat['PowerBin'] = (dat.Power//10)*10
    # Get the number of true and false positive detections
    det_class_count = dat.groupby('test')['test'].count()
    trueLen = len(dat[dat.test == 1])
    falseLen = len(dat[dat.test == 0])
    print ("The probability that a detection was classified as true was %s"%(round(trueLen/float(det_class_count.sum()),3)))
    print ("The probability that a detection was classified as false was %s"%(round(falseLen/float(det_class_count.sum()),3)))
    print ("The number of detections at receiver %s was %s"%(site,len(dat)))
    # cross tab of series hit and consecutive detection by known detection class
    seriesHit = pd.crosstab(dat.test,dat.seriesHit)
    print (seriesHit)
    consDet = pd.crosstab(dat.test,dat.consDet)
    print (consDet)

    # plot hit ratio histograms by detection class
    hitRatio = sns.FacetGrid(dat,col = "test") 
    bins = np.arange(0,1.1,0.1)
    hitRatio.map(plt.hist, "hitRatio", bins = bins)#, density = True)
    plt.show()
    
    # plot noise ratio histograms by detection class
    noiseRatio = sns.FacetGrid(dat,col = "test") 
    bins = np.arange(0,1.1,0.1)
    noiseRatio.map(plt.hist, "noiseRatio", bins = bins)#, density = True)
    plt.show()
    
    # plot power histograms by detection class
    power = sns.FacetGrid(dat,col = "test") 
    bins = np.linspace(dat.Power.min(),dat.Power.max(),10)
    power.map(plt.hist, "Power", bins = bins)#, density = True)
    plt.show()
    
    # plot power histograms by detection class
    recLength = sns.FacetGrid(dat,col = "test") 
    bins = np.linspace(1,11,11)
    recLength.map(plt.hist, "conRecLength", bins = bins)#, density = True)
    plt.show()
    
    lagBDiff = sns.FacetGrid(dat,col = "test") 
    bins = np.arange(-1000,1000,100)
    lagBDiff.map(plt.hist, "lagBdiff", bins = bins)#, density = True)
    plt.show()
    
    likeRat = sns.FacetGrid(dat,col = "test") 
    dat["LikeRat_TtoF"]=np.log10(dat.postTrue.values/dat.postFalse.values)
    likeRat.map(plt.hist, "LikeRat_TtoF")#, density = True)
    plt.show()
 
    dat[["postTrue","postFalse","LikeRat_TtoF","test"]].head()

