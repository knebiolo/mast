# -*- coding: utf-8 -*-
 # Module contains all of the objects and required for analysis of telemetry data

# import modules required for function dependencies
import numpy as np
import pandas as pd
import math
import multiprocessing as mp
import time
import os
import sqlite3
import datetime
import threading as td
import collections
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import operator
#from datetime import datetime
import networkx as nx
from matplotlib import rcParams
font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

def trainDetClass(row,projectDB):
    '''Function searches code list to see if row is a possible detection.
    This function is meant to be applied to every row in Pandas dataframe with
    pd.apply(...)
    
    Function Arguments:
        row = pandas dataframe object row
        typ = receiver type, "Lotek" or "Orion"
        codeList = list of codes used during telemetry project 
    '''
    conn = sqlite3.connect(projectDB, timeout=30.0)
    c = conn.cursor()    
    sql = 'SELECT * FROM tblMasterTag WHERE TagType == "Beacon";'
    codeList = pd.read_sql(sql,con = conn).FreqCode.values
    c.close()    

    if row['FreqCode'] in codeList:
        return 1
    else:
        return 0

def LotekFreqCode(row,channel):
    '''Function converts the Channel and Tag ID Lotek fields to an Orion formatted
    FreqCode field.  
    
    Required inputs are the row and Channel Dictionary'''
    channel = int(row['Channel'])
    Freq = channel[channel]
    code = row['Tag ID']
    FreqCode = str(Freq) + ',' + str(code)
    return FreqCode

def factors(n):
    ''''function to return primes used to quantify the least common multiplier for creation of ship tracks
    see: http://stackoverflow.com/questions/16996217/prime-factorization-list for recipe'''
    pList = []
    for i in np.arange(1, n + 1):
        if n % i == 0:
            pList.append(i)
    return pList
    
def seriesHit (row,status,datObject,data):
    '''seriesHit is a function for returning whether or not a detection on a specific 
    frequency/code is in series with the previous or next detection on that same 
    frequency/code 
        '''       
    lagB = abs(row['lagB'])
    if status == "A" or status == "U" or status == "T":
        pulseRate = datObject.PulseRate
    else:
        pulseRate = datObject.MortRate

    if lagB <= 3600:
        factB = factors(lagB)
        if pulseRate in factB:
            return 1
        else:
            return 0
    else:
        return 0

def lagBack(df):
    if df.backInTime == df.timeStamp:
        lagB = 0
    else:
        lagB = df.backInTime - df.timeStamp
    return lagB/np.timedelta64(1,'s')

def status (row,datObject):
    '''Function determines whether or not a fish has died based upon its pulse rate.
    If the mort rate is a factor of the forward lag than the fish is declared dead.
    If there is no forward lag than the backwards lag is used.
    If there is no forward or backwards lag, then the fish is declared unknown.    
    '''
    pulseRate = datObject.PulseRate
    mortRate = datObject.MortRate
    lagB = row['lagB']
    status = "A"
    if lagB != None:
        factF = factors(lagB)
        if pulseRate in factF:
            status = "A"
        elif mortRate in factF:
            status = "M"
        else:
            status = "U"
    else:
        status = "U"
    return status
                           
def detHist (row,masterTags,det,status,data,projectDB):
    '''detHist is a function for returning a detection history string at a time index.
    The function looks into the future and past to see if the receiver recorded a 
    detection in series.  
    If there is a record in series then the resultant detection is 1, 0 otherwise. 
    
    The function requires:
    A pandas dataframe with the following columns:
    (det): a number of detections to search, passed as an integer;
    (pulseRate): A double indicating the pulse rate of the tag in question;
    (data): A pandas dataframe containing the data in question, assumes a time stamp index; 
    (t): the current time stamp.
    
    The Input dataframe will be in a common data frame format'''

    # do some data management, declare some local variables
    site = row['recID1']                                                    # get the current site
    freqCode = row['FreqCode']                                                 # get the current frequency - code
    status = status
    rowSeconds = (row['Epoch'])
    scanTime = row['ScanTime']                                             # get the scan time, and number of channels scanned from the current row
    channels = row['Channels']
    if freqCode in masterTags.FreqCode.values:                    # if the frequency code exists in the master code list, get the pulse rate 
        pulses = masterTags[(masterTags.FreqCode == freqCode)]
        pulseRate = float(pulses.PulseRate.values[0])
        mortRate = float(pulses.MortRate.values[0])
    else:
        pulseRate = 2.0                                                    # if it is not in the master code list, give it a default pulse rate of 2.0 seconds
        mortRate = 9.0
    if status == "A" or status == "U" or status == "T":
        rate = pulseRate
    else:
        rate = mortRate
    detHist = str("1")                                                         # start the detection history string by placing 1 to indicate a positive detection at time step i
    
    '''To Do:
    Detection history can benefit from multiprocessing.  Use Python Decorated
    Concurrency, pass dictionary of indexed detection history, return detection 
    history dictionary
    '''
    # start creating the detection history for this row of data
    for i in np.arange(1,det+1):                                                   # for every detection period
        # create window 
        tBackMinus1 = rowSeconds - ((scanTime * channels * i,0) + 0.5)   # number of detection period time steps backward buffered by -1 pulse rate seconds that we will be checking for possible detections in series
        tBackPlus1 = rowSeconds - ((scanTime * channels * i,0) - 0.5 )     # number of detection period time steps backward buffered by +1 pulse rate that we will be checking for possible detections in series                        
        
        dataBack = data[(data.index > tBackMinus1) & (data.index < tBackPlus1)]  # truncate fish data frame records to select detection period recordset
        if len(dataBack) > 0:
            detHist = "1" + detHist            

        else:
            detHist = "0" + detHist
        del dataBack                                                           # delete errrithing
        tForwardMinus1 = rowSeconds + ((scanTime * channels * i,0) - 0.5)    # number of detection period time steps backward buffered by -1 seconds that we will be checking for possible detections in series
        tForwardPlus1 = rowSeconds + ((scanTime * channels * i,0) + 0.5)    # number of detection period time steps backward buffered by +1 seconds that we will be checking for possible detections in series            
        
        dataForward = data[(data.index > tForwardMinus1) & (data.index <  tForwardPlus1)]
        if len(dataForward) > 0:
            detHist = detHist + "1"
        else:
            detHist = detHist + "0"
        del dataForward
    del site                                                                   # clean up, delete errrything
    return detHist

class detection_history():
    '''Python class (detection history object) which is nothing more than a dictionary
    making up the detection history indexed by set detections from the original detection.
    
    Class also contains a method for the creation of a detection history using multithreading.'''
    def __init__(self,data,det,FreqCode,recType,channels,scanTime,rate,Epoch,projectDB):
        self.det = det
        self.FreqCode = FreqCode
        self.recType = recType
        self.channels = float(channels)
        self.scanTime = float(scanTime)
        self.rate = float(rate)
        self.Epoch = float(Epoch)
        self.projectDB = projectDB
        self.data = data
        self.det_hist_dict = {}
        for i in np.arange(self.det * -1, self.det + 1,1): # use this statement if we want to look both forward or backward in time
            self.det_hist_dict[i] = ''

    def histBuilder(self,i):
        if i < 0: # if i is negative, we are looking back in time
            # create window
            if self.channels > 1:
                ll = self.Epoch - ((self.scanTime * self.channels * np.abs(i)) + (self.scanTime))   # number of detection period time steps backward buffered by -1 pulse rate seconds that we will be checking for possible detections in series
                ul = self.Epoch - ((self.scanTime * self.channels * np.abs(i)) - (self.scanTime))     # number of detection period time steps backward buffered by +1 pulse rate that we will be checking for possible detections in series                        
            else: # if there is only 1 channel, the lotek behaves like an Orion
                ll = self.Epoch - ((self.rate * np.abs(i)) + (0.25 * self.rate))   # number of detection period time steps backward buffered by -1 pulse rate seconds that we will be checking for possible detections in series
                ul = self.Epoch - ((self.rate * np.abs(i)) - (0.25 * self.rate))     # number of detection period time steps backward buffered by +1 pulse rate that we will be checking for possible detections in series                                                            
            back = self.data[(self.data.index >= ll) & (self.data.index <= ul)]   
            if len(back) > 0: # if data is not empty
                self.det_hist_dict[i] = "1" 
            else:
                self.det_hist_dict[i] = "0"
            del back, ll, ul
        elif i > 0: # if i is positive, we are looking into the future
            # create window
            if self.channels >1:
                ll = self.Epoch + ((self.scanTime * self.channels * i) - (self.scanTime))    # number of detection period time steps backward buffered by -1 seconds that we will be checking for possible detections in series
                ul = self.Epoch + ((self.scanTime * self.channels * i) + (self.scanTime))    # number of detection period time steps backward buffered by +1 seconds that we will be checking for possible detections in series             
            else: # if there is only 1 chanenl, the lotek behaves like an orion
                ll = self.Epoch + ((self.rate * i) - (0.25 * self.rate))   # number of detection period time steps backward buffered by -1 pulse rate seconds that we will be checking for possible detections in series
                ul = self.Epoch + ((self.rate * i) + (0.25 * self.rate))     # number of detection period time steps backward buffered by +1 pulse rate that we will be checking for possible detections in series                                                                       
            forward = self.data[(self.data.index >= ll) & (self.data.index <= ul)] 
            if len(forward) > 0: # if data is not empty
                self.det_hist_dict[i] = "1" 
            else:
                self.det_hist_dict[i] = "0"
            del forward, ll, ul
        else:
            self.det_hist_dict[i] = "1"        
    
    def build(self):
        threads = []
        for i in np.arange(self.det * -1,self.det + 1,1):
            t = td.Thread(target = self.histBuilder, args = (i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

def detHist_2(row,status,datObject,data):
    '''Function creates a detection history object and returns a detection history string.  
    
    '''    
    # do some data management, declare some local variables                                                    # get the current site
    freqCode = row['FreqCode']                                                 # get the current frequency - code
    status = status
    Epoch = row['Epoch']
    scanTime = row['ScanTime']                                             # get the scan time, and number of channels scanned from the current row
    channels = row['Channels']
    if status == "A" or status == "U" or status == "T":
        rate = datObject.PulseRate
    else:
        rate = datObject.MortRate
        
    # create a detection history instance 
    detHist = detection_history(data,datObject.det,freqCode,datObject.recType,channels,scanTime,rate,Epoch,datObject.projectDB)
    # build a history
    detHist.build() # function uses multithreading, indexes can be out of order, iterate over sorted keys to build a detection history string
    # return a history
    det_hist_str = ''
    for key in sorted(detHist.det_hist_dict.keys()):
        det_hist_str = det_hist_str + detHist.det_hist_dict[key]
               
    return det_hist_str
    
def detHist_3 (status,det,data,masterTags,projectDB):
    '''Detection history 3 is yet another iteration of the detection history function.
    What it lacks in programming elegance, it makes up for speed.  Hopefully.
    
    What if we just shift the epoch column for the set number of detections 
    forward and backward, then perform logic on each column.  Is this shifted epoch
    within an acceptable window of time?      
    '''
            
    det_hist_window_dict = {}                                                   # dictionary key = detection history, value = window tuple (ll,ul)
    for i in np.arange(1,det + 1,1):
        # create forwards and backwards window boundaries
        b_ll = data['Epoch'].astype(float) - ((data['ScanTime'].astype(float) * data['Channels'].astype(float) * i) - 0.5)
        b_ul = data['Epoch'].astype(float) - ((data['ScanTime'].astype(float) * data['Channels'].astype(float) * i) + 0.5)
        f_ll = data['Epoch'].astype(float) + ((data['ScanTime'].astype(float) * data['Channels'].astype(float) * i) - 0.5)
        f_ul = data['Epoch'].astype(float) + ((data['ScanTime'].astype(float) * data['Channels'].astype(float) * i) + 0.5)           
        det_hist_window_dict[i] = (b_ll.values,b_ul.values,f_ll.values,f_ul.values)                                                        # start the detection history string by placing 1 to indicate a positive detection at time step i
    
    shift_dict = {}
    # let's get shifty
    shift_dict[0] = data['Epoch'].values.astype(float) 
    for i in np.arange(1,det + 1,1):
        b_shift = data['Epoch'].shift(i)
        f_shift = data['Epoch'].shift(-1 * i)
        shift_dict[i] = (b_shift.values.astype(float),f_shift.values.astype(float))
    
    # create a base detection history, repeat 1 for number of rows in dataframe - our 0 detection 
    det_hist_dict = dict(zip(np.arange(0,len(data)),np.repeat('1',len(data))))    
        
    # this is the function we are mapping, we feed it an iterator - tuple of input dictionaries - 
    def histBuilder(iters):           
        shift_dict = iters[0]
        det_hist_window_dict = iters[1]
        det_hist_dict = iters[2]
        for i in np.arange(0,len(det_hist_dict)):
            # forward shift 
            # get forward shifts
            f_shift1 = shift_dict[1][1][i]               #(b_shift,f_shift)
            f_shift2 = shift_dict[2][1][i]               #(b_shift,f_shift)
            f_shift3 = shift_dict[3][1][i]               #(b_shift,f_shift)
            f_shift4 = shift_dict[4][1][i]               #(b_shift,f_shift)
            f_shift5 = shift_dict[5][1][i]               #(b_shift,f_shift)    
            # get forward windows
            f_ll1 = det_hist_window_dict[1][2][i]     #(b_ll,b_ul,f_ll,f_ul)
            f_ul1 = det_hist_window_dict[1][3][i]
            f_ll2 = det_hist_window_dict[2][2][i]     #(b_ll,b_ul,f_ll,f_ul)
            f_ul2 = det_hist_window_dict[2][3][i]
            f_ll3 = det_hist_window_dict[3][2][i]     #(b_ll,b_ul,f_ll,f_ul)
            f_ul3 = det_hist_window_dict[3][3][i]
            f_ll4 = det_hist_window_dict[4][2][i]     #(b_ll,b_ul,f_ll,f_ul)
            f_ul4 = det_hist_window_dict[4][3][i]
            f_ll5 = det_hist_window_dict[5][2][i]     #(b_ll,b_ul,f_ll,f_ul)
            f_ul5 = det_hist_window_dict[5][3][i]
        
            # first shift logic
            if f_ll1 <= f_shift1 <= f_ul1:
                det_hist_dict[i] = det_hist_dict[i] + '1'
            else:
                det_hist_dict[i] = det_hist_dict[i] + '0'
                
            # second shift logic
            if f_ll2 <= f_shift1 <= f_ul2 or f_ll2 <= f_shift2 <= f_ul2:
                det_hist_dict[i] = det_hist_dict[i] + '1'
            else:
                det_hist_dict[i] = det_hist_dict[i] + '0'
        
            # third shift logic
            if f_ll3 <= f_shift1 <= f_ul3 or f_ll3 <= f_shift2 <= f_ul3 or f_ll3 <= f_shift3 <= f_ul3:
                det_hist_dict[i] = det_hist_dict[i] + '1'
            else:
                det_hist_dict[i] = det_hist_dict[i] + '0'    
        
            # fourth shift logic
            if f_ll4 <= f_shift1 <= f_ul4 or f_ll4 <= f_shift2 <= f_ul4 or f_ll4 <= f_shift3 <= f_ul4  or f_ll4 <= f_shift4 <= f_ul4:
                det_hist_dict[i] = det_hist_dict[i] + '1'
            else:
                det_hist_dict[i] = det_hist_dict[i] + '0'  
        
            # fourth shift logic
            if f_ll5 <= f_shift1 <= f_ul5 or f_ll5 <= f_shift2 <= f_ul5 or f_ll5 <= f_shift3 <= f_ul5  or f_ll5 <= f_shift4 <= f_ul5   or f_ll5 <= f_shift5 <= f_ul5:
                det_hist_dict[i] = det_hist_dict[i] + '1'
            else:
                det_hist_dict[i] = det_hist_dict[i] + '0'        
            # backward shift
            
            b_shift1 = shift_dict[1][0][i]
            b_shift2 = shift_dict[2][0][i]
            b_shift3 = shift_dict[3][0][i]
            b_shift4 = shift_dict[4][0][i]
            b_shift5 = shift_dict[5][0][i]
        
            b_ll1 = det_hist_window_dict[1][0][i]
            b_ul1 = det_hist_window_dict[1][1][i]
            b_ll2 = det_hist_window_dict[2][0][i]
            b_ul2 = det_hist_window_dict[2][1][i]
            b_ll3 = det_hist_window_dict[3][0][i]
            b_ul3 = det_hist_window_dict[3][1][i]
            b_ll4 = det_hist_window_dict[4][0][i]
            b_ul4 = det_hist_window_dict[4][1][i]
            b_ll5 = det_hist_window_dict[5][0][i]
            b_ul5 = det_hist_window_dict[5][1][i]
                
            # back shift 1 logic
            if b_ll1 <= b_shift1 <= b_ul1:
                det_hist_dict[i] = '1' + det_hist_dict[i]
            else:
                det_hist_dict[i] = '0' + det_hist_dict[i]
                
            # back shift 2 logic
            if b_ll2 <= b_shift1 <= b_ul2 or b_ll2 <= b_shift2 <= b_ul2:
                det_hist_dict[i] = '1' + det_hist_dict[i]
            else:
                det_hist_dict[i] = '0' + det_hist_dict[i]
                
            # back shift 3 logic
            if b_ll3 <= b_shift1 <= b_ul3 or b_ll3 <= b_shift2 <= b_ul3 or b_ll3 <= b_shift3 <= b_ul3:
                det_hist_dict[i] = '1' + det_hist_dict[i]
            else:
                det_hist_dict[i] = '0' + det_hist_dict[i]
        
            # back shift 4 logic
            if b_ll4 <= b_shift1 <= b_ul4 or b_ll4 <= b_shift2 <= b_ul4 or b_ll4 <= b_shift3 <= b_ul4 or b_ll4 <= b_shift4 <= b_ul4:
                det_hist_dict[i] = '1' + det_hist_dict[i]
            else:
                det_hist_dict[i] = '0' + det_hist_dict[i]
        
            # back shift 5 logic
            if b_ll5 <= b_shift1 <= b_ul5 or b_ll5 <= b_shift2 <= b_ul5 or b_ll5 <= b_shift3 <= b_ul5 or b_ll5 <= b_shift4 <= b_ul5 or b_ll5 <= b_shift5 <= b_ul5:
                det_hist_dict[i] = '1' + det_hist_dict[i]
            else:
                det_hist_dict[i] = '0' + det_hist_dict[i]
    
    iters = [[shift_dict,det_hist_window_dict,det_hist_dict]]
    
    def build(iters):
        threads = []
        for i in iters:
            t = td.Thread(target = histBuilder, args = iters)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            
    build(iters)
    
    detHistCol = []
    for key in sorted(det_hist_dict.keys()):
        detHistCol.append(det_hist_dict[key])       
    
    data['detHist_%s'%(status)] = detHistCol
    #data.drop('rate',inplace = True)
    return data                                                             # return the detection history string    

def consDet (row, status, datObject):
    '''consDet is a function for returning whether or not consecutive detections 
    occur 
    '''
       
    if status == "A" or status == "U":
        detHist = row['detHist']
    elif status == "T":
        detHist = row['detHist']
    else:
        detHist = row['detHist']
    left = int(detHist[datObject.det - 1])
    right = int(detHist[datObject.det + 1])
    if left == 1 or right == 1:
        consDet = 1
    else:
        consDet = 0
    return consDet
 
def hitRatio(row,status):
    '''The hit ratio function quantifies the hit ratio, or the number of positive 
    detections within series divided by the total number of records within a 
    detection history string.  
    
    The intent is to see if true detections have a higher ratio than false 
    positives, meaning that there are more hits in series.  Therefore, a false 
    positive detection history string should appear sparse.
    
    101010101 = 5/9
    111110000 = 5/9
    '''
    if status == "A":
        detHist = row['detHist']
    elif status == "T":
        detHist = row['detHist']
    else:
        detHist = row['detHist']
    counter = 0
    arr = np.zeros(len(detHist))
    for i in range(0,len(detHist),1):
        det = int(detHist[i])
        arr[counter] = det
        counter = counter + 1
    hitRatio = float(np.sum(arr))/float(len(arr))
    return hitRatio
   
def conRecLength(row,status):
    ''' This function quantifies the length of consecutive detection in series.
    The only input required is the detection history string.
    
    101010101 = 1
    111110000 = 5
    '''
    if status == "A":
        detHist = row['detHist']
    elif status == "T":
        detHist = row['detHist']
    else:
        detHist = row['detHist']
    maxCount = 0
    count = 0
    for i in range(0,len(detHist),1):
        det = int(detHist[i])
        if det == 1:
            count = count + 1
        else:
            if count > maxCount:
                maxCount = count
            count = 0
        if count > maxCount:
            maxCount = count
    del i
    return maxCount

def miss_to_hit(row,status):
    ''' This function quantifies the length of consecutive detection in series.
    The only input required is the detection history string.
    
    101010101 = 1/1 = 1
    111110000 = 6/5 = 1.2
    '''
    if status == "A":
        detHist = row['detHist']
    elif status == "T":
        detHist = row['detHist']
    else:
        detHist = row['detHist']
    maxHitCount = 0
    maxMissCount = 0
    hitCount = 0
    missCount = 0
    for i in range(0,len(detHist),1):
        det = int(detHist[i])
        if det == 1:
            hitCount = hitCount + 1
        else:
            if hitCount > maxHitCount:
                maxHitCount = hitCount
            hitCount = 0
        if hitCount > maxHitCount:
            maxHitCount = hitCount
    del i
    for i in range(0,len(detHist),1):
        det = int(detHist[i])
        if det == 0:
            missCount = missCount + 1
        else:
            if missCount > maxMissCount:
                maxMissCount = missCount
            missCount = 0
        if missCount > maxMissCount:
            maxMissCount = missCount
    del i
    return round(float(maxMissCount)/float(maxHitCount), 2)
         
def fishCount (row,datObject,data):
    ''' function calculates the number of fish present during the duration time.
    The more fish present, the higher the likelihood of collisions.
    '''
    tags = datObject.studyTags
    duration = datObject.duration            
    # create window    
    rowSeconds = (row['Epoch'])
    ll = rowSeconds - (duration * 60.0)
    ul = rowSeconds + (duration * 60.0) 
    # extract data
    trunc = data[(data.index >= ll) & (data.index <  ul)]                       # truncate the dataset, we only care about these records
    
    # get list of potential fish present 
    fish = 0.0
    for i in trunc.FreqCode.values:                                             # for every row in the truncated dataset                                                     # get the frequency/code
        if i in tags:                                                           # if the code is not in the accepted code list, it's a miscode! egads!
            fish = fish + 1.0
    return fish                                                                # return length of that list 

def noiseRatio (row,datObject,data):
    
    ''' function calculates the ratio of miscoded, pure noise detections, to matching frequency/code 
    detections within the duration specified.
    
    In other words, what is the ratio of miscoded to correctly coded detections within the duration specified
    
    datObject = the current classify_data or training_data object
    data = current data file
    '''
 
    rowSeconds = (row['RowSeconds'])
    ll = rowSeconds - (datObject.duration * 60.0)
    ul = rowSeconds + (datObject.duration * 60.0) 
    # extract data
    trunc = data[(data.index >= ll) & (data.index <  ul)]               # truncate the dataset, we only care about these records
    # calculate noise ratio 
    total = float(len(trunc))                                                   # how many records are there in this dataset?
    miscode = 0.0                                                               # start a miscode counter, 0.0 for float on return
    for i in trunc.FreqCode.values:                                               # for every row in the truncated dataset                                                     # get the frequency/code
        if i in datObject.studyTags:                                                           # if the code is not in the accepted code list, it's a miscode! egads!
            miscode = miscode + 0.0
        else:
            miscode = miscode + 1.0
    return round((miscode/total),4)                                       # the noise ratio is the number of miscoded tags divided by the total number of records in the truncated dataset, which happens to be on a specified interval
    
def MAP (row):
    if row['postTrue'] > row['postFalse']:
        test = "True"
    else:
        test = "False"
    return test
  
def posterior(row,hypothesis):
    if hypothesis == "T":
        arr = np.array([row['priorT'],row['LDetHistT'],row['LPowerT'],row['LHitRatioT'],row['LnoiseT'],row['LconRecT']])
        posterior = np.product(arr)
    else:
        arr = np.array([row['priorF'],row['LDetHistF'],row['LPowerF'],row['LHitRatioF'],row['LnoiseF'],row['LconRecF']])
        posterior = np.product(arr)
    return posterior

def createTrainDB(project_dir, dbName):
    ''' function creates empty project database, user can edit project parameters using 
    DB Broswer for sqlite found at: http://sqlitebrowser.org/'''
    
    # first step creates a project directory if it doesn't already exist 
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    data_dir = os.path.join(project_dir,'Data')                                # raw data goes here
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 
    training_dir = os.path.join(data_dir,'Training_Files')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    output_dir = os.path.join(project_dir, 'Output')                           # intermediate data products, final data products and images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    scratch_dir = os.path.join(output_dir,'Scratch')
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    figures_dir = os.path.join(output_dir, 'Figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)    
    program_dir = os.path.join(project_dir, 'Program')                         # this is where we will create a local clone of the Git repository
    if not os.path.exists(program_dir):
        os.makedirs(program_dir)     
    
    dbDir = os.path.join(data_dir,dbName)
    
    # connect to and create the project geodatabase
    conn = sqlite3.connect(dbDir, timeout=30.0)
    c = conn.cursor()
    # mandatory project tables
    c.execute('''DROP TABLE IF EXISTS tblMasterReceiver''')                    # receiver ID, receiver type
    c.execute('''DROP TABLE IF EXISTS tblMasterTag''')                         # tag ID, frequency, freqcode  
    c.execute('''DROP TABLE IF EXISTS tblReceiverParameters''')                # field crews fuck up, we need these parameters to correctly quantify detection history             
    c.execute('''DROP TABLE IF EXISTS tblAlgParams''')
    c.execute('''DROP TABLE IF EXISTS tblNodes''')
    c.execute('''CREATE TABLE tblMasterReceiver(recID TEXT, Name TEXT, RecType TEXT, Node TEXT)''')
    c.execute('''CREATE TABLE tblReceiverParameters(recID TEXT, RecType TEXT, ScanTime REAL, Channels INTEGER, fileName TEXT)''')
    c.execute('''CREATE TABLE tblMasterTag(FreqCode TEXT, PIT_ID TEXT, PulseRate REAL, MortRate REAL, CapLoc TEXT, RelLoc TEXT, TagType TEXT, Length INTEGER, Sex TEXT, RelDate TIMESTAMP, Study TEXT, Species TEXT)''')
    c.execute('''CREATE TABLE tblAlgParams(det INTEGER, duration INTEGER)''')
    c.execute('''CREATE TABLE tblNodes(Node TEXT, Reach TEXT, RecType TEXT, X INTEGER, Y INTEGER)''')
    
    ''' note these top three tables are mandatory, depending upon how many receivers
    we train and/or use for a study we may not need all of these tables, or we may 
    need more.  This must be addressed in future iterations, can we keep adding empty
    tables at the onset of the project???'''
    
    c.execute('''DROP TABLE IF EXISTS tblRaw''')
    c.execute('''DROP TABLE IF EXISTS tblTrain''')

    c.execute('''CREATE TABLE tblTrain(Channels INTEGER, Detection INTEGER, FreqCode TEXT, Power REAL, lagB INTEGER, lagBdiff REAL, FishCount INTEGER, conRecLength INTEGER, miss_to_hit REAL, consDet INTEGER, detHist TEXT, hitRatio REAL, noiseRatio REAL, seriesHit INTEGER, timeStamp TIMESTAMP, Epoch INTEGER, Seconds INTEGER, fileName TEXT, recID TEXT, recType TEXT, ScanTime REAL)''') # create full radio table - table includes all records, final version will be designed for specific receiver types
    c.execute('''CREATE TABLE tblRaw(timeStamp TIMESTAMP, Epoch INTEGER, FreqCode TEXT, Power REAL, fileName TEXT, recID TEXT)''')
    c.execute('''CREATE INDEX idx_fileNameRaw ON tblRaw (fileName)''')
    c.execute('''CREATE INDEX idx_RecID_Raw ON tblRaw (recID)''')
    c.execute('''CREATE INDEX idx_FreqCode On tblRaw (FreqCode)''')
    c.execute('''CREATE INDEX idx_fileNameTrain ON tblTrain (fileName)''')
    conn.commit()   
    c.close()

def setAlgorithmParameters(det,duration,dbName):
    '''Function sets parameters for predictor variables used in the naive bayes
    classifier
    
    det = number of detections to look forward and backward in times for detection
    history strings
    
    duration = moving window around each detection, used to calculate the noise 
    ratio and number of fish present (fish count)
    
    '''
    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()
    params = [(det,duration)]
    conn.executemany('INSERT INTO tblAlgParams VALUES (?,?)',params)
    conn.commit()
    conn.commit()   
    c.close()     

def studyDataImport(dataFrame,dbName,tblName):
    '''function imports formatted data into project database. The code in its current 
    function does not check for inconsistencies with data structures.  If you're 
    shit isn't right, this isn't going to work for you.  Make sure your table data 
    structures match exactly, that column names and datatypes match.  I'm not your 
    mother, clean up your shit.
    
    dataFrame = pandas dataframe imported from your structured file.  
    dbName = full directory path to project database
    tblName = the name of the data you can import to.  If you are brave, import to 
    tblRaw, but really this is meant for tblMasterTag and tblMasterReceiver'''
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    dataFrame.to_sql(tblName,con = conn,index = False, if_exists = 'append') 
    conn.commit()   
    c.close() 

def orionImport(fileName,dbName,recName):
    '''Function imports raw Sigma Eight orion data.  
    
    Text parser uses simple column fixed column widths.
    
    '''
    
    recType = 'orion'
    scanTime = 1
    channels = 1
    # with our data row, extract information using pandas fwf import procedure
    telemDat = pd.read_fwf(fileName,colspecs = [(0,12),(13,23),(24,30),(31,35),(36,45),(46,54),(55,60),(61,65)],
                            names = ['Date','Time','Site','Ant','Freq','Type','Code','Power'],
                            skiprows = 1,
                            dtype = {'Date':str,'Time':str,'Site':np.int32,'Ant':str,'Freq':str,'Type':str,'Code':str,'Power':np.float64})
    
    telemDat['fileName'] = np.repeat(fileName,len(telemDat))
    telemDat['FreqCode'] = telemDat['Freq'].astype(str) + ' ' + telemDat['Code'].astype(str)
    telemDat = telemDat[telemDat.Type != 'STATUS']
    telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'],errors = 'coerce')# create timestamp field from date and time and apply to index
    telemDat = telemDat[telemDat.timeStamp.notnull()]
    telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()        
    telemDat.drop (['Date','Time','Freq','Code','Ant','Site','Type'],axis = 1, inplace = True)
    telemDat['recID'] = np.repeat(recName,len(telemDat))
    tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
    index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
    telemDat.set_index(index,inplace = True,drop = False)

    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()       
    telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
    recParamLine = [(recName,recType,scanTime,channels,fileName)]
    conn.executemany('INSERT INTO tblReceiverParameters VALUES (?,?,?,?,?)',recParamLine)
    conn.commit() 
    c.close()    
                        
def lotek_import(fileName,dbName,recName):
    ''' function imports raw lotek data, reads header data to find receiver parameters
    and automatically locates raw telemetry data.  Import procedure works with 
    standardized project database. Database must be created before function can be run'''
    
    '''to do: in future iterations create a check for project database, if project 
    data base does not exist, throw an exception
    
    inputs:
    fileName = name of raw telemetry data file with full directory and extenstion
    dbName = name of project database with full directory and extension
    recName = official receiver name'''

    # declare the workspace - in practice we will identify all files in diretory and iterate over them as part of function, all this crap passed as parameters
    recType = 'lotek'
        
    headerDat = {}                                                              # create empty dictionary to hold Lotek header data indexed by line number - to be imported to Pandas dataframe 
    lineCounter = []                                                            # create empty array to hold line indices
    lineList = []                                                               # generate a list of header lines - contains all data we need to write to project set up database
    o_file =open(fileName)                                                      # open file we are currently iterating on
    counter = 0                                                                 # start line counter
    line = o_file.readline()[:-1]                                               # read first line in file
    lineCounter.append(counter)                                                 # append the current line counter to the counter array
    lineList.append(line)                                                       # append the current line of header data to the line list
    
    if line == "SRX800 / 800D Information:":
        # find where data begins and header data ends
        with o_file as f:
            for line in f:              
                if "** Data Segment **" in line:
                    counter = counter + 1
                    dataRow = counter + 5                                               # if this current line signifies the start of the data stream, the data starts three rows down from this 
                    break                                                               # break the loop, we have reached our stop point 
                else:
                    counter = counter + 1                                               # if we are still reading header data increase the line counter by 1
                    lineCounter.append(counter)                                         # append the line counter to the count array 
                    lineList.append(line)                                               # append line of data to the data array 
                        
        headerDat['idx'] = lineCounter                                                  # add count array to dictionary with field name 'idx' as key
        headerDat['line'] = lineList                                                    # add data line array to dictionary with field name 'line' as key
        headerDF = pd.DataFrame.from_dict(headerDat)                                    # create pandas dataframe of header data indexed by row number 
        headerDF.set_index('idx',inplace = True)
        
        # find scan time
        for row in headerDF.iterrows():                                                # for every header data row
            if 'Scan Time' in row[1][0]:                                            # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document 
                scanTimeStr = row[1][0][-7:-1]                                          # get the number value from the row
                scanTimeSplit = scanTimeStr.split(':')                                  # split the string
                scanTime = float(scanTimeSplit[1])                                      # convert the scan time string to float
                break                                                                   # stop that loop, we done
        del row       
        
        # find number of channels and create channel dictionary
        scanChan = []                                                                   # create empty array of channel ID's
        channelDict = {}                                                                # create empty channel ID: frequency dictionary
        counter = 0                                                                     # create counter
        rows = headerDF.iterrows()                                                      # create row iterator
        for row in rows:                                                               # for every row
            if 'Active scan_table:' in row[1][0]:                                   # if the first 18 characters say what that says
                idx0 = counter + 2                                                      # channel dictionary data starts two rows from here
                while rows.next()[1][0] != '\n':                                       # while the next row isn't empty
                    counter = counter + 1                                               # increase the counter, when the row is empty we have reached the end of channels, break loop
                idx1 = counter + 1                                                      # get index of last data row 
                break                                                                   # break that loop, we done
            else:
                counter = counter + 1                                                   # if it isn't a data row, increase the counter by 1
        del row, rows
        channelDat = headerDF.iloc[idx0:idx1]                                           # extract channel dictionary data using rows identified earlier
        
        for row in channelDat.iterrows():
            dat = row[1][0]
            channel = int(dat[0:4])
            frequency = dat[10:17]
            channelDict[channel] = frequency
            scanChan.append(channel)                                              # extract that channel ID from the data row and append to array
        
        channels = len(scanChan)
    
        conn = sqlite3.connect(dbName, timeout=30.0)
        c = conn.cursor()   
        
        # with our data row, extract information using pandas fwf import procedure
        telemDat = pd.read_fwf(fileName,colspecs = [(0,8),(8,18),(18,28),(28,36),(36,51),(51,59)],names = ['Date','Time','ChannelID','TagID','Antenna','Power'],skiprows = dataRow)
        telemDat = telemDat.iloc[:-2]                                                   # remove last two rows, Lotek adds garbage at the end
        telemDat['fileName'] = np.repeat(fileName,len(telemDat))
        def id_to_freq(row,channelDict):
            if row[2] in channelDict:
                return channelDict[row[2]]
            else:
                return '888'
        if len(telemDat) > 0:
            telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
            telemDat = telemDat[telemDat.Frequency != '888']
            telemDat = telemDat[telemDat.TagID != 999]
            telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
            telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'])# create timestamp field from date and time and apply to index
            telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()    
            telemDat.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
            telemDat['fileName'] = np.repeat(fileName,len(telemDat))
            telemDat['recID'] = np.repeat(recName,len(telemDat))
            telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
    else:
        # find where data begins and header data ends
        with o_file as f:
            for line in f:              
                if "********************************* Data Segment *********************************" in line:
                    counter = counter + 1
                    dataRow = counter + 5                                               # if this current line signifies the start of the data stream, the data starts three rows down from this 
                    break                                                               # break the loop, we have reached our stop point 
                else:
                    counter = counter + 1                                               # if we are still reading header data increase the line counter by 1
                    lineCounter.append(counter)                                         # append the line counter to the count array 
                    lineList.append(line)                                               # append line of data to the data array 
                        
        headerDat['idx'] = lineCounter                                                  # add count array to dictionary with field name 'idx' as key
        headerDat['line'] = lineList                                                    # add data line array to dictionary with field name 'line' as key
        headerDF = pd.DataFrame.from_dict(headerDat)                                    # create pandas dataframe of header data indexed by row number 
        headerDF.set_index('idx',inplace = True)
        
        # find scan time
        for row in headerDF.iterrows():                                                # for every header data row
            if 'scan time' in row[1][0]:                                            # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document 
                scanTimeStr = row[1][0][-7:-1]                                          # get the number value from the row
                scanTimeSplit = scanTimeStr.split(':')                                  # split the string
                scanTime = float(scanTimeSplit[1])                                      # convert the scan time string to float
                break                                                                   # stop that loop, we done
        del row       
        
        # find number of channels and create channel dictionary
        scanChan = []                                                                   # create empty array of channel ID's
        channelDict = {}                                                                # create empty channel ID: frequency dictionary
        counter = 0                                                                     # create counter
        rows = headerDF.iterrows()                                                      # create row iterator
        for row in rows:                                                               # for every row
            if 'Active scan_table:' in row[1][0]:                                   # if the first 18 characters say what that says
                idx0 = counter + 2                                                      # channel dictionary data starts two rows from here
                while rows.next()[1][0] != '\n':                                       # while the next row isn't empty
                    counter = counter + 1                                               # increase the counter, when the row is empty we have reached the end of channels, break loop
                idx1 = counter + 1                                                      # get index of last data row 
                break                                                                   # break that loop, we done
            else:
                counter = counter + 1                                                   # if it isn't a data row, increase the counter by 1
        del row, rows
        channelDat = headerDF.iloc[idx0:idx1]                                           # extract channel dictionary data using rows identified earlier
        
        for row in channelDat.iterrows():
            dat = row[1][0]
            channel = int(dat[0:4])
            frequency = dat[10:17]
            channelDict[channel] = frequency
            scanChan.append(channel)                                              # extract that channel ID from the data row and append to array
        
        channels = len(scanChan)
    
        conn = sqlite3.connect(dbName, timeout=30.0)
        c = conn.cursor()   
        
        # with our data row, extract information using pandas fwf import procedure
        telemDat = pd.read_fwf(os.path.join(fileName),colspecs = [(0,5),(5,14),(14,20),(20,26),(26,30),(30,36)],names = ['DayNumber','Time','ChannelID','Power','Antenna','TagID'],skiprows = dataRow)
        telemDat = telemDat.iloc[:-2]                                                   # remove last two rows, Lotek adds garbage at the end
        telemDat['fileName'] = np.repeat(fileName,len(telemDat))
        def id_to_freq(row,channelDict):
            if row[2] in channelDict:
                return channelDict[row[2]]
            else:
                return '888'
        if len(telemDat) > 0:
            telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
            telemDat = telemDat[telemDat.Frequency != '888']
            telemDat = telemDat[telemDat.TagID != 999]
            telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
            telemDat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telemDat))
            telemDat['Date'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber'].astype(int), unit='d')
            telemDat['Date'] = telemDat.Date.astype('str')
            telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'])# create timestamp field from date and time and apply to index
            telemDat.drop(['day0','DayNumber'],axis = 1, inplace = True)       
            telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()    
            telemDat.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
            telemDat['fileName'] = np.repeat(fileName,len(telemDat))
            telemDat['recID'] = np.repeat(recName,len(telemDat))
            tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
            index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
            telemDat.set_index(index,inplace = True,drop = False)

        telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
        
    # add receiver parameters to database    
    recParamLine = [(recName,recType,scanTime,channels,fileName)]
    conn.executemany('INSERT INTO tblReceiverParameters VALUES (?,?,?,?,?)',recParamLine)
    conn.commit() 
    c.close()

def telemDataImport(site,recType,file_directory,projectDB):
    tFiles = os.listdir(file_directory)
    for f in tFiles:
        f_dir = os.path.join(file_directory,f)
        if recType == 'lotek':
            lotek_import(f_dir,projectDB,site)
        elif recType == 'orion':
            orionImport(f_dir,projectDB,site)
        else:
            print ("There currently is not an import routine created for this receiver type.  Please try again")
    print ("Raw Telemetry Data Import Completed")

class training_data():
    '''A class object for a training dataframe and related data objects.
    
    This class object creates a training dataframe for animal i at site j.
    
    The class is written in such a manner to take advantage of Python's multiprocessing
    capabilities.     
    '''
    def __init__(self,i,site,projectDB,scratchWS):
        '''when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).  
        '''
        conn = sqlite3.connect(projectDB, timeout=30.0)        
        c = conn.cursor()
        sql = "SELECT FreqCode, Epoch, tblRaw.recID, timeStamp, Power, ScanTime, Channels FROM tblRaw LEFT JOIN tblReceiverParameters ON tblRaw.fileName = tblReceiverParameters.fileName WHERE FreqCode == '%s' AND tblRaw.recID == '%s';"%(i,site)
        self.histDF = pd.read_sql(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        sql = 'SELECT PulseRate,MortRate FROM tblMasterTag WHERE FreqCode == "%s"'%(i)
        rates = pd.read_sql(sql,con = conn)
        sql = 'SELECT FreqCode FROM tblMasterTag' 
        allTags = pd.read_sql(sql,con = conn)
        sql = 'SELECT * FROM tblAlgParams'
        algParams = pd.read_sql(sql,con = conn)
        sql = 'SELECT RecType FROM tblReceiverParameters WHERE RecID == "%s"'%(site)
        recType = pd.read_sql(sql,con = conn)
        c.close()
        # do some data management when importing training dataframe
        self.histDF['recID1'] = np.repeat(site,len(self.histDF))
        self.histDF['timeStamp'] = pd.to_datetime(self.histDF['timeStamp'])
        self.histDF[['Power','Epoch']] = self.histDF[['Power','Epoch']].apply(pd.to_numeric)                  # sometimes we import from SQLite and our number fields are objects, fuck that noise, let's make sure we are good
        self.histDF['RowSeconds'] = self.histDF['Epoch']
        self.histDF.sort_values(by = 'Epoch', inplace = True)
        self.histDF.set_index('Epoch', drop = False, inplace = True)
        self.histDF = self.histDF.drop_duplicates(subset = 'timeStamp')
        # set some object variables
        self.i = i
        self.site = site
        self.projectDB = projectDB
        self.scratchWS = scratchWS
        self.det = algParams.get_value(0,'det') 
        self.duration = float(algParams.get_value(0,'duration')) 
        self.studyTags = allTags.FreqCode.values
        self.recType = recType.get_value(0,'RecType')
        self.histDF['recType'] = np.repeat(self.recType,len(self.histDF))


        # for training data, we know the tag's detection class ahead of time,
        # if the tag is in the study tag list, it is a known detection class, if 
        # it is a beacon tag, it is definite, if it is a study tag, it's plausible 
        if self.i in self.studyTags:
            self.plausible = 1
        else:
            self.plausible = 0
        # get rate
        if len(rates)>0:  
            self.PulseRate = rates.get_value(0,'PulseRate')
            self.MortRate = rates.get_value(0,'MortRate')
        else:
            self.PulseRate = 2.0
            self.MortRate = 11.0
        # get 
        
# we are going to map this function 
def calc_train_params_map(trainee):                                                       # for every Freq/Code combination...    
    '''function to claculate the training parameters using a training data object.
    The function is mappable across processes'''
    # extract data from trainee
    histDF = trainee.histDF
    i = trainee.i
    site = trainee.site
    projectDB = trainee.projectDB
    scratchWS = trainee.scratchWS
    det = trainee.det
    
    # get data
    conn = sqlite3.connect(projectDB, timeout=30.0)
    c = conn.cursor()
    sql = "SELECT FreqCode, Epoch FROM tblRaw WHERE recID == '%s'"%(site)
    allData = pd.read_sql(sql,con = conn, coerce_float  = True)
    c.close()
    allData.sort_values(by = 'Epoch', inplace = True)
    allData.set_index('Epoch', drop = False, inplace = True)   

    # calculate predictors
    histDF['Detection'] = np.repeat(trainee.plausible,len(histDF))
    histDF['lagB'] = histDF.Epoch.diff()                                       # calculate the difference in seconds until the next detection   
    histDF['lagBdiff'] = histDF.lagB.diff()
    histDF.lagBdiff.fillna(999999999,inplace = True)
    histDF['detHist'] = histDF.apply(detHist_2, axis = 1, args = ("T",trainee,histDF))   # calculate detection history
    histDF['consDet'] = histDF.apply(consDet, axis = 1, args = ("T",trainee))  # determine whether or not to previous record or next record is consecutive in series
    histDF['hitRatio'] = histDF.apply(hitRatio,axis =1, args = 'T')            # calculate hit ratio from detection history
    histDF['conRecLength'] = histDF.apply(conRecLength, axis = 1, args = 'T')  # calculate the number of consecutive detections in series
    histDF['miss_to_hit'] = histDF.apply(miss_to_hit, axis = 1, args = 'T')    # calculate the hit to miss length ratio 
    histDF['seriesHit'] = histDF.apply(seriesHit, axis = 1, args = ("T",trainee,histDF))   # determine whether or not a row record is in series 
    histDF['noiseRatio'] = histDF.apply(noiseRatio, axis = 1, args = (trainee,allData))            
    histDF['FishCount'] = histDF.apply(fishCount, axis = 1, args = (trainee,allData))
    histDF.set_index('timeStamp',inplace = True, drop = True) 
    histDF.drop(['recID1','RowSeconds'],axis = 1, inplace = True)
    histDF['Seconds'] = histDF.index.hour * 3600 + histDF.index.minute * 60 + histDF.index.second

    histDF.to_csv(os.path.join(scratchWS,'%s_%s.csv'%(i,site)))
    del allData, histDF
    
    ''' To Do:
        Figure out how to do write concurrency for sqlite databases.  
        If sqlite can't hang, get another database  
    '''
    # conn = sqlite3.connect(projectDB, timeout=30.0)
    # c = conn.cursor()
    # #histDF.to_sql('tblTrain',con = conn,index = False, if_exists = 'append', chunksize = 1000) # until I figure out how to do database write concurrency, this stays undone
    # c.close() 
                
def trainDatAppend(inputWS,projectDB):
    # As soon as I figure out how to do this function is moot.
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f),dtype = {"detHist":str, "detHist":str})
        #dat.drop(['recID1'],axis = 1,inplace = True)
        dat.to_sql('tblTrain',con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
        del dat                
    c.close()

class classify_data():
    '''A class object for a classification dataframe and related data objects.
    
    This class object creates a dataframe for animal i at site j.
    
    The class is written in such a manner to take advantage of Python's multiprocessing
    capabilities.     
    '''
    def __init__(self,i,site,classifyFields,projectDB,scratchWS,informed_prior = True,training = None, reclass_iter = None):
        '''when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).  
        '''
        conn = sqlite3.connect(projectDB, timeout=30.0)        
        c = conn.cursor()
        sql = "SELECT FreqCode, Epoch, tblRaw.recID, timeStamp, Power, ScanTime, Channels, recType FROM tblRaw LEFT JOIN tblReceiverParameters ON tblRaw.fileName = tblReceiverParameters.fileName WHERE FreqCode == '%s' AND tblRaw.recID == '%s';"%(i,site)
        self.histDF = pd.read_sql(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        sql = 'SELECT PulseRate,MortRate FROM tblMasterTag WHERE FreqCode == "%s"'%(i)
        rates = pd.read_sql(sql,con = conn)
        sql = 'SELECT FreqCode FROM tblMasterTag' 
        allTags = pd.read_sql(sql,con = conn)
        sql = 'SELECT * FROM tblAlgParams'
        algParams = pd.read_sql(sql,con = conn)
        sql = 'SELECT RecType FROM tblReceiverParameters WHERE RecID == "%s"'%(site)
        recType = pd.read_sql(sql,con = conn)
        c.close()
        # do some data management when importing training dataframe
        self.histDF['recID1'] = np.repeat(site,len(self.histDF))
        self.histDF['timeStamp'] = pd.to_datetime(self.histDF['timeStamp'])
        self.histDF[['Power','Epoch']] = self.histDF[['Power','Epoch']].apply(pd.to_numeric)                  # sometimes we import from SQLite and our number fields are objects, fuck that noise, let's make sure we are good
        self.histDF['RowSeconds'] = self.histDF['Epoch']
        self.histDF.sort_values(by = 'Epoch', inplace = True)
        self.histDF.set_index('Epoch', drop = False, inplace = True)
        self.histDF = self.histDF.drop_duplicates(subset = 'timeStamp')
        # set some object variables
        self.fields = classifyFields
        self.i = i
        self.site = site
        self.projectDB = projectDB
        self.scratchWS = scratchWS
        self.det = algParams.get_value(0,'det') 
        self.duration = float(algParams.get_value(0,'duration')) 
        self.studyTags = allTags.FreqCode.values
        self.recType = recType.get_value(0,'RecType')
        self.PulseRate = rates.get_value(0,'PulseRate')
        self.MortRate = rates.get_value(0,'MortRate')
        self.informed = informed_prior
        self.reclass_iter = reclass_iter
        if training != None:
            self.trainingDB = training
        else:
            self.trainingDB = projectDB

def likelihood(assumption,classify_object):
    '''calculates likelihood based on true or false assumption and fields provided to classify object'''
    fields = classify_object.fields
    trueFields = {'conRecLength':'LconRecT','consDet':'LconsDetT','hitRatio':'LHitRatioT','noiseRatio':'LnoiseT','seriesHit':'LseriesHitT','power':'LPowerT','lagDiff':'LlagT'}
    falseFields = {'conRecLength':'LconRecF','consDet':'LconsDetF','hitRatio':'LHitRatioF','noiseRatio':'LnoiseF','seriesHit':'LseriesHitF','power':'LPowerF','lagDiff':'LlagF'}

    if assumption == True:
        if len(fields) == 1:
            return classify_object.histDF[trueFields[fields[0]]] 
        elif len(fields) == 2:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]]
        elif len(fields) == 3:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]] * classify_object.histDF[trueFields[fields[2]]]
        elif len(fields) == 4:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]] * classify_object.histDF[trueFields[fields[2]]]  * classify_object.histDF[trueFields[fields[3]]]
        elif len(fields) == 5:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]] * classify_object.histDF[trueFields[fields[2]]]  * classify_object.histDF[trueFields[fields[3]]]  * classify_object.histDF[trueFields[fields[4]]]
        elif len(fields) == 6:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]] * classify_object.histDF[trueFields[fields[2]]]  * classify_object.histDF[trueFields[fields[3]]]  * classify_object.histDF[trueFields[fields[4]]]  * classify_object.histDF[trueFields[fields[5]]]
        elif len(fields) == 7:
            return classify_object.histDF[trueFields[fields[0]]] * classify_object.histDF[trueFields[fields[1]]] * classify_object.histDF[trueFields[fields[2]]]  * classify_object.histDF[trueFields[fields[3]]]  * classify_object.histDF[trueFields[fields[4]]]  * classify_object.histDF[trueFields[fields[5]]]  * classify_object.histDF[trueFields[fields[6]]]

    elif assumption == False:
        if len(fields) == 1:
            return classify_object.histDF[falseFields[fields[0]]] 
        elif len(fields) == 2:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]]
        elif len(fields) == 3:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]] * classify_object.histDF[falseFields[fields[2]]]
        elif len(fields) == 4:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]] * classify_object.histDF[falseFields[fields[2]]]  * classify_object.histDF[falseFields[fields[3]]]
        elif len(fields) == 5:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]] * classify_object.histDF[falseFields[fields[2]]]  * classify_object.histDF[falseFields[fields[3]]]  * classify_object.histDF[falseFields[fields[4]]]
        elif len(fields) == 6:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]] * classify_object.histDF[falseFields[fields[2]]]  * classify_object.histDF[falseFields[fields[3]]]  * classify_object.histDF[falseFields[fields[4]]]  * classify_object.histDF[falseFields[fields[5]]]
        elif len(fields) == 7:
            return classify_object.histDF[falseFields[fields[0]]] * classify_object.histDF[falseFields[fields[1]]] * classify_object.histDF[falseFields[fields[2]]]  * classify_object.histDF[falseFields[fields[3]]]  * classify_object.histDF[falseFields[fields[4]]]  * classify_object.histDF[falseFields[fields[5]]]  * classify_object.histDF[falseFields[fields[6]]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def calc_class_params_map(classify_object):
    '''
    
    '''
    # extract 
    #classDF = classify_object.histDF
    i = classify_object.i
    site = classify_object.site
    projectDB = classify_object.projectDB
    scratchWS = classify_object.scratchWS
    det = classify_object.det
    
    # get data
    conn = sqlite3.connect(projectDB, timeout=30.0)
    c = conn.cursor()
    sql = "SELECT FreqCode, Epoch FROM tblRaw WHERE recID == '%s'"%(site)
    allData = pd.read_sql(sql,con = conn, coerce_float  = True)
    c.close() 
    
    allData.sort_values(by = 'Epoch', inplace = True)
    allData.set_index('Epoch', drop = False, inplace = True)
    
    # calculate parameters
    classify_object.histDF['lagF'] = classify_object.histDF.Epoch.diff()                          # calculate the difference in seconds until the next detection   
    classify_object.histDF['lagB'] = classify_object.histDF.Epoch.diff(-1)
    classify_object.histDF['lagFdiff'] = classify_object.histDF.lagF.diff()
    classify_object.histDF['lagBdiff'] = classify_object.histDF.lagB.diff(-1)   
    classify_object.histDF['seriesHit'] = classify_object.histDF.apply(seriesHit, axis = 1, args = ("A",classify_object,classify_object.histDF))             # determine whether or not a row record is in series 
    classify_object.histDF['detHist'] = classify_object.histDF.apply(detHist_2, axis = 1, args = ("A",classify_object,classify_object.histDF))   # calculate detection history
    classify_object.histDF['consDet'] = classify_object.histDF.apply(consDet, axis = 1, args = ("A",classify_object))                           # determine whether or not to previous record or next record is consecutive in series
    classify_object.histDF['hitRatio'] = classify_object.histDF.apply(hitRatio,axis =1, args = 'A')                                            # calculate hit ratio from detection history
    classify_object.histDF['conRecLength'] = classify_object.histDF.apply(conRecLength, axis = 1, args = 'A')                                  # calculate the number of consecutive detections in series
    classify_object.histDF['noiseRatio'] = classify_object.histDF.apply(noiseRatio, axis = 1, args = (classify_object,allData))   # calculate the noise ratio    
    classify_object.histDF['fishCount'] = classify_object.histDF.apply(fishCount, axis = 1, args = (classify_object,allData))
    classify_object.histDF['powerBin'] = (classify_object.histDF.Power//10)*10
    classify_object.histDF['noiseBin'] = (classify_object.histDF.noiseRatio//.1)*.1
    classify_object.histDF['lagBdiffBin'] = (classify_object.histDF.lagBdiff//10)*.10
    
    #get training data
    conn = sqlite3.connect(classify_object.trainingDB, timeout=30.0)
    c = conn.cursor()
    if classify_object.reclass_iter == None:
        sql = "SELECT * FROM tblTrain WHERE recType == '%s'"%(classify_object.recType)
        trainDF = pd.read_sql(sql,con = conn, coerce_float = True)
    else:
        if classify_object.reclass_iter<=2:
            trainDF = pd.read_sql("select * from tblTrain",con=conn)#This will read in tblTrain and create a pandas dataframe
            classDF = pd.read_sql("select test, FreqCode,Power,lagB,lagBdiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s"%(site),con=conn)
        else:
            trainDF = pd.read_sql("select * from tblTrain",con=conn)#This will read in tblTrain and create a pandas dataframe        
            classDF = pd.read_sql("select test, FreqCode,Power,lagB,lagBdiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s_%s"%(site,classify_object.reclass_iter-1),con=conn)
        trainDF = trainDF[trainDF.Detection==0]
        classDF = classDF[classDF.test==1]    
        classDF['Channels']=np.repeat(1,len(classDF))
        classDF.rename(columns={"test":"Detection","fishCount":"FishCount","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
        #Next we append the classdf to the traindf
        trainDF=trainDF.append(classDF)   
    c.close()
         
    # Update Data Types - they've got to match or the merge doesn't work!!!!
    trainDF.Detection = trainDF.Detection.astype(int)
    trainDF.FreqCode = trainDF.FreqCode.astype(str)
    trainDF['seriesHit'] = trainDF.seriesHit.astype(int)
    trainDF['consDet'] = trainDF.consDet.astype(int)
    trainDF['detHist'] = trainDF.detHist.astype(str)
    trainDF['noiseRatio'] = trainDF.noiseRatio.astype(float).round(4)
    trainDF['conRecLength'] = trainDF.conRecLength.astype(int)
    trainDF['hitRatio'] = trainDF.hitRatio.astype(float).round(4)
    trainDF['powerBin'] = (trainDF.Power//10)*10
    trainDF['noiseBin'] = (trainDF.noiseRatio//.1)*.1
    trainDF['lagBdiffBin'] = (trainDF.lagBdiff//10)*.10
    
    # making sure our classify object data types match
    classify_object.histDF.seriesHit = classify_object.histDF.seriesHit.astype(np.int64)
    classify_object.histDF.consDet = classify_object.histDF.consDet.astype(int)
    classify_object.histDF.detHist = classify_object.histDF.detHist.astype(str)
    classify_object.histDF.conRecLength = classify_object.histDF.conRecLength.astype(int)
    classify_object.histDF.noiseRatio = classify_object.histDF.noiseRatio.astype(float).round(4)
    classify_object.histDF['HT'] = np.repeat(1,len(classify_object.histDF))
    classify_object.histDF['HF'] = np.repeat(0,len(classify_object.histDF))
    classify_object.histDF.hitRatio = classify_object.histDF.hitRatio.astype(float).round(4) 

    # Make a Count of the predictor variables and join to training data frame - For ALIVE Strings
    seriesHitCount = trainDF.groupby(['Detection','seriesHit'])['seriesHit'].count()
    seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitCountT')
    seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HT','seriesHit'], right_on = ['HT','seriesHit'])
    seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitCountT':'seriesHitCountF'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HF','seriesHit'], right_on = ['HF','seriesHit'])

    # count the number of instances of consective detections by detection class and write to data frame   
    consDetCount = trainDF.groupby(['Detection','consDet'])['consDet'].count()
    consDetCount = pd.Series(consDetCount, name = 'consDetCountT')
    consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HT','consDet'], right_on = ['HT','consDet'])
    consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetCountT':'consDetCountF'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HF','consDet'], right_on = ['HF','consDet'])
        
    # count the number of instances of certain detection histories by detection class and write to data frame       
    detHistCount = trainDF.groupby(['Detection','detHist'])['detHist'].count()
    detHistCount = pd.Series(detHistCount, name = 'detHistCountT')
    detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HT','detHist'],right_on =['HT','detHist'])
    detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistCountT':'detHistCountF'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HF','detHist'],right_on =['HF','detHist'])
    
    # count the number of instances of consecutive record lengths by detection class and write to data frame           
    conRecLengthCount = trainDF.groupby(['Detection','conRecLength'])['conRecLength'].count()
    conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthCountT')
    conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HT','conRecLength'], right_on = ['HT','conRecLength'])
    conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthCountT':'conRecLengthCountF'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HF','conRecLength'], right_on = ['HF','conRecLength'])

    # count the number of instances of hit ratios by detection class and write to data frame           
    hitRatioCount = trainDF.groupby(['Detection','hitRatio'])['hitRatio'].count()
    hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioCountT')
    hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HT','hitRatio'], right_on = ['HT','hitRatio'])
    hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioCountT':'hitRatioCountF'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HF','hitRatio'], right_on = ['HF','hitRatio'])
  
    # Power
    powerCount = trainDF.groupby(['Detection','powerBin'])['powerBin'].count()
    powerCount = pd.Series(powerCount, name = 'powerCount_T')
    powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = powerCount, how = u'left', left_on = ['HT','powerBin'], right_on = ['HT','powerBin'])
    powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = powerCount, how = u'left', left_on = ['HF','powerBin'], right_on = ['HF','powerBin'])
    
    # NoiseR$atio
    noiseCount = trainDF.groupby(['Detection','noiseBin'])['noiseBin'].count()
    noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
    noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = noiseCount, how = u'left', left_on = ['HT','noiseBin'], right_on = ['HT','noiseBin'])
    noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = noiseCount, how = u'left', left_on = ['HF','noiseBin'], right_on = ['HF','noiseBin'])

    # Lag Bin
    lagCount = trainDF.groupby(['Detection','lagBdiffBin'])['lagBdiffBin'].count()
    lagCount = pd.Series(lagCount, name = 'lagDiffCount_T')
    lagCount = pd.DataFrame(lagCount).reset_index().rename(columns = {'Detection':'HT'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = lagCount, how = u'left', left_on = ['HT','lagBdiffBin'], right_on = ['HT','lagBdiffBin'])
    lagCount = lagCount.rename(columns = {'HT':'HF','lagDiffCount_T':'lagDiffCount_F'})
    classify_object.histDF = pd.merge(left = classify_object.histDF, right = lagCount, how = u'left', left_on = ['HF','lagBdiffBin'], right_on = ['HF','lagBdiffBin'])

    
    classify_object.histDF.fillna(0.0000001,inplace = True)
    # Calculate Number of True and False Positive Detections in Training Dataset
    try: 
        priorCountT = float(len(trainDF[trainDF.Detection == 1]))
    except KeyError:
        priorCountT = 1.0
    try:
        priorCountF = float(len(trainDF[trainDF.Detection == 0]))
    except KeyError:
        priorCountF = 1.0
    trueCount = priorCountT + 2.0
    falseCount = priorCountF + 2.0
    classify_object.histDF['priorCount_T'] = np.repeat(priorCountT,len(classify_object.histDF))
    classify_object.histDF['priorCount_F'] = np.repeat(priorCountF,len(classify_object.histDF))
    classify_object.histDF['LDenomCount_T'] = np.repeat(trueCount,len(classify_object.histDF))
    classify_object.histDF['LDenomCount_F'] = np.repeat(falseCount,len(classify_object.histDF))
    
        
    # calculation of the probability of a false positive given the data
    classify_object.histDF['priorF'] = round(priorCountF/float(len(trainDF)),5)                    # calculate the prior probability of a false detection from the training dataset
    classify_object.histDF['LconRecF'] = (classify_object.histDF['conRecLengthCountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
    classify_object.histDF['LseriesHitF'] = (classify_object.histDF['seriesHitCountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LconsDetF'] = (classify_object.histDF['consDetCountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LHitRatioF'] = (classify_object.histDF['hitRatioCountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive    
    classify_object.histDF['LPowerF'] = (classify_object.histDF['powerCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LnoiseF'] = (classify_object.histDF['noiseCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive      
    classify_object.histDF['LlagF'] = (classify_object.histDF['lagDiffCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive      

        
    # calculation of the probability of a true detection given the data
    classify_object.histDF['priorT'] = round(priorCountT/float(len(trainDF)),5)                    # calculate the prior probability of a true detection from the training dataset            
    classify_object.histDF['LconRecT'] = (classify_object.histDF['conRecLengthCountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive                           # calculate the posterior probability of a false positive detection given this row's detection history, power bin and noise ratio
    classify_object.histDF['LseriesHitT'] = (classify_object.histDF['seriesHitCountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LconsDetT'] = (classify_object.histDF['consDetCountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LHitRatioT'] = (classify_object.histDF['hitRatioCountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LPowerT'] = (classify_object.histDF['powerCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
    classify_object.histDF['LnoiseT'] = (classify_object.histDF['noiseCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive        
    classify_object.histDF['LlagT'] = (classify_object.histDF['lagDiffCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive      

    # Calculate the likelihood of each hypothesis being true
    classify_object.histDF['LikelihoodTrue'] = likelihood(True,classify_object)
    classify_object.histDF['LikelihoodFalse'] = likelihood(False,classify_object)
    classify_object.histDF['logLikelihoodRatio'] = np.log10(classify_object.histDF.LikelihoodTrue.values/classify_object.histDF.LikelihoodFalse.values)
   
    #classify_object.histDF['LikelihoodTrue_A'] =  classify_object.histDF['LnoiseT'] * classify_object.histDF['LPowerT'] * classify_object.histDF['LHitRatioT_A'] * classify_object.histDF['LconRecT_A'] * classify_object.histDF['LseriesHitT_A'] * classify_object.histDF['LconsDetT_A']
    #classify_object.histDF['LikelihoodFalse_A'] =  classify_object.histDF['LnoiseF'] * classify_object.histDF['LPowerF'] * classify_object.histDF['LHitRatioF_A'] * classify_object.histDF['LconRecF_A'] * classify_object.histDF['LseriesHitF_A'] * classify_object.histDF['LconsDetF_A']
     
    # Calculate the posterior probability of each Hypothesis occuring
    if classify_object.informed == True:
        classify_object.histDF['postTrue'] = classify_object.histDF['priorT'] * classify_object.histDF['LikelihoodTrue']
        classify_object.histDF['postFalse'] = classify_object.histDF['priorF'] * classify_object.histDF['LikelihoodFalse'] 
    else:
        classify_object.histDF['postTrue'] = 0.5 * classify_object.histDF['LikelihoodTrue']
        classify_object.histDF['postFalse'] = 0.5 * classify_object.histDF['LikelihoodFalse']  
    
    # apply the MAP hypothesis
    classify_object.histDF['test'] = classify_object.histDF.apply(MAP,axis =1) 
    
    classify_object.histDF.to_csv(os.path.join(classify_object.scratchWS,"%s.csv"%(classify_object.i)))
    del trainDF, allData

def classDatAppend(site,inputWS,projectDB):
    # As soon as I figure out how to do this function is moot.
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f),dtype = {"detHist":str})
        #dat.drop(['recID1'],axis = 1,inplace = True)
        dat.to_sql('tblClassify_%s'%(site),con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
        del dat                
    c.close()        
                
                                
class cross_validated():
    '''We validate the training data against itself with a cross validated data 
    object. To implement the k-fold cross validation procedure, we simply pass 
    the number of folds and receiver type we wish to validate.  
    '''    
    def __init__(self,folds,recType,likelihood_model,projectDB,figureWS):
        self.folds = folds
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        sql = "SELECT * FROM tblTrain  WHERE recType == '%s';"%(recType)
        self.trainDF = pd.read_sql_query(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        c.close()
        self.k = folds
        self.trainDF.Detection = self.trainDF.Detection.astype(int)
        self.trainDF.FreqCode = self.trainDF.FreqCode.astype(str)
        self.trainDF.seriesHit = self.trainDF.seriesHit.astype(np.int64)
        self.trainDF.consDet = self.trainDF.consDet.astype(int)
        self.trainDF.detHist = self.trainDF.detHist.astype(str)
        self.trainDF.noiseRatio = self.trainDF.noiseRatio.astype(float)
        self.trainDF.conRecLength = self.trainDF.conRecLength.astype(int)
        self.trainDF.hitRatio = self.trainDF.hitRatio.astype(float)
        self.trainDF['powerBin'] = (self.trainDF.Power//10)*10
        self.trainDF['noiseBin'] = (self.trainDF.noiseRatio//.1)*.1
        cols = ['priorF','LDetHistF','LPowerF','LHitRatioF','LnoiseF','LconRecF','postFalse','priorT','LDetHistT','LPowerT','LHitRatioT','LnoiseT','LconRecT','postTrue','test']
        for i in self.trainDF.columns:
            cols.append(i)
        self.histDF = pd.DataFrame(columns = cols)                                                                           # number of folds                                                                        # set number of folds in cross validation
        fSize = round(len(self.trainDF)/self.folds,0)+1
        kList = np.arange(1,11,1)
        kList = np.repeat(kList,fSize)
        self.trainDF['fold'] = np.random.choice(kList,len(self.trainDF),False)
        print ("Folds created for k fold cross validation")
        self.folds = np.arange(0,folds,1)
        self.fields = likelihood_model
        
    def fold(self,i):
        testDat = self.trainDF[self.trainDF.fold == i]
        trainDat = self.trainDF[self.trainDF.fold != i]                                      # create a test dataset that is the current fold
        testDat['HT'] = np.repeat(1,len(testDat))
        testDat['HF'] = np.repeat(0,len(testDat))    
        
        # Make a Count of the predictor variables and join to training data frame - For ALIVE Strings
        
        # Series Hit
        seriesHitCount = trainDat.groupby(['Detection','seriesHit'])['seriesHit'].count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitCountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = testDat.reset_index()
        testDat = pd.merge(left = testDat, right = seriesHitCount, how = u'left',left_on = ['HT','seriesHit'], right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitCountT':'seriesHitCountF'})
        testDat = pd.merge(left = testDat, right = seriesHitCount, how = u'left',left_on = ['HF','seriesHit'], right_on = ['HF','seriesHit'])
        #testDat.drop(['seriesHit_x','seriesHit_y'], axis = 1, inplace = True)
        
        # Consecutive Detections 
        consDetCount = trainDat.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetCountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = consDetCount, how = u'left', left_on = ['HT','consDet'], right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetCountT':'consDetCountF'})
        testDat = pd.merge(left = testDat, right = consDetCount, how = u'left', left_on = ['HF','consDet'], right_on = ['HF','consDet'])
        #testDat.drop(['consDet_x','consDet_y'], axis = 1, inplace = True)
        
        # Detection History  
        detHistCount = trainDat.groupby(['Detection','detHist'])['detHist'].count()
        detHistCount = pd.Series(detHistCount, name = 'detHistCountT')
        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = detHistCount, how = u'left', left_on = ['HT','detHist'],right_on =['HT','detHist'])
        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistCountT':'detHistCountF'})
        testDat = pd.merge(left = testDat, right = detHistCount, how = u'left', left_on = ['HF','detHist'],right_on =['HF','detHist'])
        #testDat.drop(['detHist_x','detHist_y'], axis = 1, inplace = True)
        
        # Consecutive Record Length 
        conRecLengthCount = trainDat.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthCountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = conRecLengthCount, how = u'left', left_on = ['HT','conRecLength'], right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthCountT':'conRecLengthCountF'})
        testDat = pd.merge(left = testDat, right = conRecLengthCount, how = u'left', left_on = ['HF','conRecLength'], right_on = ['HF','conRecLength'])
        #testDat.drop(['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)
        
        # Hit Ratio 
        hitRatioCount = trainDat.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioCountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = hitRatioCount, how = u'left', left_on = ['HT','hitRatio'], right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioCountT':'hitRatioCountF'})
        testDat = pd.merge(left = testDat, right = hitRatioCount, how = u'left', left_on = ['HF','hitRatio'], right_on = ['HF','hitRatio'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)
        
        # Power
        powerCount = trainDat.groupby(['Detection','powerBin'])['powerBin'].count()
        powerCount = pd.Series(powerCount, name = 'powerCount_T')
        powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = powerCount, how = u'left', left_on = ['HT','powerBin'], right_on = ['HT','powerBin'])
        powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
        testDat = pd.merge(left = testDat, right = powerCount, how = u'left', left_on = ['HF','powerBin'], right_on = ['HF','powerBin'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)    
        
        # NoiseR$atio
        noiseCount = trainDat.groupby(['Detection','noiseBin'])['noiseBin'].count()
        noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
        noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
        testDat = pd.merge(left = testDat, right = noiseCount, how = u'left', left_on = ['HT','noiseBin'], right_on = ['HT','noiseBin'])
        noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
        testDat = pd.merge(left = testDat, right = noiseCount, how = u'left', left_on = ['HF','noiseBin'], right_on = ['HF','noiseBin'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)        
    
        testDat = testDat.fillna(0)                                                # Nan gives us heartburn, fill them with zeros
        # Calculate Number of True and False Positive Detections in Training Dataset
        try: 
            priorCountT = float(len(trainDat[trainDat.Detection == 1]))
        except KeyError:
            priorCountT = 1.0
        try:
            priorCountF = float(len(trainDat[trainDat.Detection == 0]))
        except KeyError:
            priorCountF = 1.0
        trueCount = priorCountT + 2.0
        falseCount = priorCountF + 2.0
        testDat['priorCount_T'] = priorCountT
        testDat['priorCount_F'] = priorCountF
        testDat['LDenomCount_T'] = trueCount
        testDat['LDenomCount_F'] = falseCount
    
        # calculation of the probability of a false positive given the data
        testDat['priorF'] = round(priorCountF/float(len(trainDat)),5)                      # calculate the prior probability of a false detection from the training dataset
        testDat['LHitRatioF'] =(testDat['hitRatioCountF'] + 1)/testDat['LDenomCount_F']      # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        testDat['LconRecF'] = (testDat['conRecLengthCountF'] + 1)/testDat['LDenomCount_F'] # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        testDat['LseriesHitF'] = (testDat['seriesHitCountF'] + 1)/testDat['LDenomCount_F'] # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LconsDetF'] = (testDat['consDetCountF'] + 1)/testDat['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LPowerF'] = (testDat['powerCount_F'] + 1)/testDat['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LnoiseF'] = (testDat['noiseCount_F'] + 1)/testDat['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
            
        # calculation of the probability of a true detection given the data
        testDat['priorT'] = round(priorCountT/float(len(trainDat)),5)                      # calculate the prior probability of a true detection from the training dataset            
        testDat['LHitRatioT'] = (testDat['hitRatioCountT'] + 1)/testDat['LDenomCount_T']     # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        testDat['LconRecT'] = (testDat['conRecLengthCountT'] + 1)/testDat['LDenomCount_T'] # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive                           # calculate the posterior probability of a false positive detection given this row's detection history, power bin and noise ratio
        testDat['LseriesHitT'] = (testDat['seriesHitCountT'] + 1)/testDat['LDenomCount_T'] # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LconsDetT'] = (testDat['consDetCountT'] + 1)/testDat['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LPowerT'] = (testDat['powerCount_T'] + 1)/testDat['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        testDat['LnoiseT'] = (testDat['noiseCount_T'] + 1)/testDat['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        
        # Calculate the likelihood of each hypothesis being true   
        #testDat['LikelihoodTrue'] = likelihood(True,self)
        #testDat['LikelihoodFalse'] = likelihood(False,self)
        testDat['LikelihoodTrue'] = testDat['LPowerT'] * testDat['LHitRatioT'] * testDat['LnoiseT'] * testDat['LconRecT'] * testDat['LseriesHitT'] * testDat['LconsDetT']
        testDat['LikelihoodFalse'] = testDat['LPowerF'] * testDat['LHitRatioF'] *  testDat['LnoiseF'] * testDat['LconRecF'] * testDat['LseriesHitF'] * testDat['LconsDetF']
        
        # Calculate the posterior probability of each Hypothesis occuring
        testDat['postTrue'] = testDat['priorT'] * testDat['LikelihoodTrue']
        testDat['postFalse'] = testDat['priorF'] * testDat['LikelihoodFalse']
        testDat['T2F_ratio'] = testDat['postTrue'] / testDat['postFalse']
        # classify detection as true or false based on MAP hypothesis
   
        testDat['test'] = testDat.postTrue > testDat.postFalse        
        self.histDF = self.histDF.append(testDat)                   
        
    def summary(self):
        metrics = pd.crosstab(self.histDF.Detection,self.histDF.test)
        rowSum = metrics.sum(axis = 1)
        colSum = metrics.sum(axis = 0)
        print ("k-Fold cross validation report for reciever type %s"%(self.recType))
        print ("----------------------------------------------------------------")
        print ("There are %s records in the training dataset "%(len(self.trainDF)))
        print ("%s are known true positive detections and" %(len(self.trainDF[self.trainDF.Detection == 1])))
        print ("%s are known false positive detections"%(len(self.trainDF[self.trainDF.Detection == 0])))
        print ("The prior probability that a detection is true positive is: %s"%(round(float(len(self.trainDF[self.trainDF.Detection == 1]))/float(len(self.trainDF)),4)))
        print ("The prior probability that a detection is false positive is: %s"%(round(float(len(self.trainDF[self.trainDF.Detection == 0]))/float(len(self.trainDF)),4)))      
        print ("----------------------------------------------------------------")
        print ("number of folds: %s"%(self.k))
        print ("----------------------------------------------------------------")
        print (" Cross Validation Table:")
        print ("          Classified     Classified")
        print ("             False          True")
        print ("      ______________________________")
        print (" Known|              |              |")
        print (" False| TN:%s  | FP:%s  |"%(format(metrics.iloc[0,0]," 8d"),format(metrics.iloc[0,1]," 8d")))
        print ("      |______________|______________|")
        print (" Known|              |              |")
        print ("  True| FN:%s  | TP:%s  |"%(format(metrics.iloc[1,0]," 8d"),format(metrics.iloc[1,1]," 8d")))
        print ("      |______________|______________|")
        print ("")
        print ("________________________________________________________________")
        print ("Positive Predictive Value: %s"%(round(float(metrics.iloc[1,1])/float(colSum[1]),4)))
        print ("PPV = TP/(TP + FP)")
        print ("Probability of a record being correctly classified as true")
        print ("----------------------------------------------------------------")
        print ("Negative Predictive Value: %s"%(round(float(metrics.iloc[0,0])/float(colSum[0]),4)))
        print ("NPV = TN/(TN + FN)")
        print ("Probability of a record being correctly classified as false")
        print ("----------------------------------------------------------------")
        print ("The sensitivity of the classifier is: %s"%(round(float(metrics.iloc[1,1])/float(rowSum[1]),4)))
        print ("sensitivity = TP /(TP + FN)")
        print ("Probability of a record being classified true,")
        print ("given that the record is in fact true")
        print ("----------------------------------------------------------------")
        print ("The specificity of the classifier is: %s"%(round(float(metrics.iloc[0,0])/float(rowSum[0]),4)))
        print ("specificity = TN / (TN + FP)")
        print ("Probability of a record being classified false,")
        print ("given that the record is in fact false")
        print ("----------------------------------------------------------------")

class classification_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS,site = None):
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)                                              # connect to the database
        self.class_stats_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','hitRatio','postTrue','postFalse','test','lagBdiff','conRecLength','noiseRatio','fishCount', 'logLikelihoodRatio'])                # set up an empty data frame
        self.site = site
        if site == None:
            recSQL = "SELECT * FROM tblMasterReceiver WHERE RecType = '%s'"%(self.recType) # SQL code to import data from this node
            receivers = pd.read_sql(recSQL,con = conn)                         # import data
            receivers = receivers.recID.unique()                               # get the unique receivers associated with this node    
            for i in receivers:                                                # for every receiver 
                print ("Start selecting and merging data for receiver %s"%(i))
                sql = "SELECT FreqCode, Epoch, recID, Power, hitRatio, postTrue, postFalse, test, lagBdiff, conRecLength, noiseRatio, fishCount, logLikelihoodRatio FROM tblClassify_%s "%(i)
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver 
                self.class_stats_data = self.class_stats_data.append(dat)
                del dat 
        else:
            print ("Start selecting and merging data for receiver %s"%(site))
            sql = "SELECT FreqCode, Epoch, recID, Power, hitRatio, postTrue, postFalse, test, lagBdiff, conRecLength, noiseRatio, fishCount, logLikelihoodRatio FROM tblClassify_%s "%(site)
            dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver 
            self.class_stats_data = self.class_stats_data.append(dat)
            del dat 
        c.close()
                    
    def classify_stats(self):
        '''function reads all classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        print ("")
        if self.site == None:
            print ("Classification summary statistics report")
        else:
            print ("Classification summary statistics report for site %s"%(self.site))
        print ("----------------------------------------------------------------------------------")
        det_class_count = self.class_stats_data.groupby('test')['test'].count()
        print ("")
        print ("%s detection class statistics:"%(self.recType)) 
        print ("The probability that a detection was classified as true was %s"%((round(float(det_class_count.get_value(1,'test'))/float(det_class_count.sum()),3))))
        print ("The probability that a detection was classified as fasle positive was %s"%((round(float(det_class_count.get_value(0,'test'))/float(det_class_count.sum()),3))))
        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("")
        self.sta_class_count = self.class_stats_data.groupby(['recID','test'])['test'].count().to_frame()#.reset_index(drop = False)
        self.recs = list(set(self.sta_class_count.index.levels[0]))
        print ("Detection Class Counts Across Stations")
        print ("          Classified     Classified")
        print ("             False          True")
        print ("       ______________________________")
        print ("      |              |              |")
        for i in self.recs:
            print ("%6s|   %8s   |   %8s   |"%(i,self.sta_class_count.loc[(i,0)].values[0],self.sta_class_count.loc[(i,1)].values[0]))
        print ("      |______________|______________|")    
        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("Compiling Figures")
        # get data by detection class for side by side histograms
        self.class_stats_data['Power'] = self.class_stats_data.Power.astype(float)
        self.class_stats_data['lagBDiff'] = self.class_stats_data.lagBdiff.astype(float)
        self.class_stats_data['conRecLength'] = self.class_stats_data.conRecLength.astype(float)
        self.class_stats_data['noiseRatio'] = self.class_stats_data.noiseRatio.astype(float)
        self.class_stats_data['fishCount'] = self.class_stats_data.fishCount.astype(float)
        self.class_stats_data['logPostRatio'] =np.log10(self.class_stats_data.postTrue.values/self.class_stats_data.postFalse.values)


        trues = self.class_stats_data[self.class_stats_data.test == 1] 
        falses = self.class_stats_data[self.class_stats_data.test == 0] 
        
        # plot hit ratio histograms by detection class
        hitRatioBins =np.linspace(0,1.0,11)
        
        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.hitRatio.values, hitRatioBins)
        axs[1].hist(falses.hitRatio.values, hitRatioBins)
        axs[0].set_xlabel('Hit Ratio')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Hit Ratio')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_hitRatioCompare.png"%(self.recType)))
        print ("Hit Ratio figure created, check your output workspace")
        
        # plot signal power histograms by detection class
        minPower = self.class_stats_data.Power.min()//10 * 10
        maxPower = self.class_stats_data.Power.max()//10 * 10
        powerBins =np.arange(minPower,maxPower+20,10)

        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.Power.values, powerBins)
        axs[1].hist(falses.Power.values, powerBins)
        axs[0].set_xlabel('%s Signal Power'%(self.recType))  
        axs[0].set_title('True')
        axs[1].set_xlabel('%s Signal Power'%(self.recType))
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_powerCompare.png"%(self.recType)))
        print ("Signal Power figure created, check your output Workspace")
        
        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,20)

        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.lagBdiff.values, lagBins)
        axs[1].hist(falses.lagBdiff.values, lagBins)
        axs[0].set_xlabel('Lag Differences')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Lag Differences')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_lagDifferences.png"%(self.recType)))
        print ("Lag differences figure created, check your output Workspace")
        
        # Consecutive Record Length ?
        conBins =np.arange(1,12,1)

        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.conRecLength.values, conBins)
        axs[1].hist(falses.conRecLength.values, conBins)
        axs[0].set_xlabel('Consecutive Hit Length')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Consecutive Hit Length')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_conRecLength.png"%(self.recType)))
        print ("Consecutive Hit Length figure created, check your output Workspace")

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.noiseRatio.values, noiseBins)
        axs[1].hist(falses.noiseRatio.values, noiseBins)
        axs[0].set_xlabel('Noise Ratio')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Noise Ratio')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_noiseRatio.png"%(self.recType)))
        print ("Noise Ratio figure created, check your output Workspace" )

        # plot fish present
        minCount = self.class_stats_data.fishCount.min()//10 * 10
        maxCount = self.class_stats_data.fishCount.max()//10 * 10
        countBins =np.arange(minCount,maxCount+20,10)

        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.fishCount.values, countBins)
        axs[1].hist(falses.fishCount.values, countBins)
        axs[0].set_xlabel('Fish Present')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Fish Present')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_fishPresentCompare.png"%(self.recType)))
        print ("Fish Present Figure Created, check output workspace")

        # plot the log of the posterior ratio 
        minPostRatio = self.class_stats_data.logLikelihoodRatio.min()//1 * 1
        maxPostRatio = self.class_stats_data.logLikelihoodRatio.max()//1 * 1
        ratioBins =np.arange(minPostRatio,maxPostRatio+1,2)
        
        plt.figure(figsize = (6,3)) 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
        axs[0].hist(trues.logLikelihoodRatio.values, ratioBins)
        axs[1].hist(falses.logLikelihoodRatio.values, ratioBins)
        axs[0].set_xlabel('Log Likelihood Ratio')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Log Likelihood Ratio')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_fishPresentCompare.png"%(self.recType)))
        print ("Fish Present Figure Created, check output workspace")

class training_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS):
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)                                 # connect to the database                                   
        sql = "SELECT * FROM tblTrain WHERE recType = '%s'"%(self.recType)
        self.train_stats_data = pd.read_sql(sql, con = conn, coerce_float = True) # get data for this receiver 
        c.close()
                    
    def train_stats(self):
        '''function reads all classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        print ("")
        print ("Training summary statistics report")
        print ("The algorithm collected %s detections from %s %s receivers"%(len(self.train_stats_data),len(self.train_stats_data.recID.unique()),self.recType))
        print ("----------------------------------------------------------------------------------")
        det_class_count = self.train_stats_data.groupby('Detection')['Detection'].count()
        print ("")
        print ("%s detection clas statistics:"%(self.recType) )
        print ("The prior probability that a detection was true was %s"%((round(float(det_class_count.get_value(1,'Detection'))/float(det_class_count.sum()),3))))
        print ("The prior probability that a detection was false positive was %s"%((round(float(det_class_count.get_value(0,'Detection'))/float(det_class_count.sum()),3))))
        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("")
        self.train_stats_data['Detection'] = self.train_stats_data.Detection.astype('str')
        self.sta_class_count = self.train_stats_data.groupby(['recID','Detection'])['Detection'].count().rename('detClassCount').to_frame().reset_index()
        self.recs = sorted(self.sta_class_count.recID.unique())
        print ("Detection Class Counts Across Stations")
        print ("             Known          Known")
        print ("             False          True")
        print ("       ______________________________")
        print ("      |              |              |")
        for i in self.recs:
            trues = self.sta_class_count[(self.sta_class_count.recID == i) & (self.sta_class_count.Detection == '1')]
            falses = self.sta_class_count[(self.sta_class_count.recID == i) & (self.sta_class_count.Detection == '0')]
            if len(trues) > 0 and len(falses) > 0:
                print ("%6s|   %8s   |   %8s   |"%(i,falses.detClassCount.values[0],trues.detClassCount.values[0]))
            elif len(trues) == 0 and len(falses) > 0:
                print ("%6s|   %8s   |   %8s   |"%(i,falses.detClassCount.values[0],0))
            else:
                print ("%6s|   %8s   |   %8s   |"%(i,0,trues.detClassCount.values[0]))

        print ("      |______________|______________|")    
        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("Compiling Figures")
        # get data by detection class for side by side histograms
        self.train_stats_data['Power'] = self.train_stats_data.Power.astype(float)
        self.train_stats_data['lagBDiff'] = self.train_stats_data.lagBdiff.astype(float)
        self.train_stats_data['conRecLength'] = self.train_stats_data.conRecLength.astype(float)
        self.train_stats_data['noiseRatio'] = self.train_stats_data.noiseRatio.astype(float)
        self.train_stats_data['FishCount'] = self.train_stats_data.FishCount.astype(float)

        trues = self.train_stats_data[self.train_stats_data.Detection == '1'] 
        falses = self.train_stats_data[self.train_stats_data.Detection == '0'] 
        # plot hit ratio histograms by detection class
        hitRatioBins =np.linspace(0,1.0,11)
        
        figSize = (3,2)
        plt.figure()
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.hitRatio.values, hitRatioBins, density = True)
        axs[1].hist(falses.hitRatio.values, hitRatioBins, density = True)
        axs[0].set_xlabel('Hit Ratio')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Hit Ratio')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_hitRatioCompare.png"%(self.recType)),bbox_inches = 'tight')
        print ("Hit Ratio figure created, check your output workspace")
        
        # plot signal power histograms by detection class
        minPower = self.train_stats_data.Power.min()//10 * 10
        maxPower = self.train_stats_data.Power.max()//10 * 10
        powerBins =np.arange(minPower,maxPower+20,10)

        plt.figure() 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.Power.values, powerBins, density = True)
        axs[1].hist(falses.Power.values, powerBins, density = True)
        axs[0].set_xlabel('%s Signal Power'%(self.recType))  
        axs[0].set_title('True')
        axs[1].set_xlabel('%s Signal Power'%(self.recType))
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_powerCompare.png"%(self.recType)),bbox_inches = 'tight')
        print ("Signal Power figure created, check your output Workspace")
        
        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,20)

        plt.figure() 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.lagBdiff.values, lagBins, density = True)
        axs[1].hist(falses.lagBdiff.values, lagBins, density = True)
        axs[0].set_xlabel('Lag Differences')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Lag Differences')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_lagDifferences.png"%(self.recType)),bbox_inches = 'tight')
        print ("Lag differences figure created, check your output Workspace")
        
        # Consecutive Record Length ?
        conBins =np.arange(1,12,1)

        plt.figure() 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.conRecLength.values, conBins, density = True)
        axs[1].hist(falses.conRecLength.values, conBins, density = True)
        axs[0].set_xlabel('Consecutive Hit Length')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Consecutive Hit Length')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_conRecLength.png"%(self.recType)),bbox_inches = 'tight')
        print ("Consecutive Hit Length figure created, check your output Workspace")

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

        plt.figure() 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.noiseRatio.values, noiseBins, density = True)
        axs[1].hist(falses.noiseRatio.values, noiseBins, density = True)
        axs[0].set_xlabel('Noise Ratio')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Noise Ratio')
        axs[1].set_title('False Positive')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_noiseRatio.png"%(self.recType)),bbox_inches = 'tight')
        print ("Noise Ratio figure created, check your output Workspace")

        # plot fish present
        minCount = self.train_stats_data.FishCount.min()//10 * 10
        maxCount = self.train_stats_data.FishCount.max()//10 * 10
        countBins =np.arange(minCount,maxCount+20,10)

        plt.figure() 
        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
        axs[0].hist(trues.FishCount.values, countBins, density = True)
        axs[1].hist(falses.FishCount.values, countBins, density = True)
        axs[0].set_xlabel('Fish Present')  
        axs[0].set_title('True')
        axs[1].set_xlabel('Fish Present')
        axs[1].set_title('Fish Present')
        axs[0].set_ylabel('Probability Density')
        plt.savefig(os.path.join(self.figureWS,"%s_fishPresentCompare.png"%(self.recType)),bbox_inches = 'tight')
        print ("Fish Present Figure Created, check output workspace")
        
class time_to_event():#inputFile,outputFile,time_dependent_covariates = False, covariates = None, bucket_length = 15):
    '''Class imports standardized raw state presences and converts data structure
    into counting process style data appropriate for time to event analysis.  
    
    Function inputs: 
        input file = directory with file name of raw timestamped state presence 
        output file = directory with file name for formatted data files.
        covariates = directory that contains covariates
        time_dependent_covariates = True/False (default = False) field indicating 
        whether or not time dependnent covariates are incorporated into the 
        counting process style
        bucket_length = covariate time interval, default 15 minutes  
    '''
    def __init__(self,receiver_list,node_to_state,dbDir, input_type = 'query', initial_state_release = False, last_presence_time0 = False):
        # Import Data using sql
        '''the default input type for this function is query, however 
        
        '''
        print ("Starting extraction of recapture data conforming to the recievers chosen")
        self.dbDir = dbDir
        conn = sqlite3.connect(dbDir)
        c = conn.cursor()
        sql = 'SELECT tblRecaptures.FreqCode, Epoch, timeStamp, Node, TagType, presence_number, overlapping FROM tblRecaptures LEFT JOIN tblMasterReceiver ON tblRecaptures.recID = tblMasterReceiver.recID LEFT JOIN tblMasterTag ON tblRecaptures.FreqCode = tblMasterTag.FreqCode WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])        
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        self.data = pd.read_sql_query(sql,con = conn)
        self.data['timeStamp'] = pd.to_datetime(self.data.timeStamp)
        self.data = self.data[self.data.TagType == 'Study']
        self.data = self.data[self.data.overlapping == 0]
        c.close()
        print ("Finished sql") 
        print ("Starting node to state classification, with %s records this takes time"%(len(self.data)))
        
        # Classify state
        def node_class(row,node_to_state):
            currNode = row['Node']
            #node_to_state = args[0]
            state = node_to_state[currNode]
            return state
        self.data['State'] = self.data.apply(node_class,axis = 1, args = (node_to_state,))

                
        if initial_state_release == True:
            '''we are modeling movement from the initial release location rather 
            than the initial location in our competing risks model.  This allows 
            us to quantify fall back.  If a fish never makes it to the intial 
            spoke, then its fall back.
            
            If we care about modeling from the release point, we need to query 
            release times of each fish, morph data into a recaptures file and 
            merge it to self.data'''
            
            sql = "SELECT FreqCode, TagType, RelDate FROM tblMasterTag WHERE TagType = 'Study'"
            conn = sqlite3.connect(self.dbDir)
            c = conn.cursor()
            relDat = pd.read_sql_query(sql,con = conn, parse_dates = 'RelDate')
            relDat['RelDate'] = pd.to_datetime(relDat.RelDate)
            relDat['Epoch'] = (relDat['RelDate'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            
            relDat.rename(columns = {'RelDate':'timeStamp'}, inplace = True)
            relDat['Node'] = np.repeat('rel',len(relDat))
            relDat['State'] = np.repeat(1,len(relDat))
            relDat['State'] = relDat['State'].astype(np.int32)

            relDat['Overlapping'] = np.zeros(len(relDat))
            print (relDat.head())
            c.close()
            self.data = self.data.append(relDat)

        self.data['State'] = pd.to_numeric(self.data['State'], errors='coerce')# make sure State is an integer
        self.data['State'] = self.data.State.astype(np.int32)
        
        if last_presence_time0 == True:
            ''' sometimes when we are modeling downstream movement, it would be 
            beneficial to start enumerating movement from the last presence at 
            the initial state.  
            
            For example, we have some long-term studies, where we are monitoring 
            fish for many months.  If the fish were transported upstream for a 
            downstream migration study, it may take months for that fish to move 
            back downstream.  For example: Short nose sturgeon in the CT River will 
            spend months to a year above Holyoke Dam potentially spawning.  They are 
            not actively migrating downstream.  In this instance, it would be beneficial 
            to model movement from the last presence at the initial state, after
            all, this is when it 'decides' to start its downstream migration.
            
            This works by finding the last presence at the initial state by fish 
            and modeling movement from that point.'''
            
            # identify the last presence at the initial state
            last_presence = self.data.groupby(['FreqCode','State'])['presence_number'].max().to_frame().reset_index()
            last_presence.rename(columns = {'presence_number':'max_presence'},inplace = True)
            last_presence = last_presence[last_presence.State == 1]
             
            # now that we have the last presence, iterate over fish 
            for i in last_presence.iterrows():
                fish = i[1]['FreqCode']
                max_pres = i[1]['max_presence']
                # get the first recapture at the last presence in the initial state for this fish
                recaps = self.data[(self.data.FreqCode == fish) & (self.data.State == 1) & (self.data.presence_number == max_pres)]
                min_epoch = recaps.Epoch.min()
                               
                # drop rows using boolean indexing 
                self.data.drop(self.data[(self.data.FreqCode == fish) & (self.data.Epoch < min_epoch)].index, inplace = True)

        # Identify first recapture times
        self.startTimes = self.data[self.data.State == 1].groupby(['FreqCode'])['Epoch'].min().to_frame()
        self.startTimes.reset_index(drop = False, inplace = True)
        self.startTimes.Epoch = self.startTimes.Epoch.astype(np.int32)
        self.startTimes.rename(columns = {'Epoch':'FirstRecapture'},inplace = True)
        for fish in self.data.FreqCode.unique():
            if fish not in self.startTimes.FreqCode.unique():                  # fish never made it to the initial state, remove - we only care about movements from the initial sstate - this is a competing risks model 
                self.data = self.data[self.data.FreqCode != fish]
        # identify unique fish
        self.fish = self.data.FreqCode.unique()                                # identify unique fish to loop through

    def data_prep(self,outputFile,time_dependent_covariates = False, unknown_state = None, bucket_length_min = 15):
        if unknown_state != None:
            '''It may be beneficial to allow fish to enter into an unknown state 
            rather than become censored at their last recapture in the initial state.
            This way the Nelson-Aalen will match empirical expectations.  If we have
            a lot of censored fish we lose information from the denominator and 
            numerator.  If we put fish into an unknown state rather than censoring 
            them we can still be informative.  For this to work, we only need to 
            know the last recapture of any fish in the initial state.  We will
            assess absorbption into the unknown state with a Boolean statement later on.''' 
            last_epoch = self.data[self.data.State == 1].Epoch.max()
            
        if time_dependent_covariates == False:
            '''This option will produce data appropriate for construction of 
            Nelson-Aalen cumulative incidence functions and to produce the state 
            tables.  
            
            This option is not appropriate if we wish to perform Cox Proportional 
            Hazards Regression modeling as we will not be able to join to time-
            dependent covariates in R.
            '''
            columns = ['FreqCode','state','presence','Epoch','timeDelta','time0','firstObs']    # create columns
            self.master_stateTable = pd.DataFrame()
            for i in self.fish:
                fishDat = self.data[self.data.FreqCode == i]                   # get data for this fish   
                # get first recapture in state 1
                fishDat.sort_values(by = 'Epoch', ascending = True, inplace = True)  # sort by exposure time
                fishDat['prevState'] = fishDat['State'].shift(1)               # get previous state
                fishDat['prevState'].fillna(fishDat.State.values[0], inplace = True)  # fill NaN states with current state - for first record in data frame
                presence = 1
                firstObs = 1
                stateTable = pd.DataFrame(columns = columns)                                # create empty data frame
                time0 = self.startTimes[self.startTimes.FreqCode == i].FirstRecapture.iloc[0]  # get initial time - need to calculate seconds between current record and time of release
                fishDat = fishDat[fishDat.Epoch >= time0]
                time1 = fishDat.Epoch.iloc[0]
                timeDelta = time1 - time0                                      # seconds between observation and release
                rowArr = [i,fishDat.State.values[0],presence,time1,timeDelta,time0,firstObs]  # create initial row for state table   
                row = pd.DataFrame(np.array([rowArr]),columns = columns)       # now it's officially a pandas row
                stateTable = stateTable.append(row)                            # now append that row
                firstObs = 0
                fishDat['idx'] = np.arange(0,len(fishDat),1)                   # gives row an arbitrary index
                maxIdx = fishDat.idx.iloc[-1]                                  # get that maximum index number        
                for j in fishDat.iterrows():                                   # for every row in fish data
                    rowIdx = j[1]['idx']                                       # what's the row number?
                    state = j[1]['State']                                      # what's the state
                    prevState = j[1]['prevState']                              # what was the previous state       
                    if state != prevState  or rowIdx == maxIdx:                # if the present state does not equal the previous state or if we reach the end of the dataframe...
                        time1 = j[1]['Epoch']                                  # what time is it?                        
                        if unknown_state != None and rowIdx == maxIdx and state == 1 and time1 < last_epoch:
                            state = unknown_state
                        timeDelta = time1 - time0                              # calculate difference in seconds between current time and release                                             # if it's a new state
                        if timeDelta > 300:
                            presence = presence + 1                                # oh snap new observation for new state              
                            rowArr = [i,state,presence,time1,timeDelta,time0,firstObs]  # start a new row    
                            row = pd.DataFrame(np.array([rowArr]),columns = columns)           
                            stateTable = stateTable.append(row)                    # add the row to the state table data frame  
                            time0 = j[1]['Epoch']
                    
                            
                    
                print ("State Table Completed for Fish %s"%(i))  
                stateTable['state'] = stateTable['state'].astype(np.int32)
                from_rec = stateTable['state'].shift(1).fillna(stateTable.iloc[0]['state']).astype(np.int32)
                to_rec = stateTable['state'].astype(np.int32)
                trans = zip(from_rec,to_rec)
                stateTable['transition'] = trans
                stateTable['startState'] = from_rec
                stateTable['endState'] = to_rec    
                stateTable['t0'] = stateTable['time0']
                stateTable['time0'] = pd.to_numeric(stateTable['time0'], errors='coerce')
                #stateTable['time1'] = pd.to_numeric(stateTable['time1'], errors='coerce')
                stateTable['Epoch'] = pd.to_numeric(stateTable['Epoch'], errors = 'coerce')
                stateTable.time0 = stateTable.time0.astype(np.int32)
                stateTable.Epoch = stateTable.Epoch.astype(np.int32)
                stateTable['t1'] = stateTable['Epoch'] - stateTable['time0']
                self.master_stateTable = self.master_stateTable.append(stateTable)    
            del i,j
        else:
            columns = ['FreqCode','startState','endState','presence','timeStamp','firstObs','t0','t1']    # create columns
            self.master_stateTable = pd.DataFrame()
            self.bucket_length = bucket_length_min
            for i in self.fish:
                fishDat = self.data[self.data.FreqCode == i]                   # get data for this fish   
                fishDat.sort_values(by = 'Epoch', ascending = True, inplace = True)   # sort by exposure time
                fishDat['prevState'] = fishDat['State'].shift(1)                      # get previous state
                fishDat['prevState'].fillna(fishDat.State.values[0], inplace = True)  # fill NaN states with current state - for first record in data frame
                presence = 0
                firstObs = 1
                stateTable = pd.DataFrame(columns = columns)                   # create empty data frame
                time0 = self.startTimes[self.startTimes.FreqCode == i].FirstRecapture.iloc[0]  # get initial time - need to calculate seconds between current record and time of release
                fishDat = fishDat[fishDat.Epoch >= time0]
                initialTime = pd.to_datetime(time0, unit = 's')
                time1 = fishDat.Epoch.iloc[0]
                timeDelta = time1 - time0                                      # seconds between observation and release
                firstObs = 0
                fishDat['idx'] = np.arange(0,len(fishDat),1)                   # gives row an arbitrary index
                maxIdx = fishDat.idx.iloc[-1]                                  # get that maximum index number        
                for j in fishDat.iterrows():                                   # for every row in fish data
                    rowIdx = j[1]['idx']                                       # what's the row number?
                    state1 = int(j[1]['prevState'])                            # what's the state
                    state2 = int(j[1]['State'])                                # what was the previous state
                    ts = j[1]['timeStamp']
                    if state1 != state2 or rowIdx == maxIdx:                   # if the present state does not equal the previous state or if we reach the end of the dataframe...
                        time1 = j[1]['Epoch']                                  # what time is it?
                        timeDelta = time1 - time0                              # calculate difference in seconds between current time and release                                             # if it's a new state
                        if state1 != state2 or rowIdx == maxIdx: 
                            presence = presence + 1                            # oh snap new observation for new state              
                        rowArr = [i,state1,state2,presence,ts,firstObs,time0,time1]  # start a new row    
                        row = pd.DataFrame(np.array([rowArr]),columns = columns)           
                        stateTable = stateTable.append(row)                    # add the row to the state table data frame  
                        time0 = j[1]['Epoch']
                print ("State Table Completed for Fish %s"%(i))

                stateTable['t0'] = pd.to_datetime(stateTable.t0.values, unit = 's') # get timestamp values
                stateTable['t1'] = pd.to_datetime(stateTable.t1.values, unit = 's') # get timestamp values
                stateTable.sort_values(by = 't0', ascending = True, inplace = True) # sort by exposure time
                timeBucket = self.bucket_length*60*1000000000                  # time bucket in nanoseconds
                stateTable['flowPeriod'] = (stateTable['t0'].astype(np.int64)//timeBucket+1) * timeBucket # round to nearest 15 minute period
                stateTable['flowPeriod'] = pd.to_datetime(stateTable['flowPeriod']) # turn it into a datetime object so we can use pandas to expand and fill 
                rowNum = np.arange(0,len(stateTable),1)
                stateTable['rowNum'] = rowNum  
                exp_stateTable = pd.DataFrame()    
                for row in stateTable.iterrows():
                    rowIdx = row[1]['rowNum']                                  # get row index number
                    t0 = row[1]['flowPeriod']                                  # identify current rows flow period
                    t1 = row[1]['t1']                                          # identify next row's flow period
                    try:
                        expand = pd.date_range(t0,t1,freq = '%smin'%(self.bucket_length)) # expand into 15 minute intervals
                    except ValueError:
                        expand = []
                    except AttributeError:
                        expand = []
                    if len(expand) > 0: 
                        #expand = expand[1:]
                        series = pd.Series(expand, index = expand, name = 'flowPeriod') # create series using exanded time intervals
                        intervals = series.to_frame()                          # convert series to dataframe
                        intervals.reset_index(inplace = True, drop = True)     # reset index
                        intervals['t0'] = row[1]['t0']                         # fill data for variables that don't change
                        intervals['t1'] = row[1]['t1']
                        intervals['startState'] = row[1]['startState']
                        intervals['endState'] = row[1]['endState']
                        intervals['timeStamp'] = row[1]['timeStamp']
                        intervals['FreqCode'] = row[1]['FreqCode']
                        intervals['presence'] = row[1]['presence']
                        newRowArr = np.array([row[1]['FreqCode'],row[1]['startState'],row[1]['endState'],row[1]['timeStamp'],row[1]['flowPeriod'],row[1]['t0'],row[1]['t1'],row[1]['presence']])
                        newRow = pd.DataFrame(np.array([newRowArr]),columns = ['FreqCode','startState','endState','timeStamp','flowPeriod','t0','t1','presence']) # add first, non expanded row to new state table
                        newRow = newRow.append(intervals)                      # add filled and expanded data
                        newRow['nextFlowPeriod'] = newRow['flowPeriod'].shift(-1) # identify the next flow period
                        newRow['idx'] = np.arange(0,len(newRow),1)             # add a count index field, but don't index it yet
                        newRow.reset_index(inplace = True, drop = True)        # remove the index
                        idxL = newRow.idx.values                               # generate list of indexes
                        newRow.loc[idxL[1]:,'t0'] = newRow.loc[idxL[1]:,'flowPeriod'].astype(str) # after the initial t0, re-write the current t0 as the current row's flow period
                        newRow.ix[:idxL[-2],'t1'] = newRow.loc[:idxL[-2],'nextFlowPeriod'].astype(str)  # other than the last t1, re-write the current t1 as the current row's next flow period - see what we did there?
                        newRow.ix[:idxL[-2]:,'endState'] = row[1]['startState']# other than the last row in the series, re-write the end state as the start state - there will be a lot of to-from same site here. it's ok, these are censored observations.  
                        newRow['t0'] = pd.to_datetime(newRow['t0'])            # convert time text to datetime - so we can do stuff with it
                        newRow['t1'] = pd.to_datetime(newRow['t1'])                               
                        exp_stateTable = exp_stateTable.append(newRow)         # now add all that stuff to the state table dataframe
                        del newRow, intervals, newRowArr, expand         
                    else:
                        newRowArr = np.array([row[1]['FreqCode'],row[1]['startState'],row[1]['endState'],row[1]['timeStamp'],row[1]['flowPeriod'],row[1]['t0'],row[1]['t1'],row[1]['presence']])
                        newRow = pd.DataFrame(np.array([newRowArr]),columns = ['FreqCode','startState','endState','timeStamp','flowPeriod','t0','t1','presence']) # add first, non expanded row to new state table
                        exp_stateTable = exp_stateTable.append(newRow)
                        del newRow, newRowArr
                exp_stateTable.sort_values(by = 't0', ascending = True, inplace = True)     # sort by exposure time
                exp_stateTable['time0'] = pd.to_datetime(exp_stateTable['t0']) # create new time columns
                exp_stateTable['time1'] = pd.to_datetime(exp_stateTable['t1'])
                exp_stateTable['t0'] = (pd.to_datetime(exp_stateTable['t0']) - initialTime)/np.timedelta64(1,'s')
                exp_stateTable['t1'] = (pd.to_datetime(exp_stateTable['t1']) - initialTime)/np.timedelta64(1,'s')

                exp_stateTable['hour'] = pd.DatetimeIndex(exp_stateTable['time0']).hour # get the hour of the day from the current time stamp
                exp_stateTable['qDay'] = exp_stateTable.hour//6                # integer division by 6 to put the day into a quarter
                exp_stateTable['test'] = exp_stateTable.t1 - exp_stateTable.t0 # this is no longer needed, but if t1 is smaller than t0 things are screwed up
                stateTable = exp_stateTable
                del exp_stateTable
                stateTable['transition'] = zip(stateTable.startState.values.astype(int),stateTable.endState.values.astype(int)) # create transition variable, this is helpful in R
                self.master_stateTable = self.master_stateTable.append(stateTable)
                # export
            self.master_stateTable.drop(labels = ['nextFlowPeriod','timeStamp'],axis = 1, inplace = True)
        self.master_stateTable.to_csv(outputFile)

    # generate summary statistics
    def summary(self):
        print ("--------------------------------------------------------------------------------------------------------------")
        print ("Time To Event Data Manage Complete")
        print ("--------------------------------------------------------------------------------------------------------------")
        print ("")
        print ("")
        print ("---------------------------------------MOVEMENT SUMMARY STATISTICS--------------------------------------------")
        print ("")
        print ("In Total, there were %s unique fish within this competing risks model"%(len(self.master_stateTable.FreqCode.unique())))
        print ("The number of unique fish per state:")
        countPerState = self.master_stateTable.groupby(['state'])['FreqCode'].nunique().to_frame()
        print (countPerState)
        print ("")
        msm_stateTable = pd.crosstab(self.master_stateTable.startState, self.master_stateTable.endState)
        print ("These fish made the following movements as enumerated in the state transition table:")
        print (msm_stateTable)
        print ("The table should read movement from a row to a column")
        print ("")
        self.countTable = self.master_stateTable.groupby(['startState','endState'])['FreqCode'].nunique().to_frame()
        self.countTable.reset_index(inplace = True)
        countPerTrans = pd.crosstab(self.countTable.startState,self.countTable.endState,values = self.countTable.FreqCode, aggfunc = 'sum')
        print ("The number of unique fish to make these movements are found in the following count table")
        print (countPerTrans)
        print ("")
        # Step 3: Describe the expected number of transitions per fish 
        self.fishTransCount = self.master_stateTable.groupby(['FreqCode','transition'])['transition'].count().to_frame().rename(columns = {'transition':'transCount'})
        self.fishTransCount.reset_index(inplace = True)
        
        print ("The number of movements a fish is expected to make is best described with min, median and maximum statistics")
        print ("The mininum number of times each transition was made:")
        min_transCount = self.fishTransCount.groupby(['transition'])['transCount'].min()
        print (min_transCount)
        print ("")
        print ("The median number of times each transition was made:")
        med_transCount = self.fishTransCount.groupby(['transition'])['transCount'].median()
        print (med_transCount)
        print ("")
        print ("The maximum number of times each transition was made:")
        max_transCount = self.fishTransCount.groupby(['transition'])['transCount'].max()
        print (max_transCount)


class fish_history():
    '''A class object to examine fish histories through space and time.
    
    When initialized, the class object connects to the project database and 
    creates a dataframe of all recaptures, filtered or unfiltered.
    
    Then, methods allow the end user to change fish and view plots.'''
    
    def __init__(self,projectDB,filtered = True, overlapping = False, rec_list = None,filter_date = None):
        ''' when this class is initialized, we connect to the project databae and 
        default filter the detections (test == 0) and overlapping = 0'''
        self.filtered = filtered
        conn = sqlite3.connect(projectDB,timeout = 30.0)
        c = conn.cursor()
        # read and import nodes
        sql = "SELECT X, Y, Node FROM tblNodes"
        self.nodes = pd.read_sql(sql,con = conn, coerce_float = True)
        self.nodes['Seconds'] = np.zeros(len(self.nodes))
        del sql
        if rec_list == None:
            # read and import receivers
            sql = "SELECT * FROM tblMasterReceiver"
            self.receivers = pd.read_sql(sql,con = conn, coerce_float = True)
            del sql
            receivers = self.receivers.recID.unique()
        else:
            sql = "SELECT * FROM tblMasterReceiver WHERE recID = '%s' "%(rec_list[0])
            for i in rec_list[1:]:
                sql = sql + "OR recID = '%s' "%(i)
            self.receivers = pd.read_sql(sql,con = conn, coerce_float = True)
            receivers = rec_list
        # read and import recaptures
        data = pd.DataFrame(columns = ['FreqCode','Epoch','timeStamp','recID','test'])
        for i in receivers:
            if filtered == False and overlapping == False:
                sql = "SELECT FreqCode, Epoch, timeStamp, recID, test FROM tblClassify_%s"%(i)
                dat = pd.read_sql(sql,con = conn)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            elif filtered == True and overlapping == False:
                #sql = "SELECT FreqCode, Epoch,  timeStamp, recID, test FROM tblClassify_%s WHERE test = 1 AND hitRatio > 0.3"%(i)
                sql = "SELECT FreqCode, Epoch,  timeStamp, recID, test FROM tblClassify_%s WHERE test = 1"%(i)
                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            else:
                sql = "SELECT tblClassify_%s.FreqCode, tblClassify_%s.Epoch,  tblClassify_%s.timeStamp, tblClassify_%s.recID, overlapping, test FROM tblClassify_%s LEFT JOIN tblOverlap ON tblClassify_%s.FreqCode = tblOverlap.FreqCode AND tblClassify_%s.Epoch = tblOverlap.Epoch AND tblClassify_%s.recID = tblOverlap.recID WHERE test = 1"%(i,i,i,i,i,i,i,i)
                #sql = "SELECT tblClassify_%s.FreqCode, tblClassify_%s.Epoch,  tblClassify_%s.timeStamp, tblClassify_%s.recID, overlapping, test FROM tblClassify_%s LEFT JOIN tblOverlap ON tblClassify_%s.FreqCode = tblOverlap.FreqCode AND tblClassify_%s.Epoch = tblOverlap.Epoch AND tblClassify_%s.recID = tblOverlap.recID WHERE test = 1 AND hitRatio > 0.3"%(i,i,i,i,i,i,i,i)
                #sql = "SELECT FreqCode, Epoch, recID, overlapping, test FROM tblClassify_%s ELFT JOIN tblOverlap ON tblClassify_%s.FreqCode = tblOverlap.FreqCode AND tblClassify_%s.Epoch = tblOverlap.Epoch AND tblClassify_%s.recID = tblOverlap.recID WHERE test = 1 AND hitRatio > 0.3"%(i,i,i,i)

                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                dat['overlapping'].fillna(0,inplace = True)
                dat = dat[dat.overlapping == 0]
                data = data.append(dat)
                #print data.head()
                #fuck
                print ("Data from reciever %s imported"%(i))
                del sql                

        self.recaptures = data
       
        del data
        # read and import tags
        sql = "SELECT * FROM tblMasterTag"
        self.tags = pd.read_sql(sql,con = conn, coerce_float = True)
        del sql
        
        # calculate first recaptures
        firstRecap = pd.DataFrame(self.recaptures.groupby(['FreqCode'])['Epoch'].min().reset_index())
        firstRecap = firstRecap.rename(columns = {'Epoch':'firstRecap'})
        
        self.recaptures = pd.merge(self.recaptures,firstRecap,how = u'left',left_on =['FreqCode'], right_on = ['FreqCode'])
        self.recaptures = pd.merge(self.recaptures,self.receivers[['recID','Node']], how = u'left', left_on = ['recID'], right_on = ['recID'])
        self.recaptures = pd.merge(self.recaptures,self.nodes, how = u'left', left_on = ['Node'], right_on = ['Node'])
        self.recaptures['TimeSinceRecap'] = (self.recaptures.Epoch - self.recaptures.firstRecap)
        self.recaptures['HoursSinceRecap'] = self.recaptures.TimeSinceRecap/3600.0
        c.close()
        
    def fish_plot(self, fish):
        '''Pick a tag and plot'''
        tagDat = self.recaptures[self.recaptures.FreqCode == fish]
        tagDat.sort_values(['Epoch'], inplace = True)
        tagFirstRecap = tagDat.firstRecap.values[0]
        print ("Fish %s had No. of Recaptures = %s with first recapture at %s" %(fish,len(tagDat),tagFirstRecap))
        # plot data
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111,projection = '3d')
        ax.plot(tagDat.X.values, tagDat.Y.values, tagDat.HoursSinceRecap.values, c = 'k')
        ax.scatter(self.nodes.X.values,self.nodes.Y.values,self.nodes.Seconds.values)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Hours')
        for row in self.nodes.iterrows():
            label = row[1][2]
            x = row[1][0]
            y = row[1][1]
            ax.text(x,y,0,label,fontsize = 8)
        plt.show()

def mortality_class(row):
    '''A function, similar to series hit, that classifies rows as mortality. 
    
    To be run on a mortality dataframe object.
        '''   
    # extract what we need from the row 
    lagB = abs(row['lagB'])
    lagF = abs(row['lagF'])
    mortRate = row['MortRate']
    '''if the lag B is less than or equal to 5 minutes, get the factors of 
    the log, MortRate should be in there somewhere.  Why only 5 minutes? 
    the fish is dead and not moving and this is an iterative search within 
    an iterative search operating over millions of rows of data.  I get bored.'''
    if lagB <= 300:
        factB = factors(lagB)
    else:
        factB = ['0']
    if lagB <= 300:
        factF = factors(lagF)
    else:
        factF = ['0']
    backMort = 0
    for i in factB:
        if mortRate - 1 <= i <= mortRate + 1:
            backMort = 1
    forwardMort = 0
    for j in factF:
        if mortRate - 1 <= j <= mortRate + 1:
            forwardMort = 1
    if backMort == 1 and forwardMort == 1:
        return 1
    else:
        return 0

class mortality():
    ''' A class object to first extract data by site,and fish to see if records
    mimic a dead pulse.  If the lag forward and backward is equal to +/- 1 from the 
    mort rate (or mort rate is factor), then the fish is dead.  
    
    We need both lags to trigger a death time'''
    
    def __init__ (self, site, dBase):
        print ("Starting to extract data for site %s"%(site))
        conn = sqlite3.connect(dBase)
        sql = "SELECT tblMasterTag.FreqCode, Epoch, recID, lagB, lagF, MortRate, test FROM tblClassify_%s LEFT JOIN tblMasterTag ON tblClassify_%s.FreqCode = tblMasterTag.FreqCode"%(site,site) 
        self.mort_df = pd.read_sql(sql, con = conn, coerce_float = True)  
        print ("Import Completed, start Mortality Calculations")      
        self.mort_df['mortality'] = self.mort_df.apply(mortality_class, axis = 1)             # determine whether or not a row record is in series 
        self.site = site
        self.dBase = dBase
        
    def summary(self):
        ''' Run some summary statistics, get time of death for each animal that died'''
        dead = self.mort_df[self.mort_df.mortality == 1]
        deadFish = dead.groupby(['FreqCode'])['Epoch'].min().to_frame().rename(columns = {'Epoch':'time_of_death'})
        deadFish['time_of_death'] = pd.to_datetime(deadFish['time_of_death'], unit ='s')
        deadFish['Site'] = np.repeat(self.site,len(deadFish))
        print ("Time of death for those fish that died at site %s"%(self.site))
        print (deadFish)
        conn = sqlite3.connect(self.dBase)
        deadFish.to_sql('tblMortality',con = conn,index = False, if_exists = 'append', chunksize = 1000) 
            
class bout():
    '''Python class object to delineate when bouts occur at receiver.'''
    def __init__ (self,node,dBase,lag_window, time_limit):
        self.lag_window = lag_window
        self.time_limit = time_limit        
        conn = sqlite3.connect(dBase)
        recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(node)   # SQL code to import data from this node
        receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)        # import data
        receivers = receivers[receivers.Node == node].recID.unique()           # get the unique receivers associated with this node    
        data = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])            # set up an empty data frame
        for i in receivers:                                                    # for every receiver 
            sql = "SELECT FreqCode, Epoch, recID, test, hitRatio FROM tblClassify_%s"%(i) 
            dat = pd.read_sql(sql, con = conn)                                 # get data for this receiver 
            print ("Got data for receiver %s"%(i))
            #dat = dat[(dat.test == 1) & (dat.hitRatio > 0.3)] # query
            dat = dat[(dat.test == 1)] # query
            dat.drop(['test','hitRatio'],axis = 1, inplace = True)
            print ("Restricted data")
            data = data.append(dat)  
        c = conn.cursor()
        c.close()
        data.drop_duplicates(keep = 'first', inplace = True)
        data['det_lag'] = (data.Epoch.diff()//lag_window * lag_window)
        data.dropna(axis = 0, inplace = True)                                  # drop Nan from the data
        self.bout_data = data[(data.det_lag > 0) & (data.det_lag <= time_limit)]
        self.node = node
        self.fishes = data.FreqCode.unique()
        
    def broken_stick_3behavior(self, plotOutputWS = None):
        '''Function to find the bout length using a broken stick/ piecewise linear
        regression technique.  We do not know the knot criteria ahead of time, therefore 
        our procedure finds the knot that minimizes error.
        
        Method will attempt to minimize the sum of the sum of squared residuals for each 
        each line.  If we assume there are two behaviors, there are two equations and one 
        breakpoint.  The position of the breakpoint is where the sum of the sum 
        squares of the residuals is minimized.  
        
        Further, we are only testing breakpoints that exist in our data, therefore lags can only 
        occur on 2 second intervals, not continuously'''  
        
        # get data
        det_lag_DF = self.bout_data
        det_lag_DF = det_lag_DF.groupby('det_lag')['det_lag'].count()          # group by the detection lag and count each of the bins
        det_lag_DF = pd.Series(det_lag_DF, name = 'lag_freq')                  # rename the series field 
        det_lag_DF = pd.DataFrame(det_lag_DF).reset_index()                    # convert to a dataframe
        det_lag_DF['log_lag_freq'] = np.log(det_lag_DF.lag_freq)               # log the lal frequencies - make things nice and normal for OLS
        knots = det_lag_DF.det_lag.values[2:-2]                                # extract every potential detection lag values that can be a knot - need at least 3 points for an OLS regression, therefore we extract every node except the first three and last three                     
        knot_ssr = dict()                                                      # create empty dictionary to hold results

        if len(knots) <= 5: # middle behavior segment must be at least 3 points long
            minKnot = 7200
            return minKnot
        else:
            # test all potential knots and write MSE to knot dictionary
            for i in np.arange(2,len(knots),1):    
                k1 = i
                for j in np.arange(i+2,len(knots)-2,1):
                    k2 = j
                    first = det_lag_DF.iloc[0:k1]                              # get data
                    second = det_lag_DF[k1:k2]
                    third = det_lag_DF.iloc[k2:]
                    mod1 = smf.ols('log_lag_freq~det_lag', data = first).fit() # fit models to the data
                    mod2 = smf.ols('log_lag_freq~det_lag', data = second).fit()
                    mod3 = smf.ols('log_lag_freq~det_lag', data = third).fit()
                    mod1ssr = mod1.ssr                                         # get the sum of the squared residuals from each model
                    mod2ssr = mod2.ssr
                    mod3ssr = mod3.ssr
                    print ("With knots at %s and %s seconds, the SSR of each model was %s, %s and %s respectivley"%(det_lag_DF.det_lag.values[k1],det_lag_DF.det_lag.values[k2],round(mod1ssr,4),round(mod2ssr,4),round(mod3ssr,4)))
                    ssr_sum = np.sum([mod1ssr,mod2ssr,mod3ssr])                # simple sum is fine here
                    knot_ssr[(k1,k2)] = ssr_sum                                     # add that sum to the dictionary
            minKnot = min(knot_ssr.iteritems(), key = operator.itemgetter(1))[0]   # find the knot locations that mimize squared error
            # You Can Plot If You Want To
            if plotOutputWS != None:
                k1 = minKnot[0]
                k2 = minKnot[1]
                dat1 = det_lag_DF.iloc[0:k1]
                dat2 = det_lag_DF.iloc[k1:k2]
                dat3 = det_lag_DF.iloc[k2:]
                mod1 = smf.ols('log_lag_freq~det_lag', data = dat1).fit()     # fit models to the data
                mod2 = smf.ols('log_lag_freq~det_lag', data = dat2).fit()
                mod3 = smf.ols('log_lag_freq~det_lag', data = dat3).fit()
                x1 = np.linspace(dat1.det_lag.min(), dat1.det_lag.max(), 100)
                x2 = np.linspace(dat2.det_lag.min(), dat2.det_lag.max(), 100) 
                x3 = np.linspace(dat3.det_lag.min(), dat3.det_lag.max(), 100)              
                plt.figure(figsize = (3,3))    
                plt.plot(det_lag_DF.det_lag.values, det_lag_DF.log_lag_freq.values, "o")        
                plt.plot(x1, mod1.params[1]*x1+mod1.params[0], color = 'red')
                plt.plot(x2, mod2.params[1]*x2+mod2.params[0], color = 'red')
                plt.plot(x3, mod3.params[1]*x3+mod3.params[0], color = 'red')
                plt.xlabel('Detection Lag (s)')
                plt.ylabel('Log Lag Count')
                plt.title('Bout Length Site %s \n %s seconds'%(self.node, det_lag_DF.det_lag.values[minKnot[1]]))
                plt.ylim(0,max(det_lag_DF.log_lag_freq.values)+1)
                plt.savefig(os.path.join(plotOutputWS,"BoutsAtSite%s.png"%(self.node)))
        
            return det_lag_DF.det_lag.values[minKnot[1]]

    def presence(self, fish, bout_length, dBase, scratch):
        '''Function takes the break point between a continuous presence and new presence, 
        enumerates the presence number at a receiver and writes the data to the 
        analysis database.'''
        # get data
        conn = sqlite3.connect(dBase)
        recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(self.node)            # SQL code to import data from this node
        receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)                 # import data
        receivers = receivers[receivers.Node == self.node].recID.unique()                    # get the unique receivers associated with this node    
        presence = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])                             # set up an empty data frame
        for i in receivers:                                                            # for every receiver 
            sql = "SELECT FreqCode, Epoch, recID, test, hitRatio, logLikelihoodRatio FROM tblClassify_%s WHERE FreqCode == '%s'"%(i,fish) 
            dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
            #dat = dat[(dat.test == 1) & (dat.hitRatio > 0.3) & (dat.logLikelihoodRatio > 0.0)] # query
            dat = dat[(dat.test == 1) & (dat.logLikelihoodRatio > 0.0)] # query

            dat.drop(['test','hitRatio','logLikelihoodRatio'],axis = 1, inplace = True)
            presence = presence.append(dat)
        c = conn.cursor()
        c.close()
        presence.drop_duplicates(keep = 'first', inplace = True)
        # get fish data
        presence['det_lag'] = (presence.Epoch.diff()//self.lag_window * self.lag_window)
        presence['det_lag'].fillna(0, inplace = True)
        presence.sort_values(by = ['Epoch'], axis = 0, inplace = True)
        presence.set_index('Epoch', drop = False, inplace = True)
        # determine new presences 
        # step 1: apply new presence criteria to lag column using numpy where
        presence['new_presence'] = np.where(presence.det_lag.values > bout_length,1,0)
        # step 2: extract new presences 
        new_presence = presence[presence.new_presence == 1]
        new_presence.reset_index(inplace = True, drop = True)
        new_presence.sort_values('Epoch', inplace = True)
        # step 3: rank each presence by epoch 
        new_presence['presence_number'] = new_presence.Epoch.rank()
        new_presence.set_index('Epoch', drop = False, inplace = True)

        # step 4: join presence number to presence data frame on epoch 
        presence = presence.join(new_presence[['presence_number']], how = 'left')
        # step 5: forward fill presence numbers until the next new presence, then fill initial presence as zero
        presence.fillna(method = 'ffill',inplace = True)
        presence.fillna(value = 0, inplace = True)
        presence.to_csv(os.path.join(scratch,'%s_at_%s_presence.csv'%(fish,self.node)), index = False)
                
def manage_node_presence_data(inputWS, projectDB):
    files = os.listdir(inputWS)       
    conn = sqlite3.connect(projectDB)        
    c = conn.cursor()        
    for f in files:           
        dat = pd.read_csv(os.path.join(inputWS,f), dtype = {'FreqCode':str,'Epoch':np.int32,'recID':str,'det_lag':np.int32,'presence_number':np.float64})           
        dat.to_sql('tblPresence',con = conn,index = False, if_exists = 'append', chunksize = 1000)           
        os.remove(os.path.join(inputWS,f))
    c.close() 
        
class overlap_reduction():
    '''Python class project to reduce redundant dections at overlappin receivers. 
    More often than not, a large aerial yagi will be placed adjacent to a dipole.
    The yagi has a much larger detection area and will pick up fish in either detection 
    zone.  The dipole is limited in its coverage area, therefore if a fish is 
    currently present at a dipole and is also detected at the yagi, we can safely 
    assume that the detection at the Yagi are overlapping and we can place the fish
    at the dipole antenna.  By identifying and removing these overlapping detections
    we remove bias in time-to-event modeling when we want to understand movement
    from detection areas with limited coverage to those areas with larger aerial 
    coverages.  
    
    This class object contains a series of methods to identify overlapping detections
    and import a table for joining into the project database.'''
    
    def __init__(self,curr_node,nodes,edges, projectDB, outputWS,figureWS = None):
        '''The initialization module imports data and creates a networkx graph object.
        
        The end user supplies a list of nodes, and a list of edges with instructions 
        on how to connect them and the function does the rest.  NO knowlege of networkx
        is required.  
        
        The nodes and edge relationships should start with the outermost nodes and 
        eventually end with the inner most node/receiver combinations.  
        
        Nodes must be a list of nodes and edges must be a list of tuples.
        Edge example: [(1,2),(2,3)], 
        Edges always in format of [(from,to)] or [(outer,inner)]'''
        
        # Step 1, create a directed graph from list of edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.curr_node = curr_node
        self.outputWS = outputWS
        # Step 2, import data and create a dictionary of node dataframes        
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        self.node_pres_dict = dict()
        self.node_recap_dict = dict()
        for i in nodes:
            #import data and add to node dict
            recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(i)  # SQL code to import data from this node
            receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)    # import data
            node_recs = receivers[receivers.Node == i].recID.unique()          # get the unique receivers associated with this node                
            pres_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','presence_number'])        # set up an empty data frame
            recap_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])
            for j in node_recs:                                                # for every receiver 
                #print "Start selecting classified and presence data that matches the current receiver (%s)"%(j)  
                presence_sql = "SELECT * FROM tblPresence WHERE recID = '%s'"%(j)
                presenceDat = pd.read_sql(presence_sql, con = conn)
                recap_sql = "SELECT FreqCode, Epoch, recID, test, hitRatio from tblClassify_%s"%(j)
                recapDat = pd.read_sql(recap_sql, con = conn)
                #recapDat = recapDat[(recapDat.test == 1) & (recapDat.hitRatio > 0.3)]
                recapDat = recapDat[(recapDat.test == 1)]

                recapDat.drop(labels = ['test','hitRatio',], axis = 1, inplace = True)
                # now that we have data, we need to summarize it, use group by to get min ans max epoch by freq code, recID and presence_number
                pres_data = pres_data.append(presenceDat)
                recap_data = recap_data.append(recapDat)
            dat = pres_data.groupby(['FreqCode','presence_number'])['Epoch'].agg({'min_Epoch':np.min,'max_Epoch':np.max}).reset_index(drop = False)
            self.node_pres_dict[i] = dat
            self.node_recap_dict[i] = recap_data
            del pres_data, recap_data, dat, recSQL, receivers, node_recs, presence_sql, recap_sql
        print ("Completed data management process for node %s"%(self.curr_node))
        c.close()
        
        # visualize the graph
        if figureWS != None:
            shells = []
            for n in list(self.G.nodes()):
                successors = list(self.G.succ[n].keys())
                shells.append(successors)
                
            fig, ax = plt.subplots(1, 1, figsize=(4, 4));
            pos= nx.circular_layout(self.G)
            nx.draw_networkx_nodes(self.G,pos,list(self.G.nodes()),node_color = 'r',node_size = 400)
            nx.draw_networkx_edges(self.G,pos,list(self.G.edges()),edge_color = 'k')
            nx.draw_networkx_labels(self.G,pos,font_size=8)
            plt.axis('off')
            plt.savefig(os.path.join(figureWS,'overlap_model.png'),bbox_inches = 'tight')
            plt.show()
        
def russian_doll(overlap):
    '''Function iterates through matching recap data from successors to see if
    current recapture row at predeccesor overlaps with successor presence.'''
    #print "Starting Russing Doll algorithm for node %s, hope you aren't busy, you will be watching this process for hours"%(overlap.curr_node)
    nodeDat = overlap.node_recap_dict[overlap.curr_node]
    fishes = nodeDat.FreqCode.unique()
    for i in fishes:
        overlap.fish = i
        #print "Let's start sifting through fish %s at node %s"%(i, overlap.curr_node)
        successors = overlap.G.succ[overlap.curr_node]
        fishDat = nodeDat[nodeDat.FreqCode == i]
        fishDat['overlapping'] = np.zeros(len(fishDat))
        fishDat['successor'] = np.repeat('',len(fishDat))
        fishDat.set_index('Epoch', inplace = True)
        if len(successors) > 0:                                            # if there is no child node, who cares? there is no overlapping detections, we are at the middle doll
            for index, row in fishDat.iterrows():                          # for every row in the fish data
                epoch = index                                              # get the current time
                #print "Fish %s epoch %s overlap check"%(i,epoch)
                for j in successors:                                       # for each successor of the current node
                    #print "current successor is receiver %s" %(j)
                    #print "Sifting through node %s for overlaps"%(j)
                    succDat = overlap.node_pres_dict[j]                       # get the current successor data 
                    if len(succDat) > 0:
                        succDat = succDat[succDat.FreqCode == i]               # extract this fish's data
                        succDat = succDat[(succDat.min_Epoch <= epoch) & (succDat.max_Epoch >= epoch)]
                        #print "Overlap Found, at %s fish %s was recaptured at both %s and %s"%(epoch,i,overlap.curr_node,j)
                        #print succDat
                        fishDat.set_value(epoch,'overlapping',1)
                        fishDat.set_value(epoch,'successor',j)
                        break
        fishDat.reset_index(inplace = True, drop = False)
        fishDat.to_csv(os.path.join(overlap.outputWS,'%s_at_%s_soverlap.csv'%(i,overlap.curr_node)), index = False)
                
def manage_node_overlap_data(inputWS, projectDB):
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)        
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f), dtype = {'FreqCode':str,'Epoch':np.int32,'recID':str,'overlapping':np.int32})
        dat.to_sql('tblOverlap',con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
    c.close()
    
def the_big_merge(outputWS,projectDB):
    '''function takes classified data, merges across sites and then joins presence 
    and overlapping data into one big file for import into Access for model building.'''
    conn = sqlite3.connect(projectDB)                                              # connect to the database
    recSQL = "SELECT * FROM tblMasterReceiver"                                 # SQL code to import data from this node
    receivers = pd.read_sql(recSQL,con = conn)                                 # import data
    receivers = receivers.recID.unique()                                       # get the unique receivers associated with this node    
    recapdata = pd.DataFrame(columns = ['FreqCode','Epoch','recID','timeStamp','Power','lagF','lagB','hitRatio','postTrue','postFalse','test','presence_number','overlapping'])                # set up an empty data frame
    for i in receivers:                                                            # for every receiver 
        print ("Start selecting and merging data for receiver %s"%(i))
        #sql = "SELECT tblClassify_%s.FreqCode, tblClassify_%s.Epoch, tblClassify_%s.recID, timeStamp, Power, LagF, LagB, tblClassify_%s.hitRatio, postTrue, postFalse, presence_number, overlapping, test FROM tblClassify_%s LEFT JOIN tblOverlap ON tblClassify_%s.FreqCode = tblOverlap.FreqCode AND tblClassify_%s.Epoch = tblOverlap.Epoch AND tblClassify_%s.recID = tblOverlap.recID LEFT JOIN tblPresence ON tblClassify_%s.FreqCode = tblPresence.FreqCode AND tblClassify_%s.Epoch = tblPresence.Epoch AND tblClassify_%s.recID = tblPresence.recID WHERE test = 1 AND tblClassify_%s.hitRatio > 0.3"%(i,i,i,i,i,i,i,i,i,i,i,i)
        sql = "SELECT tblClassify_%s.FreqCode, tblClassify_%s.Epoch, tblClassify_%s.recID, timeStamp, Power, LagF, LagB, tblClassify_%s.hitRatio, postTrue, postFalse, presence_number, overlapping, test FROM tblClassify_%s LEFT JOIN tblOverlap ON tblClassify_%s.FreqCode = tblOverlap.FreqCode AND tblClassify_%s.Epoch = tblOverlap.Epoch AND tblClassify_%s.recID = tblOverlap.recID LEFT JOIN tblPresence ON tblClassify_%s.FreqCode = tblPresence.FreqCode AND tblClassify_%s.Epoch = tblPresence.Epoch AND tblClassify_%s.recID = tblPresence.recID WHERE test = 1"%(i,i,i,i,i,i,i,i,i,i,i)

        dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver 
        dat['overlapping'].fillna(0,inplace = True)
        recapdata = recapdata.append(dat)
        del dat
    c = conn.cursor()
    recapdata.to_sql('tblRecaptures',con = conn,index = False)
    c.close()

class cjs_data_prep():
    '''Class creates input files for Cormack Jolly Seber modeling in MARK'''
    def __init__(self,receiver_list,receiver_to_recap, dbDir, input_type = 'query'):
        # Import Data using sql
        '''the default input type for this function is query, however'''
        print ("Starting extraction of recapture data conforming to the recievers chosen")
        conn = sqlite3.connect(dbDir)
        c = conn.cursor()
        sql = 'SELECT tblRecaptures.FreqCode, Epoch, timeStamp, tblRecaptures.recID, TagType, overlapping FROM tblRecaptures LEFT JOIN tblMasterReceiver ON tblRecaptures.recID = tblMasterReceiver.recID LEFT JOIN tblMasterTag ON tblRecaptures.FreqCode = tblMasterTag.FreqCode WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])        
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        self.data = pd.read_sql_query(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        self.data = self.data[self.data.TagType == 'Study']
        self.data = self.data[self.data.overlapping == 0]
        c.close()
        print ("Finished sql")
        print ("Starting receiver to recapture occasion classification, with %s records, this takes time"%(len(self.data)))
        # create function that maps recapture occasion to row given current receiver and apply 
        def node_class(row,receiver_to_recap):
            currRec = row['recID']
            #node_to_state = args[0]
            recapOcc = receiver_to_recap[currRec]
            return recapOcc
        self.data['RecapOccasion'] = self.data.apply(node_class,axis = 1, args = (receiver_to_recap,))
        # Identify first recapture times
        self.startTimes = self.data[self.data.RecapOccasion == "R0"].groupby(['FreqCode'])['Epoch'].min().to_frame()
        self.startTimes.reset_index(drop = False, inplace = True)
        self.startTimes.Epoch = self.startTimes.Epoch.astype(np.int32)
        self.startTimes.rename(columns = {'Epoch':'FirstRecapture'},inplace = True)
        for fish in self.data.FreqCode.unique():
            if fish not in self.startTimes.FreqCode.unique():                  # fish never made it to the initial state, remove - we only care about movements from the initial sstate - this is a competing risks model 
                self.data = self.data[self.data.FreqCode != fish]
        # identify unique fish
        self.fish = self.data.FreqCode.unique()                                # identify unique fish to loop through
    
    def input_file(self,modelName, outputWS):
        #Step 1: Create cross tabulated data frame with FreqCode as row index 
        #and recap occasion as column.  Identify the min epoch'''
        cross_tab = pd.pivot_table(self.data, values = 'Epoch', index = 'FreqCode', columns = 'RecapOccasion', aggfunc = 'min')
        #Step 2: Fill in those nan values with 0, we can't perform logic on nothing!'''
        cross_tab.fillna(value = 0, inplace = True)
        #Step 3: Map a simply if else statement, if value > 0,1 else 0'''
        cross_tab = cross_tab.applymap(lambda x:1 if x > 0 else 0)
        # Check your work
        print (cross_tab.head(10))
        cross_tab.to_csv(os.path.join(outputWS,'%s_cjs.csv')%(modelName))
        
class receiver_stats():
    '''Python class object that creates a receiver object of either raw or reduced 
    data to generate basic recacpture statistics'''
    
    def __init__(self,receiver,projectDB,reduced = True):
        '''initialization method that creates the reciever stats object.
        
        Arguments passed to the class include the receiver, link to the project
        database and whether or not we want reduced data.  If we want reduced data, 
        we extract information from the big merge - which includes overlap removal.'''
        self.receiver = receiver
        self.projectDB = projectDB
        if reduced == True:
            conn = sqlite3.connect(projectDB)
            sql = "SELECT * FROM tblRecaptures WHERE recID == '%s'"%(self.receiver)
            self.rec_dat = pd.read_sql(sql, con = conn, coerce_float = True)  
        else:
            conn = sqlite3.connect(projectDB)
            sql = "SELECT * FROM tblClassify_%s WHERE recID == '%s' AND test = 1"%(self.receiver,self.receiver)
            self.rec_dat = pd.read_sql(sql, con = conn, coerce_float = True)  
    def recap_count(self):
        fishes = self.rec_dat.FreqCode.unique()
        print ("The number of unique fish recaptured at telemetry site %s is %s"%(self.receiver,len(fishes)))
        #print fishes      

    def durations(self,fish = None):
        if fish != None:
            fishDat = self.rec_dat[self.rec_dat.FreqCode == fish]
        else:
            fishDat = self.rec_dat
        self.duration = fishDat.groupby(['FreqCode'])['Epoch'].agg([np.min,np.max])
        self.duration.rename(columns = {'amin':'time_0','amax':'time_1'},inplace = True)
        self.duration['duration'] = (self.duration.time_1 - self.duration.time_0)/3600.0
        self.duration['time_0ts'] = pd.to_datetime(self.duration.time_0,unit = 's')
        self.duration['time_1ts'] = pd.to_datetime(self.duration.time_1,unit = 's')
        
    def presences(self,fish):
        fishDat = self.rec_dat[self.rec_dat.FreqCode == fish]
        self.presences = fishDat.groupby(['FreqCode'])['presence_number'].max()
        
        
        
          

        
        

        
    
        
        
    
            

