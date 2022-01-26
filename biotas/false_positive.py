# -*- coding: utf-8 -*-
"""
Module contains all of the functions and classes to identify and remove false
poisitive detections from radio telemetry data.
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx
from matplotlib import rcParams
from scipy import interpolate
from sklearn import metrics

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

def factors(n):

    ''''function to return primes used to quantify the least common multiplier
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

def detHist (data,rate,det, status = 'A', training = False):
    '''New iteration of the detection history function meant to be superfast.
    We use pandas tricks to create the history by shifting epoch columns and
    perfoming boolean logic on the entire dataframe rather than row-by-row'''

    # first step, identify how many channels are in this dataset
    channels = data.Channels.max()

    # build detection history ushing shifts
    if channels > 1:
        for i in np.arange(det * -1 , det + 1, 1):
            data['epoch_shift_%s'%(np.int(i))] = data.Epoch.shift(-1 * np.int(i))
            data['ll_%s'%(np.int(i))] = data['Epoch'] + ((data['ScanTime'] * data['Channels'] * i) - 1)
            data['ul_%s'%(np.int(i))] = data['Epoch'] + ((data['ScanTime'] * data['Channels'] * i) + 1)

            '''if the scan time is 2 seconds and there are two channels and the Epoch is 100...
            We would listen to this receiver at:
                (-3) 87-89, (-2) 91-93, (-1) 95-97, (0) 100, (1) 103-105, (2) 107-109, (3) 111-113 seconds

            A tag with a 3 second burst rate at epoch of 100 would be heard at:
                100, 103, 106, 109, 112 - we would miss at least 1 out of 5 bursts!!!

            We should expect detection histories from multi channel switching receivers
            to be sparse because they are not always listening - bursts will never
            line up with scan windows - need to get over it and trust the sparseness - eek
            '''

            #print (data[['Epoch','ll','epoch_shift_%s'%(np.int(i)),'ul','det_%s'%(np.int(i))]].head(20))
    else:
        for i in np.arange(det * -1 , det + 1, 1):
            data['epoch_shift_%s'%(np.int(i))] = data.Epoch.shift(-1 * np.int(i))
            data['ll_%s'%(np.int(i))] = data['Epoch'] + (rate * i - 1)   # number of detection period time steps backward buffered by -1 pulse rate seconds that we will be checking for possible detections in series
            data['ul_%s'%(np.int(i))] = data['Epoch'] + (rate * i + 1)     # number of detection period time steps backward buffered by +1 pulse rate that we will be checking for possible detections in series
            #print (data[['Epoch','ll','epoch_shift_%s'%(np.int(i)),'ul','det_%s'%(np.int(i))]].head(20))
    for i in np.arange(det * -1 , det + 1, 1):
        if det == 0:
            data['det_0'] = '1'
        else:
            for j in np.arange(det * -1 , det + 1, 1):
                data.loc[(data['epoch_shift_%s'%(np.int(j))] >= data['ll_%s'%(np.int(i))]) &\
                         (data['epoch_shift_%s'%(np.int(j))] <= data['ul_%s'%(np.int(i))]),'det_%s'%(np.int(i))] = '1'

        data['det_%s'%(np.int(i))].fillna(value = '0', inplace = True)


    # determine if consecutive detections consDet is true
    data.loc[(data['det_-1'] == '1') | (data['det_1'] == '1'), 'consDet_%s'%(status)] = 1
    data.loc[(data['det_-1'] != '1') & (data['det_1'] != '1'), 'consDet_%s'%(status)] = 0

    # calculate hit ratio
    data['hitRatio_%s'%(status)] = np.zeros(len(data))
    for i in np.arange(det * -1, det + 1, 1):
        data['hitRatio_%s'%(status)] = data['hitRatio_%s'%(status)] + data['det_%s'%(np.int(i))].astype(np.int)
    data['hitRatio_%s'%(status)] = data['hitRatio_%s'%(status)] / np.float(len(np.arange(det * -1, det + 1, 1)))

    # concatenate a detection history string and calculate maximum consecutive hit length
    data['detHist_%s'%(status)] = np.repeat('',len(data))
    data['conRecLength_%s'%(status)] = np.zeros(len(data))
    data['score'] = np.zeros(len(data))
    for i in np.arange(det * -1, det + 1, 1):
        data['detHist_%s'%(status)] = data['detHist_%s'%(status)] + data['det_%s'%(np.int(i))].astype('str')
        if i == min(np.arange(det * -1, det + 1, 1)):
            data['score'] = data['det_%s'%(np.int(i))].astype(int)
        else:
            data['conRecLength_%s'%(status)] = np.where(data['conRecLength_%s'%(status)] < data.score, data.score, data['conRecLength_%s'%(status)])
            data.loc[(data['det_%s'%(np.int(i))] == '0','points')] = 0
            data.loc[(data['det_%s'%(np.int(i))] == '0','score')] = 0
            data.loc[(data['det_%s'%(np.int(i))] == '1','points')] = 1
            data['score'] = data.score + data.points
        data.drop(labels = ['det_%s'%(np.int(i)),'epoch_shift_%s'%(np.int(i))], axis = 1, inplace = True)
        data.drop(labels = ['ll_%s'%(np.int(i)),'ul_%s'%(np.int(i))], axis = 1, inplace = True)

    data.drop(labels = ['score','points'], axis = 1, inplace = True)

    if training == True:
        data.rename(columns = {'detHist_A':'detHist','consDet_A':'consDet','hitRatio_A':'hitRatio','conRecLength_A':'conRecLength'},inplace = True)
    return data

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
        sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType FROM tblRaw WHERE FreqCode == '%s' AND tblRaw.recID == '%s';"%(i,site)
        self.histDF = pd.read_sql(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        sql = 'SELECT PulseRate,MortRate FROM tblMasterTag WHERE FreqCode == "%s"'%(i)
        rates = pd.read_sql(sql,con = conn)
        self.rates = rates
        sql = 'SELECT FreqCode FROM tblMasterTag'
        allTags = pd.read_sql(sql,con = conn)
        sql = 'SELECT * FROM tblAlgParams'
        algParams = pd.read_sql(sql,con = conn)
        recType = self.histDF.RecType.unique()
        c.close()
        # do some data management when importing training dataframe
        self.histDF['recID1'] = np.repeat(site,len(self.histDF))
        self.histDF['timeStamp'] = pd.to_datetime(self.histDF['timeStamp'])
        self.histDF[['Power','Epoch','noiseRatio',]] = self.histDF[['Power','Epoch','noiseRatio']].apply(pd.to_numeric)                  # sometimes we import from SQLite and our number fields are objects, fuck that noise, let's make sure we are good
        self.histDF['RowSeconds'] = self.histDF['Epoch']
        self.histDF.sort_values(by = 'Epoch', inplace = True)
        self.histDF.set_index('Epoch', drop = False, inplace = True)
        self.histDF = self.histDF.drop_duplicates(subset = 'timeStamp')
        # set some object variables
        self.i = i
        self.site = site
        self.projectDB = projectDB
        self.scratchWS = scratchWS
        self.det = algParams.at[0,'det']
        self.duration = float(algParams.at[0,'duration'])
        self.studyTags = allTags.FreqCode.values
        self.recType = self.histDF.RecType.unique()

        # for training data, we know the tag's detection class ahead of time,
        # if the tag is in the study tag list, it is a known detection class, if
        # it is a beacon tag, it is definite, if it is a study tag, it's plausible
        if self.i in self.studyTags:
            self.plausible = 1
        else:
            self.plausible = 0
        # get rate
        if len(rates.PulseRate.values) > 0:
            self.PulseRate = rates.at[0,'PulseRate']
        else:
            self.PulseRate = 3.0
        if np.any(rates.MortRate.values == None) or len(rates.MortRate.values) == 0:
            self.MortRate = 9999.0
        else:
            self.MortRate = rates.at[0,'MortRate']

        # create a list of factors to search for series hit
        self.alive_factors = np.arange(self.PulseRate,3600,self.PulseRate)
        self.dead_factors = np.arange(self.MortRate,3600,self.MortRate)

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

    # calculate predictors
    histDF['Detection'] = np.repeat(trainee.plausible,len(histDF))
    histDF['lag'] = histDF.Epoch.diff().abs()                                      # calculate the difference in seconds until the next detection
    histDF['lagDiff'] = histDF.lag.diff()
    histDF.lagDiff.fillna(999999999,inplace = True)
    histDF = detHist(histDF,trainee.PulseRate,trainee.det,training = True)             # calculate detection history
    histDF['seriesHit'] = histDF['lag'].apply(lambda x: 1 if x in trainee.alive_factors else 0)
    histDF.set_index('timeStamp',inplace = True, drop = True)
    histDF.drop(['recID1','RowSeconds'],axis = 1, inplace = True)
    histDF['Seconds'] = histDF.index.hour * 3600 + histDF.index.minute * 60 + histDF.index.second

    histDF.to_csv(os.path.join(scratchWS,'%s_%s.csv'%(i,site)))
    del histDF

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
    def __init__(self,i,site,classifyFields,projectDB,scratchWS,training_data,informed_prior = True,training = None, reclass_iter = None):
        '''when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).
        '''
        conn = sqlite3.connect(projectDB, timeout=30.0)
        c = conn.cursor()
        if reclass_iter == None:
            sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType FROM tblRaw WHERE FreqCode == '%s' AND recID == '%s';"%(i,site)
            self.histDF = pd.read_sql(sql,con = conn,coerce_float  = True)


        else:
            sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType FROM tblClassify_%s_%s WHERE FreqCode == '%s' AND test == '1';"%(site,reclass_iter-1,i)
            self.histDF = pd.read_sql(sql,con = conn,coerce_float  = True)

        sql = 'SELECT PulseRate,MortRate FROM tblMasterTag WHERE FreqCode == "%s"'%(i)
        rates = pd.read_sql(sql,con = conn)
        sql = 'SELECT FreqCode FROM tblMasterTag'
        allTags = pd.read_sql(sql,con = conn)
        sql = 'SELECT * FROM tblAlgParams'
        algParams = pd.read_sql(sql,con = conn)
        sql = 'SELECT RecType FROM tblMasterReceiver WHERE recID == "%s"'%(site)
        recType = pd.read_sql(sql,con = conn).RecType.values[0]
        c.close()
        # do some data management when importing training dataframe
        self.histDF['recID1'] = np.repeat(site,len(self.histDF))
        self.histDF['timeStamp'] = pd.to_datetime(self.histDF['timeStamp'])
        self.histDF[['Power','Epoch']] = self.histDF[['Power','Epoch']].apply(pd.to_numeric)                  # sometimes we import from SQLite and our number fields are objects, fuck that noise, let's make sure we are good
        self.histDF['RowSeconds'] = self.histDF['Epoch']
        self.histDF.sort_values(by = 'Epoch', inplace = True)
        self.histDF.set_index('Epoch', drop = False, inplace = True)
        self.histDF = self.histDF.drop_duplicates(subset = 'timeStamp')
        self.trainDF = training_data
        # set some object variables
        self.fields = classifyFields
        self.i = i
        self.site = site
        self.projectDB = projectDB
        self.scratchWS = scratchWS
        self.det = algParams.at[0,'det']
        self.duration = float(algParams.at[0,'duration'])
        self.studyTags = allTags.FreqCode.values
        self.recType = recType
        self.PulseRate = rates.at[0,'PulseRate']
        if np.any(rates.MortRate.values == None) or len(rates.MortRate.values) == 0:
            self.MortRate = 9999.0
        else:
            self.MortRate = rates.at[0,'MortRate']

        # create a list of factors to search for series hit
        self.alive_factors = np.arange(self.PulseRate,3600,self.PulseRate)
        self.dead_factors = np.arange(self.MortRate,3600,self.MortRate)
        self.informed = informed_prior
        self.reclass_iter = reclass_iter
        if training != None:
            self.trainingDB = training
        else:
            self.trainingDB = projectDB

def evidence(classify_object):
    '''calculates evidence of observed data occuring while maintaining a-la carte
    likelihood functionality'''
    # get fields from classify object
    fields = classify_object.fields

    # associate field to count with a couple of dictionaries
    trueFields = {'conRecLength':'conRecLengthCountT',
                  'consDet':'consDetCountT',
                  'hitRatio':'hitRatioCountT',
                  'noiseRatio':'noiseCount_T',
                  'seriesHit':'seriesHitCountT',
                  'power':'powerCount_T',
                  'lagDiff':'lagDiffCount_T'}

    falseFields = {'conRecLength':'conRecLengthCountF',
                   'consDet':'consDetCountF',
                   'hitRatio':'hitRatioCountF',
                   'noiseRatio':'noiseCount_F',
                   'seriesHit':'seriesHitCountF',
                   'power':'powerCount_F',
                   'lagDiff':'lagDiffCount_F'}

    true_count = classify_object.testDat.priorCount_T.values[0]
    false_count = classify_object.testDat.priorCount_F.values[0]

    total_count = true_count + false_count

    if len(fields) == 1:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count

    elif len(fields) == 2:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count

    elif len(fields) == 3:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[2]]] + classify_object.testDat[falseFields[fields[2]]] + 1)/total_count

    elif len(fields) == 4:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[2]]] + classify_object.testDat[falseFields[fields[2]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[3]]] + classify_object.testDat[falseFields[fields[3]]] + 1)/total_count

    elif len(fields) == 5:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[2]]] + classify_object.testDat[falseFields[fields[2]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[3]]] + classify_object.testDat[falseFields[fields[3]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[4]]] + classify_object.testDat[falseFields[fields[4]]] + 1)/total_count

    elif len(fields) == 6:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[2]]] + classify_object.testDat[falseFields[fields[2]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[3]]] + classify_object.testDat[falseFields[fields[3]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[4]]] + classify_object.testDat[falseFields[fields[4]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[5]]] + classify_object.testDat[falseFields[fields[5]]] + 1)/total_count

    elif len(fields) == 7:
        return (classify_object.testDat[trueFields[fields[0]]] + classify_object.testDat[falseFields[fields[0]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[1]]] + classify_object.testDat[falseFields[fields[1]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[2]]] + classify_object.testDat[falseFields[fields[2]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[3]]] + classify_object.testDat[falseFields[fields[3]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[4]]] + classify_object.testDat[falseFields[fields[4]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[5]]] + classify_object.testDat[falseFields[fields[5]]] + 1)/total_count * \
            (classify_object.testDat[trueFields[fields[6]]] + classify_object.testDat[falseFields[fields[6]]] + 1)/total_count


def likelihood(assumption,classify_object,status = 'A'):
    '''calculates likelihood based on true or false assumption and fields provided to classify object'''
    fields = classify_object.fields
    if status == 'A':
        trueFields = {'conRecLength':'LconRecT_A',
                      'consDet':'LconsDetT_A',
                      'hitRatio':'LHitRatioT_A',
                      'noiseRatio':'LnoiseT',
                      'seriesHit':'LseriesHitT_A',
                      'power':'LPowerT',
                      'lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF_A',
                       'consDet':'LconsDetF_A',
                       'hitRatio':'LHitRatioF_A',
                       'noiseRatio':'LnoiseF',
                       'seriesHit':'LseriesHitF_A',
                       'power':'LPowerF',
                       'lagDiff':'LlagF'}
    elif status == 'M':
        trueFields = {'conRecLength':'LconRecT_M',
                      'consDet':'LconsDetT_M',
                      'hitRatio':'LHitRatioT_M',
                      'noiseRatio':'LnoiseT',
                      'seriesHit':'LseriesHitT_M',
                      'power':'LPowerT',
                      'lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF_M',
                       'consDet':'LconsDetF_M',
                       'hitRatio':'LHitRatioF_M',
                       'noiseRatio':'LnoiseF',
                       'seriesHit':'LseriesHitF_M',
                       'power':'LPowerF',
                       'lagDiff':'LlagF'}
    else:
        trueFields = {'conRecLength':'LconRecT',
                      'consDet':'LconsDetT',
                      'hitRatio':'LHitRatioT',
                      'noiseRatio':'LnoiseT',
                      'seriesHit':'LseriesHitT',
                      'power':'LPowerT',
                      'lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF',
                       'consDet':'LconsDetF',
                       'hitRatio':'LHitRatioF',
                       'noiseRatio':'LnoiseF',
                       'seriesHit':'LseriesHitF',
                       'power':'LPowerF',
                       'lagDiff':'LlagF'}

    if status == 'cross':
        if assumption == True:
            if len(fields) == 1:
                return classify_object.testDat[trueFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]] * \
                        classify_object.testDat[trueFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]] * \
                        classify_object.testDat[trueFields[fields[2]]]  * \
                            classify_object.testDat[trueFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]] * \
                        classify_object.testDat[trueFields[fields[2]]] * \
                            classify_object.testDat[trueFields[fields[3]]] * \
                                classify_object.testDat[trueFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]] * \
                        classify_object.testDat[trueFields[fields[2]]] * \
                            classify_object.testDat[trueFields[fields[3]]] * \
                                classify_object.testDat[trueFields[fields[4]]] * \
                                    classify_object.testDat[trueFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.testDat[trueFields[fields[0]]] * \
                    classify_object.testDat[trueFields[fields[1]]] * \
                        classify_object.testDat[trueFields[fields[2]]] * \
                            classify_object.testDat[trueFields[fields[3]]] * \
                                classify_object.testDat[trueFields[fields[4]]] * \
                                    classify_object.testDat[trueFields[fields[5]]] * \
                                        classify_object.testDat[trueFields[fields[6]]]

        elif assumption == False:
            if len(fields) == 1:
                return classify_object.testDat[falseFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]] * \
                        classify_object.testDat[falseFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]] * \
                        classify_object.testDat[falseFields[fields[2]]] * \
                            classify_object.testDat[falseFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]] * \
                        classify_object.testDat[falseFields[fields[2]]] * \
                            classify_object.testDat[falseFields[fields[3]]] * \
                                classify_object.testDat[falseFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]] * \
                        classify_object.testDat[falseFields[fields[2]]] * \
                            classify_object.testDat[falseFields[fields[3]]] * \
                                classify_object.testDat[falseFields[fields[4]]] * \
                                    classify_object.testDat[falseFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.testDat[falseFields[fields[0]]] * \
                    classify_object.testDat[falseFields[fields[1]]] * \
                        classify_object.testDat[falseFields[fields[2]]] * \
                            classify_object.testDat[falseFields[fields[3]]] * \
                                classify_object.testDat[falseFields[fields[4]]] * \
                                    classify_object.testDat[falseFields[fields[5]]] * \
                                        classify_object.testDat[falseFields[fields[6]]]

    else:
        if assumption == True:
            if len(fields) == 1:
                return classify_object.histDF[trueFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]] * \
                        classify_object.histDF[trueFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]] * \
                        classify_object.histDF[trueFields[fields[2]]] * \
                            classify_object.histDF[trueFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]] * \
                        classify_object.histDF[trueFields[fields[2]]] * \
                            classify_object.histDF[trueFields[fields[3]]] * \
                                classify_object.histDF[trueFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]] * \
                        classify_object.histDF[trueFields[fields[2]]] * \
                            classify_object.histDF[trueFields[fields[3]]] * \
                                classify_object.histDF[trueFields[fields[4]]] * \
                                    classify_object.histDF[trueFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.histDF[trueFields[fields[0]]] * \
                    classify_object.histDF[trueFields[fields[1]]] * \
                        classify_object.histDF[trueFields[fields[2]]] * \
                            classify_object.histDF[trueFields[fields[3]]] * \
                                classify_object.histDF[trueFields[fields[4]]] * \
                                    classify_object.histDF[trueFields[fields[5]]] * \
                                        classify_object.histDF[trueFields[fields[6]]]

        elif assumption == False:
            if len(fields) == 1:
                return classify_object.histDF[falseFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]] * \
                        classify_object.histDF[falseFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]] * \
                        classify_object.histDF[falseFields[fields[2]]] * \
                            classify_object.histDF[falseFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]] * \
                        classify_object.histDF[falseFields[fields[2]]] * \
                            classify_object.histDF[falseFields[fields[3]]] * \
                                classify_object.histDF[falseFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]] * \
                        classify_object.histDF[falseFields[fields[2]]] * \
                            classify_object.histDF[falseFields[fields[3]]] * \
                                classify_object.histDF[falseFields[fields[4]]] * \
                                    classify_object.histDF[falseFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.histDF[falseFields[fields[0]]] * \
                    classify_object.histDF[falseFields[fields[1]]] * \
                        classify_object.histDF[falseFields[fields[2]]] * \
                            classify_object.histDF[falseFields[fields[3]]] * \
                                classify_object.histDF[falseFields[fields[4]]] * \
                                    classify_object.histDF[falseFields[fields[5]]] * \
                                        classify_object.histDF[falseFields[fields[6]]]

def create_training_data(site,projectDB,reclass_iter = None, rec_list = None):
    '''Function creates training dataset for current round of classification -
    if we only do this once, this time suck goes away'''

    #get training data
    '''
    Reclassification code contributed by T Castro-Santos
    '''
    conn = sqlite3.connect(projectDB)#, timeout=30.0)
    sql = 'SELECT RecType FROM tblMasterReceiver WHERE recID == "%s"'%(site)
    recType = pd.read_sql(sql,con = conn).RecType.values[0]

    c = conn.cursor()
    if reclass_iter == None:
        if rec_list == None:
            sql = "SELECT * FROM tblTrain"#" WHERE RecType == '%s'"%(recType)
            trainDF = pd.read_sql(sql,con = conn)
            trainDF = trainDF[trainDF.recType == recType]
        else:
            trainDF = pd.DataFrame()
            for i in rec_list:
                sql = "SELECT * FROM tblTrain WHERE recID == '%s'"%(i)
                trainDF = trainDF.append(pd.read_sql(sql,con = conn))
    else:
        if rec_list == None:
            trainDF = pd.read_sql("select * from tblTrain",con=conn)#This will read in tblTrain and create a pandas dataframe
            trainDF = trainDF[trainDF.recType == recType]
        else:
            trainDF = pd.DataFrame()
            for i in rec_list:
                trainDF = trainDF.append(pd.read_sql("select * from tblTrain WHERE recID == '%s'"%(i),con=conn))#This will read in tblTrain and create a pandas dataframe

        classDF = pd.read_sql('''SELECT test,
                              FreqCode,
                              Power,
                              noiseRatio,
                              lag,
                              lagDiff,
                              conRecLength_A,
                              consDet_A,
                              detHist_A,
                              hitRatio_A,
                              seriesHit_A,
                              conRecLength_M,
                              consDet_M,
                              detHist_M,
                              hitRatio_M,
                              seriesHit_M,
                              postTrue_A,
                              postTrue_M,
                              timeStamp,
                              Epoch,
                              RowSeconds,
                              recID,
                              RecType,
                              ScanTime FROM tblClassify_%s_%s'''%(site,reclass_iter-1),
                              con=conn)
        classDF.drop(['conRecLength_M','consDet_M','detHist_M','hitRatio_M','seriesHit_M'],
                     axis = 1,
                     inplace = True)
        classDF.rename(columns = {'conRecLength_A':'conRecLength',
                                  'consDet_A':'consDet',
                                  'detHist_A':'detHist',
                                  'hitRatio_A':'hitRatio',
                                  'seriesHit_A':'seriesHit'},
                       inplace = True)

        trainDF = trainDF[trainDF.Detection==0]
        classDF = classDF[classDF.test==1]
        classDF['Channels']=np.repeat(1,len(classDF))
        classDF.rename(columns={"test":"Detection","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
        #Next we append the classdf to the traindf
        trainDF = trainDF.append(classDF)
        #trainDF.to_sql('tblTrain_%s'%(classify_object.reclass_iter),index=False,con=conn)#we might want to allow for further iterations
    print ("Training dataset created")
    c.close()
    return trainDF

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
    trainDF = classify_object.trainDF
    if len(classify_object.histDF) > 0:

        # calculate parameters
        classify_object.histDF['lag'] = classify_object.histDF.Epoch.diff().abs()
        classify_object.histDF['lagDiff'] = classify_object.histDF.lag.diff()
        classify_object.histDF['seriesHit_A'] = classify_object.histDF['lag'].apply(lambda x: 1 if x in classify_object.alive_factors else 0)
        classify_object.histDF['seriesHit_M'] = classify_object.histDF['lag'].apply(lambda x: 1 if x in classify_object.dead_factors else 0)
        classify_object.histDF = detHist(classify_object.histDF,classify_object.PulseRate,classify_object.det)
        classify_object.histDF = detHist(classify_object.histDF,classify_object.MortRate,classify_object.det,'M')
        classify_object.histDF['powerBin'] = (classify_object.histDF.Power//5)*5
        classify_object.histDF['noiseBin'] = (classify_object.histDF.noiseRatio//.1)*.1
        classify_object.histDF['lagDiffBin'] = (classify_object.histDF.lagDiff//classify_object.PulseRate)*classify_object.PulseRate

        # Update Data Types - they've got to match or the merge doesn't work!!!!
        trainDF.Detection = trainDF.Detection.astype(int)
        trainDF.FreqCode = trainDF.FreqCode.astype(str)
        trainDF['seriesHit'] = trainDF.seriesHit.astype(int)
        trainDF['consDet'] = trainDF.consDet.astype(int)
        trainDF['detHist'] = trainDF.detHist.astype(str)
        trainDF['noiseRatio'] = trainDF.noiseRatio.astype(float).round(4)
        trainDF['conRecLength'] = trainDF.conRecLength.astype(int)
        trainDF['hitRatio'] = trainDF.hitRatio.astype(float).round(4)
        trainDF['powerBin'] = (trainDF.Power//5)*5
        trainDF['noiseBin'] = (trainDF.noiseRatio//.1)*.1
        trainDF['lagDiffBin'] = (trainDF.lagDiff//classify_object.PulseRate)*classify_object.PulseRate

        # making sure our classify object data types match
        classify_object.histDF.seriesHit_A = classify_object.histDF.seriesHit_A.astype(np.int64)
        classify_object.histDF.seriesHit_M = classify_object.histDF.seriesHit_M.astype(np.int64)
        classify_object.histDF.consDet_A = classify_object.histDF.consDet_A.astype(int)
        classify_object.histDF.consDet_M = classify_object.histDF.consDet_M.astype(int)
        classify_object.histDF.detHist_A = classify_object.histDF.detHist_A.astype(str)
        classify_object.histDF.detHist_M = classify_object.histDF.detHist_M.astype(str)
        classify_object.histDF.conRecLength_A = classify_object.histDF.conRecLength_A.astype(int)
        classify_object.histDF.conRecLength_M = classify_object.histDF.conRecLength_M.astype(int)
        classify_object.histDF.noiseRatio = classify_object.histDF.noiseRatio.astype(float).round(4)
        classify_object.histDF['HT'] = np.repeat(1,len(classify_object.histDF))
        classify_object.histDF['HF'] = np.repeat(0,len(classify_object.histDF))
        classify_object.histDF.hitRatio_A = classify_object.histDF.hitRatio_A.astype(float).round(4)
        classify_object.histDF.hitRatio_M = classify_object.histDF.hitRatio_M.astype(float).round(4)


        # Make a Count of the predictor variables and join to training data frame - For ALIVE Strings
        seriesHitCount = trainDF.groupby(['Detection','seriesHit'])['seriesHit'].\
            count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitACountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).\
            reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = seriesHitCount,
                                          how = u'left',
                                          left_on = ['HT','seriesHit_A'],
                                          right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitACountT':'seriesHitACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = seriesHitCount,
                                          how = u'left',
                                          left_on = ['HF','seriesHit_A'],
                                          right_on = ['HF','seriesHit'])
        classify_object.histDF.drop(labels = ['seriesHit_x','seriesHit_y'],
                                    axis = 1,
                                    inplace = True)

        # count the number of instances of consective detections by detection class and write to data frame
        consDetCount = trainDF.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetACountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = consDetCount,
                                          how = u'left',
                                          left_on = ['HT','consDet_A'],
                                          right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetACountT':'consDetACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = consDetCount,
                                          how = u'left',
                                          left_on = ['HF','consDet_A'],
                                          right_on = ['HF','consDet'])
        classify_object.histDF.drop(labels = ['consDet_x','consDet_y'], axis = 1, inplace = True)

        # count the number of instances of certain detection histories by detection class and write to data frame
        detHistCount = trainDF.groupby(['Detection','detHist'])['detHist'].count()
        detHistCount = pd.Series(detHistCount, name = 'detHistACountT')
        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = detHistCount,
                                          how = u'left',
                                          left_on = ['HT','detHist_A'],
                                          right_on =['HT','detHist'])
        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistACountT':'detHistACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = detHistCount,
                                          how = u'left',
                                          left_on = ['HF','detHist_A'],
                                          right_on =['HF','detHist'])
        classify_object.histDF.drop(labels = ['detHist_x','detHist_y'], axis = 1, inplace = True)

        # count the number of instances of consecutive record lengths by detection class and write to data frame
        conRecLengthCount = trainDF.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthACountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = conRecLengthCount,
                                          how = u'left',
                                          left_on = ['HT','conRecLength_A'],
                                          right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthACountT':'conRecLengthACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = conRecLengthCount,
                                          how = u'left',
                                          left_on = ['HF','conRecLength_A'],
                                          right_on = ['HF','conRecLength'])
        classify_object.histDF.drop(labels = ['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)

        # count the number of instances of hit ratios by detection class and write to data frame
        hitRatioCount = trainDF.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioACountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = hitRatioCount,
                                          how = u'left',
                                          left_on = ['HT','hitRatio_A'],
                                          right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioACountT':'hitRatioACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = hitRatioCount,
                                          how = u'left',
                                          left_on = ['HF','hitRatio_A'],
                                          right_on = ['HF','hitRatio'])
        classify_object.histDF.drop(labels = ['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # Power
        powerCount = trainDF.groupby(['Detection','powerBin'])['powerBin'].count()
        powerCount = pd.Series(powerCount, name = 'powerCount_T')
        powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = powerCount,
                                          how = u'left',
                                          left_on = ['HT','powerBin'],
                                          right_on = ['HT','powerBin'])
        powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = powerCount,
                                          how = u'left',
                                          left_on = ['HF','powerBin'],
                                          right_on = ['HF','powerBin'])

        # Make a Count of the predictor variables and join to training data frame
        seriesHitCount = trainDF.groupby(['Detection','seriesHit'])['seriesHit'].count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitMCountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = seriesHitCount,
                                          how = u'left',
                                          left_on = ['HT','seriesHit_M'],
                                          right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitMCountT':'seriesHitMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = seriesHitCount,
                                          how = u'left',
                                          left_on = ['HF','seriesHit_M'],
                                          right_on = ['HF','seriesHit'])
        classify_object.histDF.drop(labels = ['seriesHit_x','seriesHit_y'], axis = 1, inplace = True)

        # count the number of instances of consective detections by detection class and write to data frame
        consDetCount = trainDF.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetMCountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = consDetCount,
                                          how = u'left',
                                          left_on = ['HT','consDet_M'],
                                          right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetMCountT':'consDetMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = consDetCount,
                                          how = u'left',
                                          left_on = ['HF','consDet_M'],
                                          right_on = ['HF','consDet'])
        classify_object.histDF.drop(labels = ['consDet_x','consDet_y'], axis = 1, inplace = True)

        # count the number of instances of certain detection histories by detection class and write to data frame
        detHistCount = trainDF.groupby(['Detection','detHist'])['detHist'].count()
        detHistCount = pd.Series(detHistCount, name = 'detHistMCountT')
        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = detHistCount,
                                          how = u'left',
                                          left_on = ['HT','detHist_M'],
                                          right_on =['HT','detHist'])
        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistMCountT':'detHistMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = detHistCount,
                                          how = u'left',
                                          left_on = ['HF','detHist_M'],
                                          right_on =['HF','detHist'])
        classify_object.histDF.drop(labels = ['detHist_x','detHist_y'], axis = 1, inplace = True)

        # count the number of instances of consecutive record lengths by detection class and write to data frame
        conRecLengthCount = trainDF.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthMCountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = conRecLengthCount,
                                          how = u'left',
                                          left_on = ['HT','conRecLength_M'],
                                          right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthMCountT':'conRecLengthMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = conRecLengthCount,
                                          how = u'left',
                                          left_on = ['HF','conRecLength_M'],
                                          right_on = ['HF','conRecLength'])
        classify_object.histDF.drop(labels = ['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)

        # count the number of instances of hit ratios by detection class and write to data frame
        hitRatioCount = trainDF.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioMCountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = hitRatioCount,
                                          how = u'left',
                                          left_on = ['HT','hitRatio_M'],
                                          right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioMCountT':'hitRatioMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = hitRatioCount,
                                          how = u'left',
                                          left_on = ['HF','hitRatio_M'],
                                          right_on = ['HF','hitRatio'])
        classify_object.histDF.drop(labels = ['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # NoiseRatio
        noiseCount = trainDF.groupby(['Detection','noiseBin'])['noiseBin'].count()
        noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
        noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = noiseCount,
                                          how = u'left',
                                          left_on = ['HT','noiseBin'],
                                          right_on = ['HT','noiseBin'])
        noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = noiseCount,
                                          how = u'left',
                                          left_on = ['HF','noiseBin'],
                                          right_on = ['HF','noiseBin'])

        # Lag Bin
        lagCount = trainDF.groupby(['Detection','lagDiffBin'])['lagDiffBin'].count()
        lagCount = pd.Series(lagCount, name = 'lagDiffCount_T')
        lagCount = pd.DataFrame(lagCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = lagCount,
                                          how = u'left',
                                          left_on = ['HT','lagDiffBin'],
                                          right_on = ['HT','lagDiffBin'])
        lagCount = lagCount.rename(columns = {'HT':'HF','lagDiffCount_T':'lagDiffCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF,
                                          right = lagCount,
                                          how = u'left',
                                          left_on = ['HF','lagDiffBin'],
                                          right_on = ['HF','lagDiffBin'])

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
        trueCount = priorCountT + 1.0
        falseCount = priorCountF + 1.0
        classify_object.histDF['priorCount_T'] = np.repeat(priorCountT,len(classify_object.histDF))
        classify_object.histDF['priorCount_F'] = np.repeat(priorCountF,len(classify_object.histDF))
        classify_object.histDF['LDenomCount_T'] = np.repeat(trueCount,len(classify_object.histDF))
        classify_object.histDF['LDenomCount_F'] = np.repeat(falseCount,len(classify_object.histDF))


        # calculation of the probability of a false positive given the data assuming fish is alive

        # caalculate the prior probability of a false detection from the training dataset
        classify_object.histDF['priorF'] = round(priorCountF/float(len(trainDF)),5)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LconRecF_A'] = (classify_object.histDF['conRecLengthACountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LseriesHitF_A'] = (classify_object.histDF['seriesHitACountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetF_A'] = (classify_object.histDF['consDetACountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioF_A'] = (classify_object.histDF['hitRatioACountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerF'] = (classify_object.histDF['powerCount_F'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseF'] = (classify_object.histDF['noiseCount_F'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagF'] = (classify_object.histDF['lagDiffCount_F'] + 1)\
           /classify_object.histDF['LDenomCount_F']

        # calculation of the probability of a true detection given the data assuming fish is alive

        # calculate the prior probability of a true detection from the training dataset
        classify_object.histDF['priorT'] = round(priorCountT/float(len(trainDF)),5)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LconRecT_A'] = (classify_object.histDF['conRecLengthACountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LseriesHitT_A'] = (classify_object.histDF['seriesHitACountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetT_A'] = (classify_object.histDF['consDetACountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioT_A'] = (classify_object.histDF['hitRatioACountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerT'] = (classify_object.histDF['powerCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseT'] = (classify_object.histDF['noiseCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagT'] = (classify_object.histDF['lagDiffCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']

        # calculation of the probability of a false positive given the data assuming fish is dead

        # calculate the prior probability of a false detection from the training dataset
        classify_object.histDF['priorF'] = round(priorCountF/float(len(trainDF)),5)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LconRecF_M'] = (classify_object.histDF['conRecLengthMCountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LseriesHitF_M'] = (classify_object.histDF['seriesHitMCountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetF_M'] = (classify_object.histDF['consDetMCountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioF_M'] = (classify_object.histDF['hitRatioMCountF'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerF'] = (classify_object.histDF['powerCount_F'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseF'] = (classify_object.histDF['noiseCount_F'] + 1)\
            /classify_object.histDF['LDenomCount_F']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagF'] = (classify_object.histDF['lagDiffCount_F'] + 1)\
            /classify_object.histDF['LDenomCount_F']

        # calculation of the probability of a true detection given the data assuming fish is dead

    	# calculate the prior probability of a true detection from the training dataset
        classify_object.histDF['priorT'] = round(priorCountT/float(len(trainDF)),5)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LconRecT_M'] = (classify_object.histDF['conRecLengthMCountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LseriesHitT_M'] = (classify_object.histDF['seriesHitMCountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetT_M'] = (classify_object.histDF['consDetMCountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioT_M'] = (classify_object.histDF['hitRatioMCountT'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerT'] = (classify_object.histDF['powerCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseT'] = (classify_object.histDF['noiseCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagT'] = (classify_object.histDF['lagDiffCount_T'] + 1)\
            /classify_object.histDF['LDenomCount_T']

        # Calculate the likelihood of each hypothesis being true
        classify_object.histDF['LikelihoodTrue_A'] = likelihood(True,classify_object, status = 'A')
        classify_object.histDF['LikelihoodFalse_A'] = likelihood(False,classify_object, status = 'A')
        classify_object.histDF['LikelihoodTrue_M'] = likelihood(True,classify_object, status = 'M')
        classify_object.histDF['LikelihoodFalse_M'] = likelihood(False,classify_object, status = 'M')

        classify_object.histDF['logLikelihoodRatio_A'] = np.log10(classify_object.histDF.LikelihoodTrue_A.values/classify_object.histDF.LikelihoodFalse_A.values)
        classify_object.histDF['logLikelihoodRatio_M'] = np.log10(classify_object.histDF.LikelihoodTrue_M.values/classify_object.histDF.LikelihoodFalse_M.values)
        # Calculate the posterior probability of each Hypothesis occuring
        if classify_object.informed == True:
            classify_object.histDF['postTrue_A'] = classify_object.histDF['priorT'] * classify_object.histDF['LikelihoodTrue_A']
            classify_object.histDF['postFalse_A'] = classify_object.histDF['priorF'] * classify_object.histDF['LikelihoodFalse_A']
            classify_object.histDF['postTrue_M'] = classify_object.histDF['priorT'] * classify_object.histDF['LikelihoodTrue_M']
            classify_object.histDF['postFalse_M'] = classify_object.histDF['priorF'] * classify_object.histDF['LikelihoodFalse_M']
        else:
            classify_object.histDF['postTrue_A'] = 0.5 * classify_object.histDF['LikelihoodTrue_A']
            classify_object.histDF['postFalse_A'] = 0.5 * classify_object.histDF['LikelihoodFalse_A']
            classify_object.histDF['postTrue_M'] = 0.5 * classify_object.histDF['LikelihoodTrue_M']
            classify_object.histDF['postFalse_M'] = 0.5 * classify_object.histDF['LikelihoodFalse_M']

        # apply the MAP hypothesis
        #classify_object.histDF['test'] = classify_object.histDF.apply(MAP,axis =1)
        classify_object.histDF.loc[(classify_object.histDF.postTrue_A >= classify_object.histDF.postFalse_A) | (classify_object.histDF.postTrue_M >= classify_object.histDF.postFalse_M),'test'] = True
        classify_object.histDF.loc[(classify_object.histDF.postTrue_A < classify_object.histDF.postFalse_A) & (classify_object.histDF.postTrue_M < classify_object.histDF.postFalse_M),'test'] = False
        classify_object.histDF.to_csv(os.path.join(classify_object.scratchWS,"%s.csv"%(classify_object.i)))
        del trainDF

def classDatAppend(site,inputWS,projectDB,reclass_iter = 1):
    # As soon as I figure out how to do this function is moot.
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        out_name = 'tblClassify_%s_%s'%(site,reclass_iter)
        dat = pd.read_csv(os.path.join(inputWS,f),dtype = {"detHist_A":str,"detHist_M":str})
        #dat.drop(['recID1'],axis = 1,inplace = True)
        dtype = {'FreqCode':'TEXT',
                 'Epoch': 'INTEGER',
                 'recID':'TEXT',
                 'timeStamp':'TIMESTAMP',
                 'Power':'REAL',
                 'ScanTime':'REAL',
                 'Channels':'REAL',
                 'RecType':'TEXT',
                 'recID1':'TEXT',
                 'RowSeconds':'REAL',
                 'lag':'REAL',
                 'lagDiff':'REAL',
                 'noiseRatio':'REAL',
                 'seriesHit_A':'INTEGER',
                 'seriesHit_M':'INTEGER',
                 'consDet_A':'INTEGER',
                 'hitRatio_A':'REAL',
                 'detHist_A':'TEXT',
                 'conRecLength_A':'INTEGER',
                 'consDet_M':'INTEGER',
                 'hitRatio_M':'REAL',
                 'detHist_M':'TEXT',
                 'conRecLength_M':'INTEGER',
                 'powerBin':'REAL',
                 'lagDiffBin':'REAL',
                 'HT':'INTEGER',
                 'HF':'INTEGER',
                 'seriesHitACountT':'INTEGER',
                 'seriesHitACountF':'INTEGER',
                 'consDetACountT':'INTEGER',
                 'consDetACountF':'INTEGER',
                 'detHistACountT':'INTEGER',
                 'detHistACountF':'INTEGER',
                 'conRecLengthACountT':'INTEGER',
                 'conRecLengthACountF':'INTEGER',
                 'hitRatioACountT':'INTEGER',
                 'hitRatioACountF':'INTEGER',
                 'powerCount_T':'INTEGER',
                 'powerCount_F':'INTEGER',
                 'noiseCount_T':'INTEGER',
                 'noiseCount_F':'INTEGER',
                 'seriesHitMCountT':'INTEGER',
                 'seriesHitMCountF':'INTEGER',
                 'consDetMCountT':'INTEGER',
                 'consDetMCountF':'INTEGER',
                 'detHistMCountT':'INTEGER',
                 'detHistMCountF':'INTEGER',
                 'conRecLengthMCountT':'INTEGER',
                 'conRecLengthMCountF':'INTEGER',
                 'hitRatioMCountT':'INTEGER',
                 'hitRatioMCountF':'INTEGER',
                 'lagDiffCount_T':'INTEGER',
                 'lagDiffCount_F':'INTEGER',
                 'priorCount_T':'INTEGER',
                 'priorCount_F':'INTEGER',
                 'LDenomCount_T':'INTEGER',
                 'LDenomCount_F':'INTEGER',
                 'priorF':'REAL',
                 'LconRecF_A':'REAL',
                 'LseriesHitF_A':'REAL',
                 'LconsDetF_A':'REAL',
                 'LHitRatioF_A':'REAL',
                 'LPowerF':'REAL',
                 'LnoiseF':'REAL',
                 'LlagF':'REAL',
                 'priorT':'REAL',
                 'LconRecT_A':'REAL',
                 'LseriesHitT_A':'REAL',
                 'LconsDetT_A':'REAL',
                 'LHitRatioT_A':'REAL',
                 'LPowerT':'REAL',
                 'LnoiseT':'REAL',
                 'LlagT':'REAL',
                 'LconRecF_M':'REAL',
                 'LseriesHitF_M':'REAL',
                 'LconsDetF_M':'REAL',
                 'LHitRatioF_M':'REAL',
                 'LconRecT_M':'REAL',
                 'LseriesHitT_M':'REAL',
                 'LconsDetT_M':'REAL',
                 'LHitRatioT_M':'REAL',
                 'LikelihoodTrue_A':'REAL',
                 'LikelihoodFalse_A':'REAL',
                 'LikelihoodTrue_M':'REAL',
                 'LikelihoodFalse_M':'REAL',
                 'logLikelihoodRatio_A':'REAL',
                 'logLikelihoodRatio_M':'REAL',
                 'postTrue_A':'REAL',
                 'postFalse_A':'REAL',
                 'postTrue_M':'REAL',
                 'postFalse_M':'REAL',
                 'test':'INTEGER'}
        dat.to_sql(out_name,con = conn,index = False, if_exists = 'append', chunksize = 1000, dtype = dtype)
        os.remove(os.path.join(inputWS,f))
        del dat
    c.execute('''CREATE INDEX idx_combined_%s_%s ON tblClassify_%s_%s (recID,FreqCode,Epoch)'''%(site,reclass_iter,site,reclass_iter))
    c.close()

def max_iter(rec_list, tables):
    '''Function used by training, classification, and cross validation to find the
    last iteration of a receiver classification.

    Inputs to the function are a list of recievers to iterate through and
    the list of tables contained in the database.'''

    # iterate over the receivers to find the final classification (aka the largest _n)
    max_iter_dict = {} # receiver:max iter
    curr_idx = 0
    rec_iter_list = []
    for i in rec_list:
        max_iter = 1
        while curr_idx < len(tables) - 1:
            for j in tables:
                if i in j[0]:
                    tbl_name = j[0].split('_')
                    rec_iter_list.append(j[0])
                    if int(tbl_name[-1]) >= max_iter:
                        max_iter = int(tbl_name[-1])
                        max_iter_dict[i] = j[0]
                curr_idx = curr_idx + 1
        curr_idx = 0
    del i, j, curr_idx

    return max_iter_dict, rec_iter_list

class cross_validated():
    '''We validate the training data against itself with a cross validated data
    object. To implement the k-fold cross validation procedure, we simply pass
    the number of folds and receiver type we wish to validate.
    '''
    def __init__(self,folds,recType,likelihood_model,projectDB,rec_list = None, train_on = 'Beacon'):
        self.folds = folds
        self.recType = recType
        self.projectDB = projectDB
        self.rec_list = rec_list
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        if train_on == 'Beacon':
            if rec_list != None:
                sql = "SELECT * FROM tblTrain  WHERE recID == '%s'"%(rec_list[0])
                for i in rec_list[1:]:
                    sql = sql + "OR recID == '%s'"%(i)
            else:
                sql = "SELECT * FROM tblTrain WHERE recType == '%s'"%(recType)
            self.trainDF = pd.read_sql_query(sql,con = conn,coerce_float  = True)
            c.close()

        else:
            self.trainDF = pd.DataFrame()
            if rec_list == None:
                recs = pd.read_sql("select * from tblMasterReceiver WHERE recType == '%s'"%(recType),con=conn)['recID'].values
            else:
                recs = self.rec_list

            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()

            # create a maximum iteration dictionary
            max_iter_dict = max_iter(recs,tbls)[0]

            # once we have a hash table of receiver to max classification, extract the classification dataset
            trues = pd.DataFrame()
            falses = pd.DataFrame()
            for i in max_iter_dict:
                trues_i = pd.read_sql('''select test, FreqCode,Power,noiseRatio,
                                       lag,lagDiff,conRecLength_A,consDet_A,
                                       detHist_A,hitRatio_A,seriesHit_A,postTrue_A,
                                       timeStamp,Epoch,RowSeconds,recID,RecType,
                                       ScanTime from %s'''%(max_iter_dict[i]),con=conn)

                trues_i.rename(columns = {'conRecLength_A':'conRecLength',
                                          'consDet_A':'consDet',
                                          'detHist_A':'detHist',
                                          'hitRatio_A':'hitRatio',
                                          'seriesHit_A':'seriesHit'},
                               inplace = True)

                trues_i = trues_i[trues_i.test == 1]
                trues = trues.append(trues_i)
                del trues_i
                print ("Imported Classified Valid from Table %s"%(i))

            # iterate over receivers, extract false positives from tblTrain.
            for i in self.rec_list:
                falses_i = pd.read_sql("select * from tblTrain WHERE recID == '%s'"%(i),con=conn)
                if len(falses_i) > 0:
                    falses_i = falses_i[falses_i.Detection==0]
                    falses = falses.append(falses_i)
                    del falses_i
                    print ("Imported Known False Positive from tblTrain at Receiver %s"%(i))

            print ("Training data created")

            trues['Channels']=np.repeat(1,len(trues))
            trues.rename(columns={"test":"Detection","RowSeconds":"Seconds","RecType":"recType"},inplace=True)
            self.trainDF = self.trainDF.append(trues)
            self.trainDF = self.trainDF.append(falses)
            del trues, falses
            c.close()

        self.k = folds

        # force data types
        self.trainDF.Detection = self.trainDF.Detection.fillna(0)
        self.trainDF.Detection = self.trainDF.Detection.astype(int)
        self.trainDF.FreqCode = self.trainDF.FreqCode.astype(str)
        self.trainDF.seriesHit = self.trainDF.seriesHit.astype(int)
        self.trainDF.consDet = self.trainDF.consDet.astype(int)
        self.trainDF.detHist = self.trainDF.detHist.astype(str)
        self.trainDF.noiseRatio = self.trainDF.noiseRatio.astype(float)
        self.trainDF.conRecLength = self.trainDF.conRecLength.astype(int)
        self.trainDF.hitRatio = self.trainDF.hitRatio.astype(float)

        self.trainDF['powerBin'] = (self.trainDF.Power//5)*5
        self.trainDF['noiseBin'] = (self.trainDF.noiseRatio//.1)*.1
        self.trainDF['lagDiffBin'] = (self.trainDF.lagDiff//10)*.10


        self.histDF = pd.DataFrame()                                                                                                                                                   # set number of folds in cross validation

        kList = np.arange(0,self.folds,1)

        self.trainDF['fold'] = np.random.choice(kList,len(self.trainDF))

        print ("Folds created for k fold cross validation")
        self.folds = np.arange(0,folds,1)

        self.fields = likelihood_model


    def fold(self,i):
        self.testDat = self.trainDF[self.trainDF.fold == i]
        self.trainDat = self.trainDF[self.trainDF.fold != i]                                      # create a test dataset that is the current fold
        self.testDat['HT'] = np.repeat(1,len(self.testDat))
        self.testDat['HF'] = np.repeat(0,len(self.testDat))

        # Make a Count of the predictor variables and join to training data frame - For ALIVE Strings
        self.testDat = self.testDat.reset_index()
        # Series Hit
        seriesHitCount = self.trainDat.groupby(['Detection','seriesHit'])['seriesHit'].count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitCountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = seriesHitCount,
                                how = u'left',
                                left_on = ['HT','seriesHit'],
                                right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitCountT':'seriesHitCountF'})
        self.testDat = pd.merge(left = self.testDat,
                                right = seriesHitCount,
                                how = u'left',left_on = ['HF','seriesHit'],
                                right_on = ['HF','seriesHit'])

        # Consecutive Detections
        consDetCount = self.trainDat.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetCountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = consDetCount,
                                how = u'left',
                                left_on = ['HT','consDet'],
                                right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetCountT':'consDetCountF'})
        self.testDat = pd.merge(left = self.testDat,
                                right = consDetCount,
                                how = u'left',
                                left_on = ['HF','consDet'],
                                right_on = ['HF','consDet'])

        # Consecutive Record Length
        conRecLengthCount = self.trainDat.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthCountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = conRecLengthCount,
                                how = u'left',
                                left_on = ['HT','conRecLength'],
                                right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthCountT':'conRecLengthCountF'})
        self.testDat = pd.merge(left = self.testDat,
                                right = conRecLengthCount,
                                how = u'left',
                                left_on = ['HF','conRecLength'],
                                right_on = ['HF','conRecLength'])

        # Hit Ratio
        hitRatioCount = self.trainDat.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioCountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = hitRatioCount,
                                how = u'left',
                                left_on = ['HT','hitRatio'],
                                right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioCountT':'hitRatioCountF'})
        self.testDat = pd.merge(left = self.testDat,
                                right = hitRatioCount,
                                how = u'left',
                                left_on = ['HF','hitRatio'],
                                right_on = ['HF','hitRatio'])

        # Power
        powerCount = self.trainDat.groupby(['Detection','powerBin'])['powerBin'].count()
        powerCount = pd.Series(powerCount, name = 'powerCount_T')
        powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = powerCount,
                                how = u'left',
                                left_on = ['HT','powerBin'],
                                right_on = ['HT','powerBin'])
        powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
        self.testDat = pd.merge(left = self.testDat,
                                right = powerCount,
                                how = u'left',
                                left_on = ['HF','powerBin'],
                                right_on = ['HF','powerBin'])

        # NoiseRatio
        noiseCount = self.trainDat.groupby(['Detection','noiseBin'])['noiseBin'].count()
        noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
        noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = noiseCount,
                                how = u'left',
                                left_on = ['HT','noiseBin'],
                                right_on = ['HT','noiseBin'])
        noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
        self.testDat = pd.merge(left = self.testDat,
                                right = noiseCount,
                                how = u'left',
                                left_on = ['HF','noiseBin'],
                                right_on = ['HF','noiseBin'])

        # Lag Bin
        lagCount = self.trainDat.groupby(['Detection','lagDiffBin'])['lagDiffBin'].count()
        lagCount = pd.Series(lagCount, name = 'lagDiffCount_T')
        lagCount = pd.DataFrame(lagCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat,
                                right = lagCount,
                                how = u'left',
                                left_on = ['HT','lagDiffBin'],
                                right_on = ['HT','lagDiffBin'])
        lagCount = lagCount.rename(columns = {'HT':'HF','lagDiffCount_T':'lagDiffCount_F'})
        self.testDat = pd.merge(left = self.testDat,
                                right = lagCount,
                                how = u'left',
                                left_on = ['HF','lagDiffBin'],
                                right_on = ['HF','lagDiffBin'])

    	# Nan gives us heartburn, fill them with zeros
        self.testDat = self.testDat.fillna(0)

        # Calculate Number of True and False Positive Detections in Training Dataset
        try:
            priorCountT = float(len(self.trainDat[self.trainDat.Detection == 1]))
        except KeyError:
            priorCountT = 1.0
        try:
            priorCountF = float(len(self.trainDat[self.trainDat.Detection == 0]))
        except KeyError:
            priorCountF = 1.0
        trueCount = priorCountT + 2.0
        falseCount = priorCountF + 2.0
        self.testDat['priorCount_T'] = priorCountT
        self.testDat['priorCount_F'] = priorCountF
        self.testDat['LDenomCount_T'] = trueCount
        self.testDat['LDenomCount_F'] = falseCount

        # calculation of the probability of a false positive given the data

    	# calculate the prior probability of a false detection from the training dataset
        self.testDat['priorF'] = round(priorCountF/float(len(self.trainDat)),5)
        # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        self.testDat['LHitRatioF'] =(self.testDat['hitRatioCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        self.testDat['LconRecF'] = (self.testDat['conRecLengthCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LseriesHitF'] = (self.testDat['seriesHitCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LconsDetF'] = (self.testDat['consDetCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LPowerF'] = (self.testDat['powerCount_F'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LnoiseF'] = (self.testDat['noiseCount_F'] + 1)/(self.testDat['LDenomCount_F'] + 1)
        # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LlagF'] = (self.testDat['lagDiffCount_F'] + 1)/(self.testDat['LDenomCount_F']    + 1)

        # calculation of the probability of a true detection given the data

    	# calculate the prior probability of a true detection from the training dataset
        self.testDat['priorT'] = round(priorCountT/float(len(self.trainDat)),5)
        # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        self.testDat['LHitRatioT'] = (self.testDat['hitRatioCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1)
        # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        self.testDat['LconRecT'] = (self.testDat['conRecLengthCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1)
	    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LseriesHitT'] = (self.testDat['seriesHitCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1)
	    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LconsDetT'] = (self.testDat['consDetCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1)
	    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LPowerT'] = (self.testDat['powerCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)
	    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LnoiseT'] = (self.testDat['noiseCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)
	    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LlagT'] = (self.testDat['lagDiffCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)

        # Calculate the likelihood of each hypothesis being true
        self.testDat['LikelihoodTrue'] = likelihood(True,self,status = 'cross')
        self.testDat['LikelihoodFalse'] = likelihood(False,self,status = 'cross')

        # Calculate the posterior probability of each Hypothesis occuring
        self.testDat['postTrue'] = self.testDat['priorT'] * self.testDat['LikelihoodTrue']
        self.testDat['postFalse'] = self.testDat['priorF'] * self.testDat['LikelihoodFalse']

        # calculate the evidence
        self.testDat['evidence'] = evidence(self)

        # calculate prob_a (to make this play nice with sklearn)
        self.testDat['prob_a'] = (self.testDat.priorF * self.testDat.LikelihoodFalse) / self.testDat.evidence

        self.testDat['T2F_ratio'] = self.testDat['postTrue'] / self.testDat['postFalse']
        # classify detection as true or false based on MAP hypothesis

        self.testDat['test'] = self.testDat.postTrue > self.testDat.postFalse
        self.testDat['test_inv'] = self.testDat.postFalse > self.testDat.postTrue
        self.histDF = self.histDF.append(self.testDat.reset_index())
        del self.testDat, self.trainDat

    def summary(self):
        # calculate evidence


        # create contingency table
        metrix = pd.crosstab(self.histDF.Detection,self.histDF.test)
        rowSum = metrix.sum(axis = 1)
        colSum = metrix.sum(axis = 0)

        # create correlation matrix
        if 'power' in self.fields:
            pow_index = self.fields.index('power')
            self.fields[pow_index] = 'powerBin'
        if 'noiseRatio' in self.fields:
            noi_index = self.fields.index('noiseRatio')
            self.fields[noi_index] = 'noiseBin'
        if 'lagDiff' in self.fields:
            lag_index = self.fields.index('lagDiff')
            self.fields[lag_index] = 'lagDiffBin'
        self.corr_matrix = self.histDF[self.fields].apply(pd.to_numeric).corr().round(4)
        print ("The correlation matrix for all predictors variabes:             ")
        print (self.corr_matrix)

	    # visualize the correlation matrix, closer to 1, the stronger the effect
        # if we are worried about multicollinearity, I would stear away from variable combinations where coefficient ~ 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.corr_matrix, vmin = -1, vmax = 1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(self.fields),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.fields)
        ax.set_yticklabels(self.fields)
        plt.show()
        try:
            print ("----------------------------------------------------------------")
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
            print ("             False          Valid")
            print ("      ______________________________")
            print (" Known|              |              |")
            print (" False| TN:%s  | FP:%s  |"%(format(metrix.iloc[0,0]," 8d"),format(metrix.iloc[0,1]," 8d")))
            print ("      |______________|______________|")
            print (" Known|              |              |")
            print ("  True| FN:%s  | TP:%s  |"%(format(metrix.iloc[1,0]," 8d"),format(metrix.iloc[1,1]," 8d")))
            print ("      |______________|______________|")
            print ("")
            print ("________________________________________________________________")
            print ("Positive Predictive Value: %s"%(round(float(metrix.iloc[1,1])/float(colSum[1]),4)))
            print ("PPV = TP/(TP + FP)")
            print ("Probability of a record being correctly classified as true")
            print ("----------------------------------------------------------------")
            print ("Negative Predictive Value: %s"%(round(float(metrix.iloc[0,0])/float(colSum[0]),4)))
            print ("NPV = TN/(TN + FN)")
            print ("Probability of a record being correctly classified as false")
            print ("----------------------------------------------------------------")
            print ("The sensitivity of the classifier is: %s"%(round(float(metrix.iloc[1,1])/float(rowSum[1]),4)))
            print ("sensitivity = TP /(TP + FN)")
            print ("Probability of a record being classified true,")
            print ("given that the record is in fact true")
            print ("----------------------------------------------------------------")
            print ("The specificity of the classifier is: %s"%(round(float(metrix.iloc[0,0])/float(rowSum[0]),4)))
            print ("specificity = TN / (TN + FP)")
            print ("Probability of a record being classified false,")
            print ("given that the record is in fact false")
            print ("----------------------------------------------------------------")
            fpr = round(float(metrix.iloc[0,1])/(float(metrix.iloc[0,1]) + float(metrix.iloc[0,0])),4)
            print ("The false positive rate: %s"%(fpr))
            print ("fpr = FP / (FP + TN)")
            print ("The proportion of all records classified as false,")
            print ("that are still classified as valid")
        except IndexError:
            print (metrix)
        print ("----------------------------------------------------------------")
        print ("Calculate Receiver Operating Curve and Precision Recall Curve")

        # calculate ROC curve and ROC-AUC score using sklearn

        fpr, tpr, thresholds = metrics.roc_curve(self.histDF.test_inv,self.histDF.prob_a,pos_label = 1)
        prc, rc, thresholds = metrics.precision_recall_curve(self.histDF.test_inv,self.histDF.prob_a,pos_label = 1)
        '''https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py'''

        roc_auc = round(metrics.auc(fpr, tpr),4)
        print ("The ROC-AUC for this classifier is: %s"%(roc_auc))

        ap = round(metrics.average_precision_score(self.histDF.test, self.histDF.prob_a),4)
        print ("The PRC-AUC for this classifier is: %s"%(ap))

        # make a figure
        figSize = (4,2)
        plt.figure()
        fig, axs = plt.subplots(1,2,tight_layout = True,figsize = figSize)
        # ROC on the left
        axs[0].plot(fpr, tpr, color='darkorange',lw=2)
        axs[0].plot([0,1], [0,1], color='blue',lw=2,ls = '--')
        axs[0].set_xlim([0.0, 1.05])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Curve')
        # PRC on the right
        axs[1].plot(rc, prc, color='darkorange',lw=2)
        axs[1].plot([0,1], [0,0], color='blue',lw=2,ls = '--')
        axs[1].set_xlim([0.0, 1.05])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].set_title('Precision-Recall Curve')
        print ("________________________________________________________________")

class classification_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS = None,rec_list = None, reclass_iter = None):
        # initialize some variables
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS

    	# connect to the project database and get some data
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)
        self.final_iter = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','noiseRatio','hitRatio_A','hitRatio_M','postTrue_A','postTrue_M','postFalse_A','postFalse_M','test','lagDiff','consDet_A', 'consDet_M','conRecLength_A', 'conRecLength_M','logLikelihoodRatio_A', 'logLikelihoodRatio_M'])                # set up an empty data frame
        self.init_iter = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','noiseRatio','hitRatio_A','hitRatio_M','postTrue_A','postTrue_M','postFalse_A','postFalse_M','test','lagDiff','consDet_A', 'consDet_M','conRecLength_A', 'conRecLength_M','logLikelihoodRatio_A', 'logLikelihoodRatio_M'])
        self.reclass_iter = reclass_iter
        self.rec_list = rec_list

	    # if there is no receiver list we want it all
        if rec_list == None:
            recSQL = "SELECT * FROM tblMasterReceiver WHERE RecType = '%s'"%(self.recType) # SQL code to import data from this node
            receivers = pd.read_sql(recSQL,con = conn)                         # import data
            receivers = receivers.recID.unique()                               # get the unique receivers associated with this node
            for i in receivers:                                                # for every receiver
                print ("Start selecting and merging data for receiver %s"%(i))

                sql = '''SELECT FreqCode,
                                Epoch,
                                Power,
                                noiseRatio,
                                hitRatio_M,
                                postTrue_A,
                                postTrue_M,
                                postFalse_A,
                                postFalse_M,
                                test,
                                lagDiff,
                                consDet_A,
                                consDet_M,
                                conRecLength_A,
                                conRecLength_M,
                                logLikelihoodRatio_A,
                                logLikelihoodRatio_M
                        FROM tblClassify_%s_%s '''%(i, reclass_iter)

		        # get data for this receiver
                dat = pd.read_sql(sql, con = conn, coerce_float = True)
                self.final_iter = self.final_iter.append(dat)
                del dat

                sql = '''SELECT FreqCode,
                                Epoch,
                                recID,
                                Power,
                                noiseRatio,
                                hitRatio_A,
                                hitRatio_M,
                                postTrue_A,
                                postTrue_M,
                                postFalse_A,
                                postFalse_M,
                                test,
                                lagDiff,
                                consDet_A,
                                consDet_M,
                                conRecLength_A,
                                conRecLength_M,
                                logLikelihoodRatio_A,
                                logLikelihoodRatio_M
                        FROM tblClassify_%s_%s '''%(i, 1)

                # get data for this receiver
                dat = pd.read_sql(sql, con = conn, coerce_float = True)
                self.init_iter = self.init_iter.append(dat)
                del dat

        else:
            # get all the table names
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()

            # generate maximum iteration dictionary
            max_iter_dict = max_iter(rec_list,tbls)[0]

            # once we have a hash table of receiver to max classification, extract the classification dataset
            self.classDF = pd.DataFrame()
            for i in rec_list:
                if reclass_iter == None:
                    print ("Start selecting and merging data for receiver %s"%(i))

                    sql = '''SELECT FreqCode,
                                    Epoch,
                                    recID,
                                    Power,
                                    noiseRatio,
                                    hitRatio_A,
                                    hitRatio_M,
                                    postTrue_A,
                                    postTrue_M,
                                    postFalse_A,
                                    postFalse_M,
                                    test,
                                    lagDiff,
                                    consDet_A,
                                    consDet_M,
                                    conRecLength_A,
                                    conRecLength_M,
                                    logLikelihoodRatio_A,
                                    logLikelihoodRatio_M
                            FROM %s'''%(max_iter_dict[i])

        		    # get data for this receiver
                    dat = pd.read_sql(sql, con = conn, coerce_float = True)
                    self.final_iter = self.final_iter.append(dat)
                    del dat

                    sql = '''SELECT FreqCode,
                                    Epoch,
                                    recID,
                                    Power,
                                    noiseRatio,
                                    hitRatio_A,
                                    hitRatio_M,
                                    postTrue_A,
                                    postTrue_M,
                                    postFalse_A,
                                    postFalse_M,
                                    test,
                                    lagDiff,
                                    consDet_A,
                                    consDet_M,
                                    conRecLength_A,
                                    conRecLength_M,
                                    logLikelihoodRatio_A,
                                    logLikelihoodRatio_M
                            FROM tblClassify_%s_1 '''%(i)

        		    # get data for this receiver
                    dat = pd.read_sql(sql, con = conn, coerce_float = True)
                    self.init_iter = self.init_iter.append(dat)
                    del dat
                else:

                    print ("Start selecting and merging data for receiver %s"%(i))

                    sql = '''SELECT FreqCode,
                                    Epoch,
                                    recID,
                                    Power,
                                    noiseRatio,
                                    hitRatio_A,
                                    hitRatio_M,
                                    postTrue_A,
                                    postTrue_M,
                                    postFalse_A,
                                    postFalse_M,
                                    test,
                                    lagDiff,
                                    consDet_A,
                                    consDet_M,
                                    conRecLength_A,
                                    conRecLength_M,
                                    logLikelihoodRatio_A,
                                    logLikelihoodRatio_M
                            FROM tblClassify_%s_%s '''%(i,reclass_iter)

                    # get data for this receiver
                    dat = pd.read_sql(sql, con = conn, coerce_float = True)
                    self.final_iter = self.final_iter.append(dat)
                    del dat

                    sql = '''SELECT FreqCode,
                                    Epoch,
                                    recID,
                                    Power,
                                    noiseRatio,
                                    hitRatio_A,
                                    hitRatio_M,
                                    postTrue_A,
                                    postTrue_M,
                                    postFalse_A,
                                    postFalse_M,
                                    test,
                                    lagDiff,
                                    consDet_A,
                                    consDet_M,
                                    conRecLength_A,
                                    conRecLength_M,
                                    logLikelihoodRatio_A,
                                    logLikelihoodRatio_M
                            FROM tblClassify_%s_1 '''%(i)

                    # get data for this receiver
                    dat = pd.read_sql(sql, con = conn, coerce_float = True)
                    self.init_iter = self.init_iter.append(dat)
                    del dat
            trues = self.final_iter[self.final_iter.test == 1]
            falses = pd.DataFrame()

            rec_iter_list = max_iter(rec_list,tbls)[1]

            for i in rec_iter_list:
                sql = '''SELECT FreqCode,
                                Epoch,
                                recID,
                                Power,
                                noiseRatio,
                                hitRatio_A,
                                hitRatio_M,
                                postTrue_A,
                                postTrue_M,
                                postFalse_A,
                                postFalse_M,
                                test,
                                lagDiff,
                                consDet_A,
                                consDet_M,
                                conRecLength_A,
                                conRecLength_M,
                                logLikelihoodRatio_A,
                                logLikelihoodRatio_M
                        FROM %s'''%(i)
                # get data for this receiver
                dat = pd.read_sql(sql, con = conn, coerce_float = True)
                dat = dat[dat.test == 0]
                falses = falses.append(dat)

                del dat

            self.final_iter = trues.append(falses)
            del trues, falses
        c.close()


        trues = self.final_iter[self.final_iter.test == 1][['FreqCode','Epoch','test']]
        trues.rename(columns = {'test':'final_test'},inplace = True)
        print (len(trues))
        # join true detections to initial data
        self.init_iter = self.init_iter.merge(trues,how = 'left',left_on = ['FreqCode','Epoch'], right_on = ['FreqCode','Epoch'])
        self.init_iter = cons_det_filter(self.init_iter)
        self.init_iter.final_test.fillna(0,inplace = True)
        self.init_iter.drop_duplicates(keep = 'first', inplace = True)

    def classify_stats(self):
        '''function reads all	 classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        print ("")
        if self.rec_list != None:
            print ("Classification summary statistics report for sites %s"%(self.rec_list))
        else:
            print ("Classification summary statistics report")
        print ("----------------------------------------------------------------------------------")
        det_class_count = self.init_iter.groupby('final_test')['final_test'].count().to_frame()
        if len(det_class_count)>1:
            print ("")
            print ("%s detection class statistics:"%(self.recType))
            print ("The probability that a detection was classified as true was %s"%((round(float(det_class_count.at[1,'final_test'])/float(det_class_count.sum()),3))))
            print ("The probability that a detection was classified as false positive was %s"%((round(float(det_class_count.at[0,'final_test'])/float(det_class_count.sum()),3))))
            print ("")
            print ("----------------------------------------------------------------------------------")
            print ("")
            self.sta_class_count = self.final_iter.groupby(['recID','test'])['test'].count().to_frame()#.reset_index(drop = False)
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
            print ("----------------------------------------------------------------------------------")
            print ("Assess concordance with consecutive detection requirement (Beeman and Perry)")

            # calculate Cohen's Kappa (concordance)
            n11 = len(self.init_iter[(self.init_iter.final_test == 1) & (self.init_iter.consDet_A == 1)])
            print ("The algorithm and Beeman and Perry classified the same %s recoreds as true "%(n11))
            n10 = len(self.init_iter[(self.init_iter.final_test == 1) & (self.init_iter.consDet_A == 0)])
            print ("The algorithm classified %s records as true while Beeman and Perry classified them as false"%(n10))
            n01 = len(self.init_iter[(self.init_iter.final_test == 0) & (self.init_iter.consDet_A == 1)])
            print ("The algorithm classified %s records as false while Beeman and Perry classified them as true"%(n01))
            n00 = len(self.init_iter[(self.init_iter.final_test == 0) & (self.init_iter.consDet_A == 0)])
            print ("The algorithm and Beeman and Perry classified the same %s records as false positive"%(n00))
            I_o = (n11 + n00)/(n11 + n10 + n01 + n00)
            print ("The observed propotion of agreement was %s"%(I_o))
            I_e = (((n11 + n01)*(n11 + n10)/(n11 + n10 + n01 + n00)) + ((n10 + n00)*(n01 + n00)/(n11 + n10 + n01 + n00)))/(n11 + n10 + n01 + n00)
            print ("The expected agreement due to chance alone was %s"%(I_e))

            self.kappa = (I_o - I_e)/(1.- I_e)

            print ("Cohen's Kappa: %s"%(self.kappa))
            print ("----------------------------------------------------------------------------------")
            print ("Compiling Figures")

            # get data by detection class for side by side histograms
            self.final_iter['Power'] = self.final_iter.Power.astype(float)
            self.final_iter['lagDiff'] = self.final_iter.lagDiff.astype(float)
            self.final_iter['conRecLength_A'] = self.final_iter.conRecLength_A.astype(float)
            self.final_iter['noiseRatio'] = self.final_iter.noiseRatio.astype(float)
            self.final_iter['logPostRatio_A'] =np.log10(self.final_iter.postTrue_A.values/self.final_iter.postFalse_A.values)
            self.final_iter['logPostRatio_M'] =np.log10(self.final_iter.postTrue_M.values/self.final_iter.postFalse_M.values)

            '''Currently these figures only return data for hypothesized alive fish - but we can have dead fish in here and
            their inlcusion in the histograms may be biasing the results - find the max hit ratio, consecutive record length, and likelihood ratio
            '''
            self.final_iter['hitRatio_max'] = self.final_iter[['hitRatio_A','hitRatio_M']].max(axis = 1)
            self.final_iter['conRecLength_max'] = self.final_iter[['conRecLength_A','conRecLength_M']].max(axis = 1)
            self.final_iter['logLikelihoodRatio_max'] = self.final_iter[['logLikelihoodRatio_A','logLikelihoodRatio_M']].max(axis = 1)
            self.final_iter['logPostRatio_max'] = self.final_iter[['logPostRatio_A','logPostRatio_M']].max(axis = 1)

            trues = self.final_iter[self.final_iter.test == 1]
            falses = self.final_iter[self.final_iter.test == 0]
            self.trues = trues
            self.falses = falses

            # plot hit ratio histograms by detection class
            hitRatioBins =np.linspace(0,1.0,11)

            # plot signal power histograms by detection class
            minPower = self.final_iter.Power.min()//5 * 5
            maxPower = self.final_iter.Power.max()//5 * 5
            powerBins =np.arange(minPower,maxPower+20,10)

            # Lag Back Differences - how stdy are detection lags?
            lagBins =np.arange(-100,110,20)

            # Consecutive Record Length ?
            conBins =np.arange(1,12,1)

            # Noise Ratio
            noiseBins =np.arange(0,1.1,0.1)

            # plot the log likelihood ratio
            minLogRatio = self.final_iter.logLikelihoodRatio_max.min()//1 * 1
            maxLogRatio = self.final_iter.logLikelihoodRatio_max.max()//1 * 1
            ratioBins =np.arange(minLogRatio,maxLogRatio+1,2)


            # plot the log of the posterior ratio
            minPostRatio = self.final_iter.logPostRatio_max.min()
            maxPostRatio = self.final_iter.logPostRatio_max.max()
            postRatioBins = np.linspace(minPostRatio,maxPostRatio,10)

            # make lattice plot for pubs
            figSize = (8,6)
            plt.figure()
            fig, axs = plt.subplots(3,4,tight_layout = True,figsize = figSize)
            # hit ratio
            axs[0,1].hist(trues.hitRatio_max.values,
                          hitRatioBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[0,0].hist(falses.hitRatio_max.values,
                          hitRatioBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[0,0].set_xlabel('Hit Ratio')
            axs[0,1].set_title('Valid')
            axs[0,1].set_xlabel('Hit Ratio')
            axs[0,0].set_title('False Positive')
            axs[0,0].set_title('A',loc = 'left')

            # consecutive record length
            axs[0,3].hist(trues.conRecLength_max.values,
                          conBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[0,2].hist(falses.conRecLength_max.values,
                          conBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[0,2].set_xlabel('Consecutive Hit Length')
            axs[0,3].set_title('Valid')
            axs[0,3].set_xlabel('Consecutive Hit Length')
            axs[0,2].set_title('False Positive')
            axs[0,2].set_title('B',loc = 'left')

            # power
            axs[1,1].hist(trues.Power.values,
                          powerBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[1,0].hist(falses.Power.values,
                          powerBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[1,0].set_xlabel('Signal Power')
            axs[1,1].set_xlabel('Signal Power')
            axs[1,0].set_ylabel('Probability Density')
            axs[1,0].set_title('C',loc = 'left')

            # noise ratio
            axs[1,3].hist(trues.noiseRatio.values,
                          noiseBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[1,2].hist(falses.noiseRatio.values,
                          noiseBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[1,2].set_xlabel('Noise Ratio')
            axs[1,3].set_xlabel('Noise Ratio')
            axs[1,2].set_title('D',loc = 'left')

            # lag diff
            axs[2,1].hist(trues.lagDiff.values,
                          lagBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[2,0].hist(falses.lagDiff.values,
                          lagBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[2,0].set_xlabel('Lag Differences')
            axs[2,1].set_xlabel('Lag Differences')
            axs[2,0].set_title('E',loc = 'left')

            # log posterior ratio
            axs[2,3].hist(trues.logLikelihoodRatio_max.values,
                          postRatioBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[2,2].hist(falses.logLikelihoodRatio_max.values,
                          postRatioBins,
                          density = True,
                          color = 'grey',
                          edgecolor='black',
                          linewidth=1.2)
            axs[2,2].set_xlabel('Log Likelihood Ratio')
            axs[2,3].set_xlabel('Log Likelihood Ratio')
            axs[2,2].set_title('F',loc = 'left')

            if self.figureWS != None:
                if self.rec_list != None:
                   plt.savefig(os.path.join(self.figureWS,"%s_lattice_class.png"%(self.recType)),
                               bbox_inches = 'tight',
                               dpi = 900)
                else:
                   plt.savefig(os.path.join(self.figureWS,"%s_%s_lattice_class.png"%(self.recType,self.site)),
                               bbox_inches = 'tight',
                               dpi = 900)
        else:
           print("There were insufficient data to quantify summary statistics or histogram plots, either because there were no false positives or because there were no valid detections")

class training_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS,site = None):
        # initialize some variables
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        self.site = site

        # connect to database and get data
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)                                 # connect to the database

        if self.site == None:
            sql = "SELECT * FROM tblTrain WHERE recType = '%s'"%(self.recType)
        else:
            sql = "SELECT * FROM tblTrain WHERE recType = '%s' AND recID == '%s'"%(self.recType,self.site)
        trainDF = pd.read_sql(sql,con=conn,coerce_float  = True)#This will read in tblTrain and create a pandas dataframe

        # recs = pd.read_sql("SELECT recID from tblMasterReceiver", con = conn).recID.values

        # c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        # tbls = c.fetchall()
        # # iterate over the receivers to find the final classification (aka the largest _n)
        # max_iter_dict = {} # receiver:max iter
        # curr_idx = 0
        # for i in recs:
        #     max_iter = 1
        #     while curr_idx < len(tbls) - 1:
        #         for j in tbls:
        #             if i in j[0]:
        #                 if int(j[0][-1]) >= max_iter:
        #                     max_iter = int(j[0][-1])
        #                     max_iter_dict[i] = j[0]
        #             curr_idx = curr_idx + 1
        #     curr_idx = 0
        # print (max_iter_dict)
        # del i, j, curr_idx

        # if len(max_iter_dict) > 0:
        #     # once we have a hash table of receiver to max classification, extract the classification dataset
        #     classDF = pd.DataFrame()
        #     for i in max_iter_dict:
        #         classDat = pd.read_sql("select test, FreqCode,Power,noiseRatio, lag,lagDiff,conRecLength_A,consDet_A,detHist_A,hitRatio_A,seriesHit_A,conRecLength_M,consDet_M,detHist_M,hitRatio_M,seriesHit_M,postTrue_A,postTrue_M,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from %s"%(max_iter_dict[i]),con=conn)
        #         #classDat = classDat[classDat.postTrue_A >= classDat.postTrue_M]
        #         classDat.drop(['conRecLength_M','consDet_M','detHist_M','hitRatio_M','seriesHit_M'], axis = 1, inplace = True)
        #         classDat.rename(columns = {'conRecLength_A':'conRecLength','consDet_A':'consDet','detHist_A':'detHist','hitRatio_A':'hitRatio','seriesHit_A':'seriesHit'}, inplace = True)
        #         classDF = classDF.append(classDat)

        #     trainDF = trainDF[trainDF.Detection==0]
        #     classDF = classDF[classDF.test==1]
        #     classDF['Channels']=np.repeat(1,len(classDF))
        #     classDF.rename(columns={"test":"Detection","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
        #     trainDF = trainDF.append(classDF)

        self.train_stats_data = trainDF
        c.close()

    def train_stats(self):
        '''function reads all classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        det_class_count = self.train_stats_data.groupby('Detection')['Detection'].count().to_frame()

        print ("")
        print ("Training summary statistics report")
        print ("The algorithm collected %s detections from %s %s receivers"%(len(self.train_stats_data),len(self.train_stats_data.recID.unique()),self.recType))
        print ("----------------------------------------------------------------------------------")
        print ("")
        print ("%s detection clas statistics:"%(self.recType) )
        try:
            print ("The prior probability that a detection was true was %s"%((round(float(det_class_count.at[1,'Detection'])/float(det_class_count.sum()),3))))
        except KeyError:
            print ("No known true detections found at %s"%(self.site))
            pass
        try:
            print ("The prior probability that a detection was false positive was %s"%((round(float(det_class_count.at[0,'Detection'])/float(det_class_count.sum()),3))))
        except KeyError:
            print ("No known true detections found at %s"%(self.site))
            pass

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
        self.train_stats_data['lagDiff'] = self.train_stats_data.lagDiff.astype(float)
        self.train_stats_data['conRecLength'] = self.train_stats_data.conRecLength.astype(float)
        self.train_stats_data['noiseRatio'] = self.train_stats_data.noiseRatio.astype(float)
        #self.train_stats_data['FishCount'] = self.train_stats_data.FishCount.astype(float)

        trues = self.train_stats_data[self.train_stats_data.Detection == '1']
        falses = self.train_stats_data[self.train_stats_data.Detection == '0']
        # plot hit ratio histograms by detection class
        hitRatioBins =np.linspace(0,1.0,11)

        # plot signal power histograms by detection class
        minPower = self.train_stats_data.Power.min()//5 * 5
        maxPower = self.train_stats_data.Power.max()//5 * 5
        powerBins =np.arange(minPower,maxPower+20,10)

        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,6)

        # Consecutive Record Length ?
        conBins =np.arange(1,12,1)

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

        # make lattice plot for pubs
        figSize = (3,7)
        plt.figure()
        fig, axs = plt.subplots(5,2,tight_layout = True,figsize = figSize)
        # hit ratio
        axs[0,1].hist(trues.hitRatio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].hist(falses.hitRatio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].set_xlabel('Hit Ratio')
        axs[0,1].set_title('True')
        axs[0,1].set_xlabel('Hit Ratio')
        axs[0,0].set_title('False Positive')
        axs[0,0].set_title('A',loc = 'left')

        # consecutive record length
        axs[1,1].hist(trues.conRecLength.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].hist(falses.conRecLength.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].set_xlabel('Consecutive Hit Length')
        axs[1,1].set_xlabel('Consecutive Hit Length')
        axs[1,0].set_title('B',loc = 'left')

        # power
        axs[2,1].hist(trues.Power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].hist(falses.Power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].set_xlabel('Signal Power')
        axs[2,1].set_xlabel('Signal Power')
        axs[2,0].set_ylabel('Probability Density')
        axs[2,0].set_title('C',loc = 'left')

        # noise ratio
        axs[3,1].hist(trues.noiseRatio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].hist(falses.noiseRatio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].set_xlabel('Noise Ratio')
        axs[3,1].set_xlabel('Noise Ratio')
        axs[3,0].set_title('D',loc = 'left')

        # lag diff
        axs[4,1].hist(trues.lagDiff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].hist(falses.lagDiff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].set_xlabel('Lag Differences')
        axs[4,1].set_xlabel('Lag Differences')
        axs[4,0].set_title('E',loc = 'left')


        if self.site != None:
           plt.savefig(os.path.join(self.figureWS,"%s_%s_lattice_train.png"%(self.recType,self.site)),bbox_inches = 'tight', dpi = 900)
        else:
           plt.savefig(os.path.join(self.figureWS,"%s_lattice_train.png"%(self.recType)),bbox_inches = 'tight', dpi = 900)

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

def cons_det_filter(classify_data):
    '''Function that applies a filter based on consecutive detections.  Replicates
    Beeman and Perry 2012'''
    # determine if consecutive detections consDet is true
    classify_data.loc[(classify_data['consDet_A'] == 1) | (classify_data['consDet_M'] == 1), 'cons_det_filter'] = 1
    classify_data.loc[(classify_data['consDet_A'] != 1) & (classify_data['consDet_M'] != 1), 'cons_det_filter'] = 0

    return classify_data
