# -*- coding: utf-8 -*-
 # Module contains all of the objects and required for analysis of telemetry data

# import modules required for function dependencies
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
                data.loc[(data['epoch_shift_%s'%(np.int(j))] >= data['ll_%s'%(np.int(i))]) & (data['epoch_shift_%s'%(np.int(j))] <= data['ul_%s'%(np.int(i))]),'det_%s'%(np.int(i))] = '1'

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
    fishies = trunc.FreqCode.unique()
    return np.sum(np.isin(fishies,tags))


def noiseRatio (duration,data,study_tags):

    ''' function calculates the ratio of miscoded, pure noise detections, to matching frequency/code
    detections within the duration specified.

    In other words, what is the ratio of miscoded to correctly coded detections within the duration specified

    duration = moving window length in minutes
    data = current data file
    study_tags = list or list like object of study tags
    '''
    # identify miscodes
    data['miscode'] = np.isin(data.FreqCode.values, study_tags, invert = True)

    # bin everything into nearest 5 min time bin and count miscodes and total number of detections
    duration_s = str(int(duration * 60)) + 's'
    miscode = data.groupby(pd.Grouper(key = 'timeStamp', freq = duration_s)).miscode.sum().to_frame()
    total = data.groupby(pd.Grouper(key = 'timeStamp', freq = duration_s)).FreqCode.count().to_frame()

    # rename
    total.rename(columns = {'FreqCode':'total'}, inplace = True)

    # merge dataframes, calculate noise ratio
    noise = total.merge(miscode, left_on = 'timeStamp', right_on ='timeStamp')
    noise.reset_index(inplace = True)
    noise.fillna(value = 0, inplace = True)
    noise['noiseRatio'] = noise.miscode / noise.total
    noise.dropna(inplace = True)
    noise['Epoch'] = (noise['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
    # create function for noise ratio at time
    if len(noise) >= 2:
        noise_ratio_fun = interpolate.interp1d(noise.Epoch.values,noise.noiseRatio.values,kind = 'linear',bounds_error = False, fill_value ='extrapolate')
        # interpolate noise ratio as a function of time for every row in data
        data['noiseRatio'] = noise_ratio_fun(data.Epoch.values)
    data.drop(columns = ['miscode'], inplace = True)
    return data

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
#    program_dir = os.path.join(project_dir, 'Program')                         # this is where we will create a local clone of the Git repository
#    if not os.path.exists(program_dir):
#        os.makedirs(program_dir)

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

    c.execute('''CREATE TABLE tblTrain(Channels INTEGER, Detection INTEGER, FreqCode TEXT, Power REAL, lag INTEGER, lagDiff REAL, FishCount INTEGER, conRecLength INTEGER, miss_to_hit REAL, consDet INTEGER, detHist TEXT, hitRatio REAL, noiseRatio REAL, seriesHit INTEGER, timeStamp TIMESTAMP, Epoch INTEGER, Seconds INTEGER, fileName TEXT, recID TEXT, recType TEXT, ScanTime REAL)''') # create full radio table - table includes all records, final version will be designed for specific receiver types
    c.execute('''CREATE TABLE tblRaw(timeStamp TIMESTAMP, Epoch INTEGER, FreqCode TEXT, Power REAL,noiseRatio, fileName TEXT, recID TEXT, ScanTime REAL, Channels REAL, RecType TEXT)''')
    #c.execute('''CREATE INDEX idx_fileNameRaw ON tblRaw (fileName)''')
    c.execute('''CREATE INDEX idx_RecID_Raw ON tblRaw (recID)''')
    c.execute('''CREATE INDEX idx_FreqCode On tblRaw (FreqCode)''')
    #c.execute('''CREATE INDEX idx_fileNameTrain ON tblTrain (fileName)''')
    c.execute('''CREATE INDEX idx_RecType ON tblTrain (recType)''')

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

def orionImport(fileName,rxfile,dbName,recName,switch = False, scanTime = None, channels = None, ant_to_rec_dict = None):
    '''Function imports raw Sigma Eight orion data.

    Text parser uses simple column fixed column widths.

    '''
    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()
    study_tags = pd.read_sql('SELECT FreqCode, TagType FROM tblMasterTag',con = conn)
    study_tags = study_tags[study_tags.TagType == 'Study'].FreqCode.values

    recType = 'orion'
    if ant_to_rec_dict != None:
        scanTime = 1
        channels = 1
    # what orion firmware is it?  the header row is the key
    o_file =open(fileName, encoding='utf-8')
    header = o_file.readline()[:-1]                                            # read first line in file
    columns = str.split(header)
    o_file.close()
    if 'Type' in columns:
        # with our data row, extract information using pandas fwf import procedure
        telemDat = pd.read_fwf(fileName,colspecs = [(0,12),(13,23),(24,30),(31,35),(36,45),(46,54),(55,60),(61,65)],
                                names = ['Date','Time','Site','Ant','Freq','Type','Code','Power'],
                                skiprows = 1,
                                dtype = {'Date':str,'Time':str,'Site':np.int32,'Ant':str,'Freq':str,'Type':str,'Code':str,'Power':np.float64})
        telemDat = telemDat[telemDat.Type != 'STATUS']
        telemDat.drop(['Type'], axis = 1, inplace = True)

    else:
        # with our data row, extract information using pandas fwf import procedure
        telemDat = pd.read_fwf(fileName,colspecs = [(0,11),(11,20),(20,26),(26,30),(30,37),(37,42),(42,48)],
                                names = ['Date','Time','Site','Ant','Freq','Code','Power'],
                                skiprows = 1,
                                dtype = {'Date':str,'Time':str,'Site':str,'Ant':str,'Freq':str,'Code':str,'Power':str})

    if len(telemDat) > 0:
        telemDat['fileName'] = np.repeat(rxfile,len(telemDat))    #Note I'm going back here to the actual file name without the path.  Is that OK?  I prefer it, but it's a potential source of confusion
        telemDat['FreqCode'] = telemDat['Freq'].astype(str) + ' ' + telemDat['Code'].astype(str)
        telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'],errors = 'coerce')# create timestamp field from date and time and apply to index
        telemDat['ScanTime'] = np.repeat(scanTime,len(telemDat))
        telemDat['Channels'] = np.repeat(channels,len(telemDat))
        telemDat['RecType'] = np.repeat('orion',len(telemDat))
        telemDat = telemDat[telemDat.timeStamp.notnull()]
        if len(telemDat) == 0:
            print ("Invalid timestamps in raw data, cannot import")
        else:
            telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            telemDat.drop (['Date','Time','Freq','Code','Site'],axis = 1, inplace = True)
            telemDat = noiseRatio(5.0,telemDat,study_tags)
            if ant_to_rec_dict == None:
                telemDat.drop(['Ant'], axis = 1, inplace = True)
                telemDat['recID'] = np.repeat(recName,len(telemDat))
                tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
                index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                telemDat.set_index(index,inplace = True,drop = False)
                telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
#                recParamLine = [(recName,recType,scanTime,channels,fileName)]
#                conn.executemany('INSERT INTO tblReceiverParameters VALUES (?,?,?,?,?)',recParamLine)
                conn.commit()
                c.close()
            else:
                for i in ant_to_rec_dict:
                    site = ant_to_rec_dict[i]
                    telemDat_sub = telemDat[telemDat.Ant == str(i)]
                    telemDat_sub['recID'] = np.repeat(site,len(telemDat_sub))
                    tuples = zip(telemDat_sub.FreqCode.values,telemDat_sub.recID.values,telemDat_sub.Epoch.values)
                    index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                    telemDat_sub.set_index(index,inplace = True,drop = False)
                    telemDat_sub.drop(['Ant'], axis = 1, inplace = True)
                    telemDat_sub.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
#                    recParamLine = [(site,recType,scanTime,channels,fileName)]
#                    conn.executemany('INSERT INTO tblReceiverParameters VALUES (?,?,?,?,?)',recParamLine)
                    conn.commit()
                    c.close()


def lotek_import(fileName,rxfile,dbName,recName,ant_to_rec_dict = None):
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
    o_file = open(fileName, encoding='utf-8')
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
                while next(rows)[1][0] != '\n':                                       # while the next row isn't empty
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
        study_tags = pd.read_sql('SELECT FreqCode, TagType FROM tblMasterTag',con = conn)
        study_tags = study_tags[study_tags.TagType == 'Study'].FreqCode.values


        # with our data row, extract information using pandas fwf import procedure

        #Depending on firmware the data structure will change.  This is for xxx firmware.  See below for additional firmware configs

#       telemDat = pd.read_fwf(fileName,colspecs = [(0,8),(8,18),(18,28),(28,36),(36,51),(51,59)],names = ['Date','Time','ChannelID','TagID','Antenna','Power'],skiprows = dataRow)
#       telemDat = telemDat.iloc[:-2]                                                   # remove last two rows, Lotek adds garbage at the end

        #Master Firmware: Version 9.12.5
        telemDat = pd.read_fwf(fileName,colspecs = [(0,8),(8,23),(23,33),(33,41),(41,56),(56,64)],names = ['Date','Time','ChannelID','TagID','Antenna','Power'],skiprows = dataRow)
        telemDat = telemDat.iloc[:-2]                                                   # remove last two 

        telemDat['Antenna'] = telemDat['Antenna'].astype(str)                   #TCS Added this to get dict to line up with data

        telemDat['fileName'] = np.repeat(rxfile,len(telemDat))                   # Adding the filename into the dataset...drop the path (note this may cause confusion because above we use filename with path.  Decide what to do and fix)
        def id_to_freq(row,channelDict):
            if row[2] in channelDict:
                return channelDict[row[2]]
            else:
                return '888'
        if len(telemDat) > 0:
            if ant_to_rec_dict == None:
                telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
                telemDat = telemDat[telemDat.Frequency != '888']
                telemDat = telemDat[telemDat.TagID != 999]
                telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
                telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'])# create timestamp field from date and time and apply to index
                telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                telemDat = noiseRatio(5.0,telemDat,study_tags)
                telemDat.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                telemDat['ScanTime'] = np.repeat(scanTime,len(telemDat))
                telemDat['Channels'] = np.repeat(channels,len(telemDat))
                telemDat['RecType'] = np.repeat(recType,len(telemDat))
                telemDat['recID'] = np.repeat(recName,len(telemDat))
                telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
            else:
                for ant in ant_to_rec_dict:
                    site = ant_to_rec_dict[ant]
                    telemDat_sub = telemDat[telemDat.Antenna == str(ant)]
                    telemDat_sub['Frequency'] = telemDat_sub.apply(id_to_freq, axis = 1, args = (channelDict,))
                    telemDat_sub = telemDat_sub[telemDat_sub.Frequency != '888']
                    telemDat_sub = telemDat_sub[telemDat_sub.TagID != 999]
                    telemDat_sub['FreqCode'] = telemDat_sub['Frequency'].astype(str) + ' ' + telemDat_sub['TagID'].astype(int).astype(str)
                    telemDat_sub['timeStamp'] = pd.to_datetime(telemDat_sub['Date'] + ' ' + telemDat_sub['Time'])# create timestamp field from date and time and apply to index
                    telemDat_sub['Epoch'] = (telemDat_sub['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                    telemDat_sub = noiseRatio(5.0,telemDat_sub,study_tags)
                    telemDat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                    telemDat_sub['ScanTime'] = np.repeat(scanTime,len(telemDat_sub))
                    telemDat_sub['Channels'] = np.repeat(channels,len(telemDat_sub))
                    telemDat_sub['RecType'] = np.repeat(recType,len(telemDat_sub))
                    telemDat_sub['recID'] = np.repeat(site,len(telemDat_sub))
                    telemDat_sub.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')

    else:
        lotek400 = False
        # find where data begins and header data ends
        with o_file as f:
            for line in f:
                if "********************************* Data Segment *********************************" in line:
                    counter = counter + 1
                    dataRow = counter + 5                                               # if this current line signifies the start of the data stream, the data starts three rows down from this
                    break                                                               # break the loop, we have reached our stop point
                elif line[0:14] == "Code_log data:":
                    counter = counter + 1
                    dataRow = counter + 3
                    lotek400 = True
                    break
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
            if 'scan time' in row[1][0] or 'Scan time' in row[1][0]:                   # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document
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
                while next(rows)[1][0] != '\n':                                       # while the next row isn't empty
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
        study_tags = pd.read_sql('SELECT FreqCode FROM tblMasterTag WHERE TagType == "Study" OR TagType == "Beacon"',con = conn).FreqCode.values
        def id_to_freq(row,channelDict):
            channel = row['ChannelID']
            if np.int(channel) in channelDict:
                return channelDict[np.int(channel)]
            else:
                return '888'

        # with our data row, extract information using pandas fwf import procedure
        if lotek400 == False:
            telemDat = pd.read_fwf(os.path.join(fileName),colspecs = [(0,5),(5,14),(14,23),(23,31),(31,46),(46,54)],names = ['DayNumber','Time','ChannelID','TagID','Antenna','Power'],skiprows = dataRow)
            telemDat = telemDat.iloc[:-2]                                                   # remove last two rows, Lotek adds garbage at the end
            telemDat.dropna(inplace = True)
            if len(telemDat) > 0:
                if ant_to_rec_dict == None:
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
                    telemDat['fileName'] = np.repeat(rxfile,len(telemDat))            #Made change here as above--taking jsut the file name and writing it to the dataset.  Note naming issue.
                    telemDat['recID'] = np.repeat(recName,len(telemDat))
                    telemDat['noiseRatio'] = noiseRatio(5.0,telemDat,study_tags)
                    telemDat['ScanTime'] = np.repeat(scanTime,len(telemDat))
                    telemDat['Channels'] = np.repeat(channels,len(telemDat))
                    telemDat['RecType'] = np.repeat(recType,len(telemDat))
                    tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
                    index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                    telemDat.set_index(index,inplace = True,drop = False)
                    telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')
                else:
                    site = ant_to_rec_dict[ant]
                    telemDat_sub = telemDat[telemDat.Antenna == str(ant)]
                    telemDat_sub['Frequency'] = telemDat_sub.apply(id_to_freq, axis = 1, args = (channelDict,))
                    telemDat_sub = telemDat_sub[telemDat_sub.Frequency != '888']
                    telemDat_sub = telemDat_sub[telemDat_sub.TagID != 999]
                    telemDat_sub['FreqCode'] = telemDat_sub['Frequency'].astype(str) + ' ' + telemDat_sub['TagID'].astype(int).astype(str)
                    telemDat_sub['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telemDat_sub))
                    telemDat_sub['Date'] = telemDat_sub['day0'] + pd.to_timedelta(telemDat_sub['DayNumber'].astype(int), unit='d')
                    telemDat_sub['Date'] = telemDat_sub.Date.astype('str')
                    telemDat_sub['timeStamp'] = pd.to_datetime(telemDat_sub['Date'] + ' ' + telemDat_sub['Time'])# create timestamp field from date and time and apply to index
                    telemDat.drop(['day0','DayNumber'],axis = 1, inplace = True)
                    telemDat_sub['Epoch'] = (telemDat_sub['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                    telemDat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                    telemDat_sub['fileName'] = np.repeat(rxfile,len(telemDat_sub))            #Made change here as above--taking jsut the file name and writing it to the dataset.  Note naming issue.
                    telemDat_sub['recID'] = np.repeat(recName,len(telemDat_sub))
                    telemDat_sub['noiseRatio'] = noiseRatio(5.0,telemDat_sub,study_tags)
                    telemDat_sub['ScanTime'] = np.repeat(scanTime,len(telemDat_sub))
                    telemDat_sub['Channels'] = np.repeat(channels,len(telemDat_sub))
                    telemDat_sub['RecType'] = np.repeat(recType,len(telemDat_sub))
                    tuples = zip(telemDat_sub.FreqCode.values,telemDat_sub.recID.values,telemDat_sub.Epoch.values)
                    index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                    telemDat_sub.set_index(index,inplace = True,drop = False)
                    telemDat_sub.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')


        else:
            telemDat = pd.read_fwf(os.path.join(fileName),colspecs = [(0,6),(6,14),(14,22),(22,27),(27,35),(35,41),(41,48),(48,56),(56,67),(67,80)],names = ['DayNumber_Start','StartTime','ChannelID','TagID','Antenna','Power','Data','Events','DayNumber_End','EndTime'],skiprows = dataRow)

            telemDat.dropna(inplace = True)
#            if len(telemDat) > 0:
#                telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
#                telemDat = telemDat[telemDat.Frequency != '888']
#                telemDat = telemDat[telemDat.TagID != 999]
#                telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
#                telemDat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telemDat))
#                telemDat['Date_Start'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber_Start'].astype(int), unit='d')
#                telemDat['Date_Start'] = telemDat.Date_Start.astype('str')
#                telemDat['Date_End'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber_End'].astype(int), unit='d')
#                telemDat['Date_End'] = telemDat.Date_End.astype('str')
#                telemDat['timeStamp'] = pd.to_datetime(telemDat['Date_Start'] + ' ' + telemDat['StartTime'])# create timestamp field from date and time and apply to index
#                telemDat['time_end'] = pd.to_datetime(telemDat['Date_End'] + ' ' + telemDat['EndTime'])# create timestamp field from date and time and apply to index
#                telemDat.drop(['day0','DayNumber_Start','DayNumber_End'],axis = 1, inplace = True)
#                telemDat['duration'] = (telemDat.time_end - telemDat.timeStamp).astype('timedelta64[s]')
#                telemDat['events_per_duration'] = telemDat.Events / telemDat.duration
#                telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
#                telemDat.drop (['Date_Start','Date_End','time_end','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
#                telemDat['fileName'] = np.repeat(fileName,len(telemDat))
#                telemDat['recID'] = np.repeat(recName,len(telemDat))
#                tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
#                index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
#                telemDat.set_index(index,inplace = True,drop = False)
#                telemDat.to_sql('tblRaw_Lotek400',con = conn,index = False, if_exists = 'append')
            if len(telemDat) > 0:
                telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
                telemDat = telemDat[telemDat.Frequency != '888']
                telemDat = telemDat[telemDat.TagID != 999]
                telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
                telemDat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telemDat))
                telemDat['Date_Start'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber_Start'].astype(int), unit='d')
                telemDat['Date_Start'] = telemDat.Date_Start.astype('str')
                telemDat['Date_End'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber_End'].astype(int), unit='d')
                telemDat['Date_End'] = telemDat.Date_End.astype('str')
                telemDat['timeStamp'] = pd.to_datetime(telemDat['Date_Start'] + ' ' + telemDat['StartTime'])# create timestamp field from date and time and apply to index
                telemDat['time_end'] = pd.to_datetime(telemDat['Date_End'] + ' ' + telemDat['EndTime'])# create timestamp field from date and time and apply to index
                telemDat.drop(['day0','DayNumber_Start','DayNumber_End'],axis = 1, inplace = True)
                telemDat['duration'] = (telemDat.time_end - telemDat.timeStamp).astype('timedelta64[s]')
                telemDat['events_per_duration'] = telemDat.Events / telemDat.duration
                telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                telemDat.drop (['Date_Start','Date_End','time_end','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                telemDat['fileName'] = np.repeat(rxfile,len(telemDat))            #This is the 4th time I'm assigning file to fileName in the saved data table.
                telemDat['recID'] = np.repeat(recName,len(telemDat))
                telemDat['ScanTime'] = np.repeat(scanTime,len(telemDat))
                telemDat['Channels'] = np.repeat(channels,len(telemDat))
                telemDat['RecType'] = np.repeat(recType,len(telemDat))
                telemDat.drop(['StartTime','Data','Events','EndTime','duration','events_per_duration'], axis = 1, inplace = True)
                tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
                index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                telemDat.set_index(index,inplace = True,drop = False)
                telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')


    # add receiver parameters to database
#    recParamLine = [(recName,recType,scanTime,channels,fileName)]
#    conn.executemany('INSERT INTO tblReceiverParameters VALUES (?,?,?,?,?)',recParamLine)
    conn.commit()
    c.close()

def telemDataImport(site,recType,file_directory,projectDB,switch = False, scanTime = None, channels = None, ant_to_rec_dict = None):
    tFiles = os.listdir(file_directory)
    for f in tFiles:
        f_dir = os.path.join(file_directory,f)
        rxfile=f
        if recType == 'lotek':
            lotek_import(f_dir,rxfile,projectDB,site,ant_to_rec_dict)
        elif recType == 'orion':
            orionImport(f_dir,rxfile,projectDB,site,switch, scanTime, channels, ant_to_rec_dict)
        else:
            print ("There currently is not an import routine created for this receiver type.  Please try again")
        print ("File %s imported"%(f))
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
        sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType FROM tblRaw WHERE FreqCode == '%s' AND tblRaw.recID == '%s';"%(i,site)
        self.histDF = pd.read_sql(sql,con = conn, parse_dates  = 'timeStamp',coerce_float  = True)
        sql = 'SELECT PulseRate,MortRate FROM tblMasterTag WHERE FreqCode == "%s"'%(i)
        rates = pd.read_sql(sql,con = conn)
        self.rates = rates
        sql = 'SELECT FreqCode FROM tblMasterTag'
        allTags = pd.read_sql(sql,con = conn)
        sql = 'SELECT * FROM tblAlgParams'
        algParams = pd.read_sql(sql,con = conn)
        #sql = 'SELECT * FROM tblReceiverParameters WHERE RecID == "%s"'%(site)
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
        #self.histDF['recType'] = np.repeat(self.recType,len(self.histDF))

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
            self.PulseRate = 9.0
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

#    # get data
#    conn = sqlite3.connect(projectDB, timeout=30.0)
#    c = conn.cursor()
#    sql = "SELECT FreqCode, Epoch FROM tblRaw WHERE recID == '%s'"%(site)
#    allData = pd.read_sql(sql,con = conn, coerce_float  = True)
#    c.close()
#    allData.sort_values(by = 'Epoch', inplace = True)
#    allData.set_index('Epoch', drop = False, inplace = True)

    # calculate predictors
    histDF['Detection'] = np.repeat(trainee.plausible,len(histDF))
    histDF['lag'] = histDF.Epoch.diff().abs()                                      # calculate the difference in seconds until the next detection
    histDF['lagDiff'] = histDF.lag.diff()
    histDF.lagDiff.fillna(999999999,inplace = True)
    histDF = detHist(histDF,trainee.PulseRate,trainee.det,training = True)             # calculate detection history
    histDF['seriesHit'] = histDF['lag'].apply(lambda x: 1 if x in trainee.alive_factors else 0)
    #histDF['FishCount'] = histDF.apply(fishCount, axis = 1, args = (trainee,allData))
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
    def __init__(self,i,site,classifyFields,projectDB,scratchWS,training_data,informed_prior = True,training = None, reclass_iter = None):
        '''when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).
        '''
        conn = sqlite3.connect(projectDB, timeout=30.0)
        c = conn.cursor()
        if reclass_iter == None:
            sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType, fileName FROM tblRaw WHERE FreqCode == '%s' AND recID == '%s';"%(i,site)
            self.histDF = pd.read_sql(sql,con = conn,coerce_float  = True)


        else:
            sql = "SELECT FreqCode, Epoch, recID, timeStamp, Power, noiseRatio, ScanTime, Channels, RecType, fileName FROM tblClassify_%s_%s WHERE FreqCode == '%s' AND test == '1';"%(site,reclass_iter-1,i)
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



def likelihood(assumption,classify_object,status = 'A'):
    '''calculates likelihood based on true or false assumption and fields provided to classify object'''
    fields = classify_object.fields
    if status == 'A':
        trueFields = {'conRecLength':'LconRecT_A','consDet':'LconsDetT_A','hitRatio':'LHitRatioT_A','noiseRatio':'LnoiseT','seriesHit':'LseriesHitT_A','power':'LPowerT','lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF_A','consDet':'LconsDetF_A','hitRatio':'LHitRatioF_A','noiseRatio':'LnoiseF','seriesHit':'LseriesHitF_A','power':'LPowerF','lagDiff':'LlagF'}
    elif status == 'M':
        trueFields = {'conRecLength':'LconRecT_M','consDet':'LconsDetT_M','hitRatio':'LHitRatioT_M','noiseRatio':'LnoiseT','seriesHit':'LseriesHitT_M','power':'LPowerT','lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF_M','consDet':'LconsDetF_M','hitRatio':'LHitRatioF_M','noiseRatio':'LnoiseF','seriesHit':'LseriesHitF_M','power':'LPowerF','lagDiff':'LlagF'}
    else:
        trueFields = {'conRecLength':'LconRecT','consDet':'LconsDetT','hitRatio':'LHitRatioT','noiseRatio':'LnoiseT','seriesHit':'LseriesHitT','power':'LPowerT','lagDiff':'LlagT'}
        falseFields = {'conRecLength':'LconRecF','consDet':'LconsDetF','hitRatio':'LHitRatioF','noiseRatio':'LnoiseF','seriesHit':'LseriesHitF','power':'LPowerF','lagDiff':'LlagF'}

    if status == 'cross':
        if assumption == True:
            if len(fields) == 1:
                return classify_object.testDat[trueFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]] * classify_object.testDat[trueFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]] * classify_object.testDat[trueFields[fields[2]]]  * classify_object.testDat[trueFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]] * classify_object.testDat[trueFields[fields[2]]]  * classify_object.testDat[trueFields[fields[3]]]  * classify_object.testDat[trueFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]] * classify_object.testDat[trueFields[fields[2]]]  * classify_object.testDat[trueFields[fields[3]]]  * classify_object.testDat[trueFields[fields[4]]]  * classify_object.testDat[trueFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.testDat[trueFields[fields[0]]] * classify_object.testDat[trueFields[fields[1]]] * classify_object.testDat[trueFields[fields[2]]]  * classify_object.testDat[trueFields[fields[3]]]  * classify_object.testDat[trueFields[fields[4]]]  * classify_object.testDat[trueFields[fields[5]]]  * classify_object.testDat[trueFields[fields[6]]]

        elif assumption == False:
            if len(fields) == 1:
                return classify_object.testDat[falseFields[fields[0]]]
            elif len(fields) == 2:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]]
            elif len(fields) == 3:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]] * classify_object.testDat[falseFields[fields[2]]]
            elif len(fields) == 4:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]] * classify_object.testDat[falseFields[fields[2]]]  * classify_object.testDat[falseFields[fields[3]]]
            elif len(fields) == 5:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]] * classify_object.testDat[falseFields[fields[2]]]  * classify_object.testDat[falseFields[fields[3]]]  * classify_object.testDat[falseFields[fields[4]]]
            elif len(fields) == 6:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]] * classify_object.testDat[falseFields[fields[2]]]  * classify_object.testDat[falseFields[fields[3]]]  * classify_object.testDat[falseFields[fields[4]]]  * classify_object.testDat[falseFields[fields[5]]]
            elif len(fields) == 7:
                return classify_object.testDat[falseFields[fields[0]]] * classify_object.testDat[falseFields[fields[1]]] * classify_object.testDat[falseFields[fields[2]]]  * classify_object.testDat[falseFields[fields[3]]]  * classify_object.testDat[falseFields[fields[4]]]  * classify_object.testDat[falseFields[fields[5]]]  * classify_object.testDat[falseFields[fields[6]]]

    else:
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

def create_training_data(site,projectDB,reclass_iter = None):
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
        sql = "SELECT * FROM tblTrain"# WHERE RecType == '%s'"%(classify_object.recType)
        trainDF = pd.read_sql(sql,con = conn)
        trainDF = trainDF[trainDF.recType == recType]
    else:
        trainDF = pd.read_sql("select * from tblTrain",con=conn)#This will read in tblTrain and create a pandas dataframe
        trainDF = trainDF[trainDF.recType == recType]

#            classDF = pd.read_sql("select test, FreqCode,Power,lag,lagDiff,fishCount,conRecLength,consDet,detHist,hitRatio,noiseRatio,seriesHit,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s_%s"%(site,classify_object.reclass_iter-1),con=conn)
        classDF = pd.read_sql("select test, FreqCode,Power,noiseRatio, lag,lagDiff,conRecLength_A,consDet_A,detHist_A,hitRatio_A,seriesHit_A,conRecLength_M,consDet_M,detHist_M,hitRatio_M,seriesHit_M,postTrue_A,postTrue_M,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from tblClassify_%s_%s"%(site,reclass_iter-1),con=conn)
       # classDF = classDF[classDF.postTrue_A >= classDF.postTrue_M]
        classDF.drop(['conRecLength_M','consDet_M','detHist_M','hitRatio_M','seriesHit_M'], axis = 1, inplace = True)
        classDF.rename(columns = {'conRecLength_A':'conRecLength','consDet_A':'consDet','detHist_A':'detHist','hitRatio_A':'hitRatio','seriesHit_A':'seriesHit'}, inplace = True)

        trainDF = trainDF[trainDF.Detection==0]
        classDF = classDF[classDF.test==1]
        classDF['Channels']=np.repeat(1,len(classDF))
#       classDF.rename(columns={"test":"Detection","fishCount":"FishCount","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
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
        # get data
    #    conn = sqlite3.connect(projectDB, timeout=30.0)
    #    c = conn.cursor()
    #    sql = "SELECT FreqCode, Epoch FROM tblRaw WHERE recID == '%s'"%(site)
    #    allData = pd.read_sql(sql,con = conn, coerce_float  = True)
    #    c.close()

    #    allData.sort_values(by = 'Epoch', inplace = True)
    #    allData.set_index('Epoch', drop = False, inplace = True)

        # calculate parameters
        classify_object.histDF['lag'] = classify_object.histDF.Epoch.diff().abs()
        classify_object.histDF['lagDiff'] = classify_object.histDF.lag.diff()
        classify_object.histDF['seriesHit_A'] = classify_object.histDF['lag'].apply(lambda x: 1 if x in classify_object.alive_factors else 0)
        classify_object.histDF['seriesHit_M'] = classify_object.histDF['lag'].apply(lambda x: 1 if x in classify_object.dead_factors else 0)
        classify_object.histDF = detHist(classify_object.histDF,classify_object.PulseRate,classify_object.det)             # calculate detection history
        classify_object.histDF = detHist(classify_object.histDF,classify_object.MortRate,classify_object.det,'M')             # calculate detection history
        classify_object.histDF['powerBin'] = (classify_object.histDF.Power//5)*5
        classify_object.histDF['noiseBin'] = (classify_object.histDF.noiseRatio//.1)*.1
        classify_object.histDF['lagDiffBin'] = (classify_object.histDF.lagDiff//10)*10

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
        trainDF['lagDiffBin'] = (trainDF.lagDiff//10)*10

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
        seriesHitCount = trainDF.groupby(['Detection','seriesHit'])['seriesHit'].count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitACountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HT','seriesHit_A'], right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitACountT':'seriesHitACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HF','seriesHit_A'], right_on = ['HF','seriesHit'])
        classify_object.histDF.drop(labels = ['seriesHit_x','seriesHit_y'], axis = 1, inplace = True)

        # count the number of instances of consective detections by detection class and write to data frame
        consDetCount = trainDF.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetACountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HT','consDet_A'], right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetACountT':'consDetACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HF','consDet_A'], right_on = ['HF','consDet'])
        classify_object.histDF.drop(labels = ['consDet_x','consDet_y'], axis = 1, inplace = True)

        # count the number of instances of certain detection histories by detection class and write to data frame
        detHistCount = trainDF.groupby(['Detection','detHist'])['detHist'].count()
        detHistCount = pd.Series(detHistCount, name = 'detHistACountT')
        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HT','detHist_A'],right_on =['HT','detHist'])
        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistACountT':'detHistACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HF','detHist_A'],right_on =['HF','detHist'])
        classify_object.histDF.drop(labels = ['detHist_x','detHist_y'], axis = 1, inplace = True)

        # count the number of instances of consecutive record lengths by detection class and write to data frame
        conRecLengthCount = trainDF.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthACountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HT','conRecLength_A'], right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthACountT':'conRecLengthACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HF','conRecLength_A'], right_on = ['HF','conRecLength'])
        classify_object.histDF.drop(labels = ['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)

        # count the number of instances of hit ratios by detection class and write to data frame
        hitRatioCount = trainDF.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioACountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HT','hitRatio_A'], right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioACountT':'hitRatioACountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HF','hitRatio_A'], right_on = ['HF','hitRatio'])
        classify_object.histDF.drop(labels = ['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # Power
        powerCount = trainDF.groupby(['Detection','powerBin'])['powerBin'].count()
        powerCount = pd.Series(powerCount, name = 'powerCount_T')
        powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = powerCount, how = u'left', left_on = ['HT','powerBin'], right_on = ['HT','powerBin'])
        powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = powerCount, how = u'left', left_on = ['HF','powerBin'], right_on = ['HF','powerBin'])

        # Make a Count of the predictor variables and join to training data frame - For ALIVE Strings
        seriesHitCount = trainDF.groupby(['Detection','seriesHit'])['seriesHit'].count()
        seriesHitCount = pd.Series(seriesHitCount, name = 'seriesHitMCountT')
        seriesHitCount = pd.DataFrame(seriesHitCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HT','seriesHit_M'], right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitMCountT':'seriesHitMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = seriesHitCount, how = u'left',left_on = ['HF','seriesHit_M'], right_on = ['HF','seriesHit'])
        classify_object.histDF.drop(labels = ['seriesHit_x','seriesHit_y'], axis = 1, inplace = True)

        # count the number of instances of consective detections by detection class and write to data frame
        consDetCount = trainDF.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetMCountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HT','consDet_M'], right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetMCountT':'consDetMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = consDetCount, how = u'left', left_on = ['HF','consDet_M'], right_on = ['HF','consDet'])
        classify_object.histDF.drop(labels = ['consDet_x','consDet_y'], axis = 1, inplace = True)

        # count the number of instances of certain detection histories by detection class and write to data frame
        detHistCount = trainDF.groupby(['Detection','detHist'])['detHist'].count()
        detHistCount = pd.Series(detHistCount, name = 'detHistMCountT')
        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HT','detHist_M'],right_on =['HT','detHist'])
        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistMCountT':'detHistMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = detHistCount, how = u'left', left_on = ['HF','detHist_M'],right_on =['HF','detHist'])
        classify_object.histDF.drop(labels = ['detHist_x','detHist_y'], axis = 1, inplace = True)

        # count the number of instances of consecutive record lengths by detection class and write to data frame
        conRecLengthCount = trainDF.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthMCountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HT','conRecLength_M'], right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthMCountT':'conRecLengthMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = conRecLengthCount, how = u'left', left_on = ['HF','conRecLength_M'], right_on = ['HF','conRecLength'])
        classify_object.histDF.drop(labels = ['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)

        # count the number of instances of hit ratios by detection class and write to data frame
        hitRatioCount = trainDF.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioMCountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HT','hitRatio_M'], right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioMCountT':'hitRatioMCountF'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = hitRatioCount, how = u'left', left_on = ['HF','hitRatio_M'], right_on = ['HF','hitRatio'])
        classify_object.histDF.drop(labels = ['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # NoiseR$atio
        noiseCount = trainDF.groupby(['Detection','noiseBin'])['noiseBin'].count()
        noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
        noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = noiseCount, how = u'left', left_on = ['HT','noiseBin'], right_on = ['HT','noiseBin'])
        noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = noiseCount, how = u'left', left_on = ['HF','noiseBin'], right_on = ['HF','noiseBin'])

        # Lag Bin
        lagCount = trainDF.groupby(['Detection','lagDiffBin'])['lagDiffBin'].count()
        lagCount = pd.Series(lagCount, name = 'lagDiffCount_T')
        lagCount = pd.DataFrame(lagCount).reset_index().rename(columns = {'Detection':'HT'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = lagCount, how = u'left', left_on = ['HT','lagDiffBin'], right_on = ['HT','lagDiffBin'])
        lagCount = lagCount.rename(columns = {'HT':'HF','lagDiffCount_T':'lagDiffCount_F'})
        classify_object.histDF = pd.merge(left = classify_object.histDF, right = lagCount, how = u'left', left_on = ['HF','lagDiffBin'], right_on = ['HF','lagDiffBin'])

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


        # calculation of the probability of a false positive given the data
        classify_object.histDF['priorF'] = round(priorCountF/float(len(trainDF)),5)                    # calculate the prior probability of a false detection from the training dataset
        classify_object.histDF['LconRecF_A'] = (classify_object.histDF['conRecLengthACountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LseriesHitF_A'] = (classify_object.histDF['seriesHitACountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetF_A'] = (classify_object.histDF['consDetACountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioF_A'] = (classify_object.histDF['hitRatioACountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerF'] = (classify_object.histDF['powerCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseF'] = (classify_object.histDF['noiseCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagF'] = (classify_object.histDF['lagDiffCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive


        # calculation of the probability of a true detection given the data
        classify_object.histDF['priorT'] = round(priorCountT/float(len(trainDF)),5)                    # calculate the prior probability of a true detection from the training dataset
        classify_object.histDF['LconRecT_A'] = (classify_object.histDF['conRecLengthACountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive                           # calculate the posterior probability of a false positive detection given this row's detection history, power bin and noise ratio
        classify_object.histDF['LseriesHitT_A'] = (classify_object.histDF['seriesHitACountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetT_A'] = (classify_object.histDF['consDetACountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioT_A'] = (classify_object.histDF['hitRatioACountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerT'] = (classify_object.histDF['powerCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseT'] = (classify_object.histDF['noiseCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagT'] = (classify_object.histDF['lagDiffCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive

        # calculation of the probability of a false positive given the data
        classify_object.histDF['priorF'] = round(priorCountF/float(len(trainDF)),5)                    # calculate the prior probability of a false detection from the training dataset
        classify_object.histDF['LconRecF_M'] = (classify_object.histDF['conRecLengthMCountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        classify_object.histDF['LseriesHitF_M'] = (classify_object.histDF['seriesHitMCountF'] + 1)/classify_object.histDF['LDenomCount_F']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetF_M'] = (classify_object.histDF['consDetMCountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioF_M'] = (classify_object.histDF['hitRatioMCountF'] + 1)/classify_object.histDF['LDenomCount_F']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerF'] = (classify_object.histDF['powerCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseF'] = (classify_object.histDF['noiseCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagF'] = (classify_object.histDF['lagDiffCount_F'] + 1)/classify_object.histDF['LDenomCount_F']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive


        # calculation of the probability of a true detection given the data
        classify_object.histDF['priorT'] = round(priorCountT/float(len(trainDF)),5)                    # calculate the prior probability of a true detection from the training dataset
        classify_object.histDF['LconRecT_M'] = (classify_object.histDF['conRecLengthMCountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive                           # calculate the posterior probability of a false positive detection given this row's detection history, power bin and noise ratio
        classify_object.histDF['LseriesHitT_M'] = (classify_object.histDF['seriesHitMCountT'] + 1)/classify_object.histDF['LDenomCount_T']# calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LconsDetT_M'] = (classify_object.histDF['consDetMCountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LHitRatioT_M'] = (classify_object.histDF['hitRatioMCountT'] + 1)/classify_object.histDF['LDenomCount_T']    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LPowerT'] = (classify_object.histDF['powerCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LnoiseT'] = (classify_object.histDF['noiseCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        classify_object.histDF['LlagT'] = (classify_object.histDF['lagDiffCount_T'] + 1)/classify_object.histDF['LDenomCount_T']     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive


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
        dtype = {'FreqCode':'TEXT', 'Epoch': 'INTEGER', 'recID':'TEXT','timeStamp':'TIMESTAMP', 'Power':'REAL', 'ScanTime':'REAL',
                 'Channels':'REAL', 'RecType':'TEXT', 'recID1':'TEXT', 'RowSeconds':'REAL', 'lag':'REAL', 'lagDiff':'REAL','noiseRatio':'REAL',
                 'seriesHit_A':'INTEGER', 'seriesHit_M':'INTEGER', 'consDet_A':'INTEGER', 'hitRatio_A':'REAL', 'detHist_A':'TEXT',
                 'conRecLength_A':'INTEGER', 'consDet_M':'INTEGER', 'hitRatio_M':'REAL', 'detHist_M':'TEXT','conRecLength_M':'INTEGER',
                 'powerBin':'REAL', 'lagDiffBin':'REAL', 'HT':'INTEGER', 'HF':'INTEGER', 'seriesHitACountT':'INTEGER',
                 'seriesHitACountF':'INTEGER', 'consDetACountT':'INTEGER','consDetACountF':'INTEGER', 'detHistACountT':'INTEGER',
                 'detHistACountF':'INTEGER','conRecLengthACountT':'INTEGER', 'conRecLengthACountF':'INTEGER', 'hitRatioACountT':'INTEGER',
                 'hitRatioACountF':'INTEGER', 'powerCount_T':'INTEGER','powerCount_F':'INTEGER','noiseCount_T':'INTEGER','noiseCount_F':'INTEGER','seriesHitMCountT':'INTEGER',
                 'seriesHitMCountF':'INTEGER', 'consDetMCountT':'INTEGER', 'consDetMCountF':'INTEGER', 'detHistMCountT':'INTEGER',
                 'detHistMCountF':'INTEGER', 'conRecLengthMCountT':'INTEGER', 'conRecLengthMCountF':'INTEGER', 'hitRatioMCountT':'INTEGER',
                 'hitRatioMCountF':'INTEGER', 'lagDiffCount_T':'INTEGER', 'lagDiffCount_F':'INTEGER', 'priorCount_T':'INTEGER',
                 'priorCount_F':'INTEGER', 'LDenomCount_T':'INTEGER', 'LDenomCount_F':'INTEGER', 'priorF':'REAL', 'LconRecF_A':'REAL',
                 'LseriesHitF_A':'REAL', 'LconsDetF_A':'REAL', 'LHitRatioF_A':'REAL', 'LPowerF':'REAL','LnoiseF':'REAL', 'LlagF':'REAL','priorT':'REAL',
                 'LconRecT_A':'REAL', 'LseriesHitT_A':'REAL', 'LconsDetT_A':'REAL', 'LHitRatioT_A':'REAL','LPowerT':'REAL','LnoiseT':'REAL','LlagT':'REAL',
                 'LconRecF_M':'REAL', 'LseriesHitF_M':'REAL', 'LconsDetF_M':'REAL','LHitRatioF_M':'REAL', 'LconRecT_M':'REAL',
                 'LseriesHitT_M':'REAL', 'LconsDetT_M':'REAL','LHitRatioT_M':'REAL', 'LikelihoodTrue_A':'REAL', 'LikelihoodFalse_A':'REAL',
                 'LikelihoodTrue_M':'REAL', 'LikelihoodFalse_M':'REAL', 'logLikelihoodRatio_A':'REAL','logLikelihoodRatio_M':'REAL',
                 'postTrue_A':'REAL', 'postFalse_A':'REAL', 'postTrue_M':'REAL','postFalse_M':'REAL', 'test':'INTEGER'}
        dat.to_sql(out_name,con = conn,index = False, if_exists = 'append', chunksize = 1000, dtype = dtype)
        os.remove(os.path.join(inputWS,f))
        del dat
    #c.execute('''CREATE INDEX idx_combined_%s ON tblClassify_%s (recID,FreqCode,Epoch)'''%(site,site))
    c.close()


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
            if rec_list == None:
                sql = "select * from tblTrain WHERE recType == '%s'"%(recType)
                # get receivers in study of this type
                recs = pd.read_sql("select * from tblMasterReceiver WHERE recType == '%s'"%(recType),con=conn)['recID'].values
            else:
                sql = "SELECT * FROM tblTrain  WHERE recID == '%s'"%(rec_list[0])
                for i in rec_list[1:]:
                    sql = sql + "OR recID == '%s'"%(i)
                recs = rec_list
            self.trainDF = pd.read_sql(sql,con=conn,coerce_float  = True)#This will read in tblTrain and create a pandas dataframe

            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            for i in recs:
                max_iter = 1
                while curr_idx < len(tbls) - 1:
                    for j in tbls:
                        if i in j[0]:
                            if int(j[0][-1]) >= max_iter:
                                max_iter = int(j[0][-1])
                                max_iter_dict[i] = j[0]
                        curr_idx = curr_idx + 1
                curr_idx = 0
            del i, j, curr_idx
            # once we have a hash table of receiver to max classification, extract the classification dataset
            self.classDF = pd.DataFrame()
            for i in max_iter_dict:
                classDat = pd.read_sql("select test, FreqCode,Power,noiseRatio, lag,lagDiff,conRecLength_A,consDet_A,detHist_A,hitRatio_A,seriesHit_A,conRecLength_M,consDet_M,detHist_M,hitRatio_M,seriesHit_M,postTrue_A,postTrue_M,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from %s"%(max_iter_dict[i]),con=conn)
                #classDat = classDat[classDat.postTrue_A >= classDat.postTrue_M]
                classDat.drop(['conRecLength_M','consDet_M','detHist_M','hitRatio_M','seriesHit_M'], axis = 1, inplace = True)
                classDat.rename(columns = {'conRecLength_A':'conRecLength','consDet_A':'consDet','detHist_A':'detHist','hitRatio_A':'hitRatio','seriesHit_A':'seriesHit'}, inplace = True)
                self.classDF = self.classDF.append(classDat)

            self.trainDF = self.trainDF[self.trainDF.Detection==0]
            self.classDF = self.classDF[self.classDF.test==1]
            self.classDF['Channels']=np.repeat(1,len(self.classDF))
            self.classDF.rename(columns={"test":"Detection","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
            self.trainDF = self.trainDF.append(self.classDF)
            del self.classDF
            c.close()

        self.k = folds
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
        self.trainDF['lagDiffBin'] = (self.trainDF.lagDiff//10)*10
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
        self.testDat = pd.merge(left = self.testDat, right = seriesHitCount, how = u'left',left_on = ['HT','seriesHit'], right_on = ['HT','seriesHit'])
        seriesHitCount = seriesHitCount.rename(columns = {'HT':'HF','seriesHitCountT':'seriesHitCountF'})
        self.testDat = pd.merge(left = self.testDat, right = seriesHitCount, how = u'left',left_on = ['HF','seriesHit'], right_on = ['HF','seriesHit'])
        #testDat.drop(['seriesHit_x','seriesHit_y'], axis = 1, inplace = True)

        # Consecutive Detections
        consDetCount = self.trainDat.groupby(['Detection','consDet'])['consDet'].count()
        consDetCount = pd.Series(consDetCount, name = 'consDetCountT')
        consDetCount = pd.DataFrame(consDetCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = consDetCount, how = u'left', left_on = ['HT','consDet'], right_on = ['HT','consDet'])
        consDetCount = consDetCount.rename(columns = {'HT':'HF','consDetCountT':'consDetCountF'})
        self.testDat = pd.merge(left = self.testDat, right = consDetCount, how = u'left', left_on = ['HF','consDet'], right_on = ['HF','consDet'])
        #testDat.drop(['consDet_x','consDet_y'], axis = 1, inplace = True)

#        # Detection History
#        detHistCount = self.trainDat.groupby(['Detection','detHist'])['detHist'].count()
#        detHistCount = pd.Series(detHistCount, name = 'detHistCountT')
#        detHistCount = pd.DataFrame(detHistCount).reset_index().rename(columns = {'Detection':'HT'})
#        self.testDat = pd.merge(how = 'left', left = self.testDat, right = detHistCount, left_on = ['HT','detHist'],right_on =['HT','detHist'])
#        detHistCount = detHistCount.rename(columns = {'HT':'HF','detHistCountT':'detHistCountF'})
#        self.testDat = pd.merge(how = 'left', left = self.testDat, right = detHistCount, left_on = ['HF','detHist'],right_on =['HF','detHist'])
#        #testDat.drop(['detHist_x','detHist_y'], axis = 1, inplace = True)

        # Consecutive Record Length
        conRecLengthCount = self.trainDat.groupby(['Detection','conRecLength'])['conRecLength'].count()
        conRecLengthCount = pd.Series(conRecLengthCount, name = 'conRecLengthCountT')
        conRecLengthCount = pd.DataFrame(conRecLengthCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = conRecLengthCount, how = u'left', left_on = ['HT','conRecLength'], right_on = ['HT','conRecLength'])
        conRecLengthCount = conRecLengthCount.rename(columns = {'HT':'HF','conRecLengthCountT':'conRecLengthCountF'})
        self.testDat = pd.merge(left = self.testDat, right = conRecLengthCount, how = u'left', left_on = ['HF','conRecLength'], right_on = ['HF','conRecLength'])
        #testDat.drop(['conRecLength_x','conRecLength_y'], axis = 1, inplace = True)

        # Hit Ratio
        hitRatioCount = self.trainDat.groupby(['Detection','hitRatio'])['hitRatio'].count()
        hitRatioCount = pd.Series(hitRatioCount, name = 'hitRatioCountT')
        hitRatioCount = pd.DataFrame(hitRatioCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = hitRatioCount, how = u'left', left_on = ['HT','hitRatio'], right_on = ['HT','hitRatio'])
        hitRatioCount = hitRatioCount.rename(columns = {'HT':'HF','hitRatioCountT':'hitRatioCountF'})
        self.testDat = pd.merge(left = self.testDat, right = hitRatioCount, how = u'left', left_on = ['HF','hitRatio'], right_on = ['HF','hitRatio'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # Power
        powerCount = self.trainDat.groupby(['Detection','powerBin'])['powerBin'].count()
        powerCount = pd.Series(powerCount, name = 'powerCount_T')
        powerCount = pd.DataFrame(powerCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = powerCount, how = u'left', left_on = ['HT','powerBin'], right_on = ['HT','powerBin'])
        powerCount = powerCount.rename(columns = {'HT':'HF','powerCount_T':'powerCount_F'})
        self.testDat = pd.merge(left = self.testDat, right = powerCount, how = u'left', left_on = ['HF','powerBin'], right_on = ['HF','powerBin'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # NoiseR$atio
        noiseCount = self.trainDat.groupby(['Detection','noiseBin'])['noiseBin'].count()
        noiseCount = pd.Series(noiseCount, name = 'noiseCount_T')
        noiseCount = pd.DataFrame(noiseCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = noiseCount, how = u'left', left_on = ['HT','noiseBin'], right_on = ['HT','noiseBin'])
        noiseCount = noiseCount.rename(columns = {'HT':'HF','noiseCount_T':'noiseCount_F'})
        self.testDat = pd.merge(left = self.testDat, right = noiseCount, how = u'left', left_on = ['HF','noiseBin'], right_on = ['HF','noiseBin'])
        #testDat.drop(['hitRatio_x','hitRatio_y'], axis = 1, inplace = True)

        # Lag Bin
        lagCount = self.trainDat.groupby(['Detection','lagDiffBin'])['lagDiffBin'].count()
        lagCount = pd.Series(lagCount, name = 'lagDiffCount_T')
        lagCount = pd.DataFrame(lagCount).reset_index().rename(columns = {'Detection':'HT'})
        self.testDat = pd.merge(left = self.testDat, right = lagCount, how = u'left', left_on = ['HT','lagDiffBin'], right_on = ['HT','lagDiffBin'])
        lagCount = lagCount.rename(columns = {'HT':'HF','lagDiffCount_T':'lagDiffCount_F'})
        self.testDat = pd.merge(left = self.testDat, right = lagCount, how = u'left', left_on = ['HF','lagDiffBin'], right_on = ['HF','lagDiffBin'])

        self.testDat = self.testDat.fillna(0)                                                # Nan gives us heartburn, fill them with zeros
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
        self.testDat['priorF'] = round(priorCountF/float(len(self.trainDat)),5)                      # calculate the prior probability of a false detection from the training dataset
        self.testDat['LHitRatioF'] =(self.testDat['hitRatioCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)      # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        self.testDat['LconRecF'] = (self.testDat['conRecLengthCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1) # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive
        self.testDat['LseriesHitF'] = (self.testDat['seriesHitCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1) # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LconsDetF'] = (self.testDat['consDetCountF'] + 1)/(self.testDat['LDenomCount_F'] + 1)     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LPowerF'] = (self.testDat['powerCount_F'] + 1)/(self.testDat['LDenomCount_F'] + 1)     # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LnoiseF'] = (self.testDat['noiseCount_F'] + 1)/(self.testDat['LDenomCount_F'] + 1)    # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LlagF'] = (self.testDat['lagDiffCount_F'] + 1)/(self.testDat['LDenomCount_F']    + 1)  # calculate the likelihood of this row's particular seriesHit given the detection is a false positive


        # calculation of the probability of a true detection given the data
        self.testDat['priorT'] = round(priorCountT/float(len(self.trainDat)),5)                      # calculate the prior probability of a true detection from the training dataset
        self.testDat['LHitRatioT'] = (self.testDat['hitRatioCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1) # calculate the likelihood of this row's particular detection history occuring giving that the detection is a false positive
        self.testDat['LconRecT'] = (self.testDat['conRecLengthCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1) # calculate the likelihood of this row's particular consecutive record length given that the detection is a false positive                           # calculate the posterior probability of a false positive detection given this row's detection history, power bin and noise ratio
        self.testDat['LseriesHitT'] = (self.testDat['seriesHitCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1) # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LconsDetT'] = (self.testDat['consDetCountT'] + 1)/(self.testDat['LDenomCount_T'] + 1)  # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LPowerT'] = (self.testDat['powerCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)   # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LnoiseT'] = (self.testDat['noiseCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)  # calculate the likelihood of this row's particular seriesHit given the detection is a false positive
        self.testDat['LlagT'] = (self.testDat['lagDiffCount_T'] + 1)/(self.testDat['LDenomCount_T'] + 1)  # calculate the likelihood of this row's particular seriesHit given the detection is a false positive


        # Calculate the likelihood of each hypothesis being true
        self.testDat['LikelihoodTrue'] = likelihood(True,self,status = 'cross')
        self.testDat['LikelihoodFalse'] = likelihood(False,self,status = 'cross')


        # Calculate the posterior probability of each Hypothesis occuring
        self.testDat['postTrue'] = self.testDat['priorT'] * self.testDat['LikelihoodTrue']
        self.testDat['postFalse'] = self.testDat['priorF'] * self.testDat['LikelihoodFalse']


        self.testDat['T2F_ratio'] = self.testDat['postTrue'] / self.testDat['postFalse']
        # classify detection as true or false based on MAP hypothesis

        self.testDat['test'] = self.testDat.postTrue > self.testDat.postFalse
        self.histDF = self.histDF.append(self.testDat)
        del self.testDat, self.trainDat

    def summary(self):
        metrics = pd.crosstab(self.histDF.Detection,self.histDF.test)
        rowSum = metrics.sum(axis = 1)
        colSum = metrics.sum(axis = 0)
        self.corr_matrix = self.histDF[['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','Power','lagDiff']].apply(pd.to_numeric).corr().round(4)

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
        print ("________________________________________________________________")
        print ("The correlation matrix for all predictors variabes:             ")
        print (self.corr_matrix)
        # visualize the correlation matrix, closer to 1, the stronger the effect
        # if we are worried about multicollinearity, I would stear away from
        # variable combinations where coefficient ~ 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.corr_matrix, vmin = -1, vmax = 1)
        fig.colorbar(cax)
        ticks = np.arange(0,7,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','Power','lagDiff'])
        ax.set_yticklabels(['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','Power','lagDiff'])
        plt.show()

class classification_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS,rec_list = None, site = None, reclass_iter = 1):
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)                                              # connect to the database
        #self.class_stats_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','hitRatio','postTrue','postFalse','test','lagDiff','conRecLength','noiseRatio','fishCount', 'logLikelihoodRatio'])                # set up an empty data frame
        self.class_stats_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','noiseRatio','hitRatio_A','hitRatio_M','postTrue_A','postTrue_M','postFalse_A','postFalse_M','test','lagDiff','consDet_A', 'consDet_M','conRecLength_A', 'conRecLength_M','logLikelihoodRatio_A', 'logLikelihoodRatio_M'])                # set up an empty data frame
        self.initial_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','Power','noiseRatio','hitRatio_A','hitRatio_M','postTrue_A','postTrue_M','postFalse_A','postFalse_M','test','lagDiff','consDet_A', 'consDet_M','conRecLength_A', 'conRecLength_M','logLikelihoodRatio_A', 'logLikelihoodRatio_M'])
        self.reclass_iter = reclass_iter
        self.rec_list = rec_list
        self.site = site
        if rec_list == None:
            recSQL = "SELECT * FROM tblMasterReceiver WHERE RecType = '%s'"%(self.recType) # SQL code to import data from this node
            receivers = pd.read_sql(recSQL,con = conn)                         # import data
            receivers = receivers.recID.unique()                               # get the unique receivers associated with this node
            for i in receivers:                                                # for every receiver
                print ("Start selecting and merging data for receiver %s"%(i))

                sql = "SELECT FreqCode,Epoch,recID,Power,noiseRatio,hitRatio_A,hitRatio_M,postTrue_A,postTrue_M,postFalse_A,postFalse_M,test,lagDiff,consDet_A, consDet_M, conRecLength_A, conRecLength_M,logLikelihoodRatio_A, logLikelihoodRatio_M FROM tblClassify_%s_%s "%(i, reclass_iter)
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                self.class_stats_data = self.class_stats_data.append(dat)
                del dat

                sql = "SELECT FreqCode,Epoch,recID,Power,noiseRatio,hitRatio_A,hitRatio_M,postTrue_A,postTrue_M,postFalse_A,postFalse_M,test,lagDiff,consDet_A, consDet_M,conRecLength_A, conRecLength_M,logLikelihoodRatio_A, logLikelihoodRatio_M FROM tblClassify_%s_%s "%(i, 1)
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                self.initial_data = self.initial_data.append(dat)
                del dat

        else:
            for i in rec_list:
                print ("Start selecting and merging data for receiver %s"%(i))

                sql = "SELECT FreqCode,Epoch,recID,Power,noiseRatio,hitRatio_A,hitRatio_M,postTrue_A,postTrue_M,postFalse_A,postFalse_M,test,lagDiff,consDet_A, consDet_M,conRecLength_A, conRecLength_M,logLikelihoodRatio_A, logLikelihoodRatio_M FROM tblClassify_%s_%s"%(i, reclass_iter)
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                self.class_stats_data = self.class_stats_data.append(dat)
                del dat

                sql = "SELECT FreqCode,Epoch,recID,Power,noiseRatio,hitRatio_A,hitRatio_M,postTrue_A,postTrue_M,postFalse_A,postFalse_M,test,lagDiff,consDet_A, consDet_M,conRecLength_A, conRecLength_M,logLikelihoodRatio_A, logLikelihoodRatio_M FROM tblClassify_%s_%s "%(i, 1)
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                self.initial_data = self.initial_data.append(dat)
                del dat
        c.close()

    def classify_stats(self):
        '''function reads all classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        print ("")
        if self.rec_list != None:
            print ("Classification summary statistics report for sites %s"%(self.rec_list))
        else:
            print ("Classification summary statistics report")
        print ("----------------------------------------------------------------------------------")
        det_class_count = self.class_stats_data.groupby('test')['test'].count().to_frame()
        if len(det_class_count)>1:
            print ("")
            print ("%s detection class statistics:"%(self.recType))
            print ("The probability that a detection was classified as true was %s"%((round(float(det_class_count.at[1,'test'])/float(det_class_count.sum()),3))))
            print ("The probability that a detection was classified as fasle positive was %s"%((round(float(det_class_count.at[0,'test'])/float(det_class_count.sum()),3))))
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
    #        self.class_stats_data = cons_det_filter(self.class_stats_data)
    #
    #        print ("The consecutive detection filter described by Beeman & Perry (2012) would retain %s detections"%(self.class_stats_data.cons_det_filter.sum()))
    #        print ("A standard of three records in a row will only retain %s records"%len(self.class_stats_data[(self.class_stats_data.conRecLength_A >= 3) |(self.class_stats_data.conRecLength_M >= 3)]))
    #        print ("A standard of four records in a row will only retain %s records"%len(self.class_stats_data[(self.class_stats_data.conRecLength_A >= 4) |(self.class_stats_data.conRecLength_M >= 4)]))
    #        print ("A standard of five records in a row will only retain %s records"%len(self.class_stats_data[(self.class_stats_data.conRecLength_A >= 5) |(self.class_stats_data.conRecLength_M >= 5)]))
    #        print ("A standard of six records in a row will only retain %s records"%len(self.class_stats_data[(self.class_stats_data.conRecLength_A >= 6) |(self.class_stats_data.conRecLength_M >= 6)]))
    #
    #        print ("The algorithm retained a total of %s detections"%(self.class_stats_data.test.sum()))
            print ("----------------------------------------------------------------------------------")
            print ("Assess concordance with consecutive detection requirement (Beeman and Perry)")

            # calculate Cohen's Kappa (concordance)
            # step 1, join the final trues to the initial classification dataset
            # get true detections
            trues = self.class_stats_data[self.class_stats_data.test == 1][['FreqCode','Epoch','test']]
            trues.rename(columns = {'test':'final_test'},inplace = True)
            print (len(trues))
            # join true detections to initial data
            self.initial_data = self.initial_data.merge(trues,how = 'left',left_on = ['FreqCode','Epoch'], right_on = ['FreqCode','Epoch'])
            self.initial_data = cons_det_filter(self.initial_data)
            self.initial_data.final_test.fillna(0,inplace = True)
            self.initial_data.drop_duplicates(keep = 'first', inplace = True)

            n11 = len(self.initial_data[(self.initial_data.final_test == 1) & (self.initial_data.cons_det_filter == 1)])
            print ("The algorithm and Beeman and Perry classified the same %s recoreds as true "%(n11))
            n10 = len(self.initial_data[(self.initial_data.final_test == 1) & (self.initial_data.cons_det_filter == 0)])
            print ("The algorithm classified %s records as true while Beeman and Perry classified them as false"%(n10))
            n01 = len(self.initial_data[(self.initial_data.final_test == 0) & (self.initial_data.cons_det_filter == 1)])
            print ("The algorithm classified %s records as false while Beeman and Perry classified them as true"%(n01))
            n00 = len(self.initial_data[(self.initial_data.final_test == 0) & (self.initial_data.cons_det_filter == 0)])
            print ("The algorithm and Beeman and Perry classified the same %s records as false positive"%(n00))
            I_o = (n11 + n00)/(n11 + n10 + n01 + n00)
            print ("The observed propotion of agreement was %s"%(I_o))
            I_e = ((n11 + n01)*(n11 + n10) + (n10 + n00)*(n01 + n00))/(n11 + n10 + n01 + n00)**2
            print ("The expected agreement due to chance alone was %s"%(I_e))

            self.kappa = (I_o - I_e)/(1.- I_e)

            print ("Cohen's Kappa: %s"%(self.kappa))
            print ("----------------------------------------------------------------------------------")
            print ("Compiling Figures")
            # get data by detection class for side by side histograms
            self.class_stats_data['Power'] = self.class_stats_data.Power.astype(float)
            self.class_stats_data['lagDiff'] = self.class_stats_data.lagDiff.astype(float)
            self.class_stats_data['conRecLength_A'] = self.class_stats_data.conRecLength_A.astype(float)
            self.class_stats_data['noiseRatio'] = self.class_stats_data.noiseRatio.astype(float)
            #self.class_stats_data['fishCount'] = self.class_stats_data.fishCount.astype(float)
            self.class_stats_data['logPostRatio_A'] =np.log10(self.class_stats_data.postTrue_A.values/self.class_stats_data.postFalse_A.values)
            self.class_stats_data['logPostRatio_M'] =np.log10(self.class_stats_data.postTrue_M.values/self.class_stats_data.postFalse_M.values)

            '''Currently these figures only return data for hypothesized alive fish - but we can have dead fish in here and
            their inlcusion in the histograms may be biasing the results - find the max hit ratio, consecutive record length, and likelihood ratio
            '''
            self.class_stats_data['hitRatio_max'] = self.class_stats_data[['hitRatio_A','hitRatio_M']].max(axis = 1)
            self.class_stats_data['conRecLength_max'] = self.class_stats_data[['conRecLength_A','conRecLength_M']].max(axis = 1)
            self.class_stats_data['logLikelihoodRatio_max'] = self.class_stats_data[['logLikelihoodRatio_A','logLikelihoodRatio_M']].max(axis = 1)
            self.class_stats_data['logPostRatio_max'] = self.class_stats_data[['logPostRatio_A','logPostRatio_M']].max(axis = 1)

            trues = self.class_stats_data[self.class_stats_data.test == 1]
            falses = self.class_stats_data[self.class_stats_data.test == 0]
            self.trues = trues
            self.falses = falses

            # plot hit ratio histograms by detection class
            hitRatioBins =np.linspace(0,1.0,11)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.hitRatio_max.values, hitRatioBins, density = True)
    #        axs[1].hist(falses.hitRatio_max.values, hitRatioBins, density = True)
    #        axs[0].set_xlabel('Hit Ratio')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Hit Ratio')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Probability Density')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_hitRatioCompare_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_hitRatioCompare_class.png"%(self.recType)),bbox_inches = 'tight')
    #
    #        print ("Hit Ratio figure created, check your output workspace")

            # plot signal power histograms by detection class
            minPower = self.class_stats_data.Power.min()//5 * 5
            maxPower = self.class_stats_data.Power.max()//5 * 5
            powerBins =np.arange(minPower,maxPower+20,10)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.Power.values, powerBins, density = True)
    #        axs[1].hist(falses.Power.values, powerBins, density = True)
    #        axs[0].set_xlabel('%s Signal Power'%(self.recType))
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('%s Signal Power'%(self.recType))
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_powerCompare_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_powerCompare_class.png"%(self.recType)),bbox_inches = 'tight')
    #        print ("Signal Power figure created, check your output Workspace")

            # Lag Back Differences - how stdy are detection lags?
            lagBins =np.arange(-100,110,20)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.lagDiff.values, lagBins, density = True)
    #        axs[1].hist(falses.lagDiff.values, lagBins, density = True)
    #        axs[0].set_xlabel('Lag Differences')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Lag Differences')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_lagDifferences_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_lagDifferences_class.png"%(self.recType)),bbox_inches = 'tight')
    #        print ("Lag differences figure created, check your output Workspace")

            # Consecutive Record Length ?
            conBins =np.arange(1,12,1)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.conRecLength_max.values, conBins, density = True)
    #        axs[1].hist(falses.conRecLength_max.values, conBins, density = True)
    #        axs[0].set_xlabel('Consecutive Hit Length')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Consecutive Hit Length')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_conRecLength_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_conRecLength_class.png"%(self.recType)),bbox_inches = 'tight')
    #        print ("Consecutive Hit Length figure created, check your output Workspace")

            # Noise Ratio
            noiseBins =np.arange(0,1.1,0.1)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.noiseRatio.values, noiseBins, density = True)
    #        axs[1].hist(falses.noiseRatio.values, noiseBins, density = True)
    #        axs[0].set_xlabel('Noise Ratio')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Noise Ratio')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_noiseRatio_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #
    #        else:
    #           plt.savefig(os.path.join(self.figureWS,"%s_noiseRatio_class.png"%(self.recType)),bbox_inches = 'tight')
    #
    #        print ("Noise Ratio figure created, check your output Workspace" )

    #        # plot fish present
    #        minCount = self.class_stats_data.fishCount.min()//10 * 10
    #        maxCount = self.class_stats_data.fishCount.max()//10 * 10
    #        countBins =np.arange(minCount,maxCount+20,10)
    #
    #        plt.figure(figsize = (6,3))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.fishCount.values, countBins)
    #        axs[1].hist(falses.fishCount.values, countBins)
    #        axs[0].set_xlabel('Fish Present')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Fish Present')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Probability Density')
    #        if self.site != None:
    #           plt.savefig(os.path.join(self.figureWS,"%s_%s_fishPresentCompare_class.png"%(self.recType,self.site)),bbox_inches = 'tight')
    #        else:
    #           plt.savefig(os.path.join(self.figureWS,"%s_fishPresentCompare_class.png"%(self.recType)),bbox_inches = 'tight')

    #        print ("Fish Present Figure Created, check output workspace")

            # plot the log likelihood ratio
            minLogRatio = self.class_stats_data.logLikelihoodRatio_max.min()//1 * 1
            maxLogRatio = self.class_stats_data.logLikelihoodRatio_max.max()//1 * 1
            ratioBins =np.arange(minLogRatio,maxLogRatio+1,2)

    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.logLikelihoodRatio_max.values, ratioBins, density = True)
    #        axs[1].hist(falses.logLikelihoodRatio_max.values, ratioBins, density = True)
    #        axs[0].set_xlabel('Log Likelihood Ratio')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Log Likelihood Ratio')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_logLikeRatio_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_logLikeRatio_class.png"%(self.recType)),bbox_inches = 'tight')
    #
    #        print ("Log Likelihood Figure Created, check output workspace")

            # plot the log of the posterior ratio
            minPostRatio = self.class_stats_data.logPostRatio_max.min()
            maxPostRatio = self.class_stats_data.logPostRatio_max.max()
            postRatioBins = np.linspace(minPostRatio,maxPostRatio,10)


    #        plt.figure(figsize = (3,2))
    #        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True)
    #        axs[0].hist(trues.logPostRatio_max.values, postRatioBins, density = True)
    #        axs[1].hist(falses.logPostRatio_max.values, postRatioBins, density = True)
    #        axs[0].set_xlabel('Log Posterior Ratio')
    #        axs[0].set_title('True')
    #        axs[1].set_xlabel('Log Posterior Ratio')
    #        axs[1].set_title('False Positive')
    #        axs[0].set_ylabel('Frequency')
    #        if self.site != None:
    #            plt.savefig(os.path.join(self.figureWS,"%s_%s_%s_logPostRatio_class.png"%(self.recType,self.site,self.reclass_iter)),bbox_inches = 'tight')
    #
    #        else:
    #            plt.savefig(os.path.join(self.figureWS,"%s_logPostRatio_class.png"%(self.recType)),bbox_inches = 'tight')
    #
    #        print ("Log Posterior Ratio Figure Created, check output workspace")

            # make lattice plot for pubs
            figSize = (6,4)
            plt.figure()
            fig, axs = plt.subplots(3,4,tight_layout = True,figsize = figSize)
            # hit ratio
            axs[0,1].hist(trues.hitRatio_max.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[0,0].hist(falses.hitRatio_max.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[0,0].set_xlabel('Hit Ratio')
            axs[0,1].set_title('True')
            axs[0,1].set_xlabel('Hit Ratio')
            axs[0,0].set_title('False Positive')
            axs[0,0].set_title('A',loc = 'left')

            # consecutive record length
            axs[0,3].hist(trues.conRecLength_max.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[0,2].hist(falses.conRecLength_max.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[0,2].set_xlabel('Consecutive Hit Length')
            axs[0,3].set_title('True')
            axs[0,3].set_xlabel('Consecutive Hit Length')
            axs[0,2].set_title('False Positive')
            axs[0,2].set_title('B',loc = 'left')

            # power
            axs[1,1].hist(trues.Power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[1,0].hist(falses.Power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[1,0].set_xlabel('Signal Power')
            axs[1,1].set_xlabel('Signal Power')
            axs[1,0].set_ylabel('Probability Density')
            axs[1,0].set_title('C',loc = 'left')

            # noise ratio
            axs[1,3].hist(trues.noiseRatio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[1,2].hist(falses.noiseRatio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[1,2].set_xlabel('Noise Ratio')
            axs[1,3].set_xlabel('Noise Ratio')
            axs[1,2].set_title('D',loc = 'left')

            # lag diff
            axs[2,1].hist(trues.lagDiff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[2,0].hist(falses.lagDiff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[2,0].set_xlabel('Lag Differences')
            axs[2,1].set_xlabel('Lag Differences')
            axs[2,0].set_title('E',loc = 'left')

            # log posterior ratio
            axs[2,3].hist(trues.logPostRatio_max.values, postRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[2,2].hist(falses.logPostRatio_max.values, postRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
            axs[2,2].set_xlabel('Log Posterior Ratio')
            axs[2,3].set_xlabel('Log Posterior Ratio')
            axs[2,2].set_title('F',loc = 'left')
            if self.rec_list != None:
               plt.savefig(os.path.join(self.figureWS,"%s_lattice_class.png"%(self.recType)),bbox_inches = 'tight', dpi = 900)
            else:
               plt.savefig(os.path.join(self.figureWS,"%s_%s_lattice_class.png"%(self.recType,self.site)),bbox_inches = 'tight', dpi = 900)
        else:
           print("There were insufficient data to quantify summary statistics or histogram plots, either because there were no false positives or because there were no valid detections")

class training_results():
    '''Python class object to hold the results of false positive classification'''
    def __init__(self,recType,projectDB,figureWS,site = None):
        self.recType = recType
        self.projectDB = projectDB
        self.figureWS = figureWS
        self.site = site
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        conn = sqlite3.connect(self.projectDB)                                 # connect to the database


        if self.site == None:
            sql = "SELECT * FROM tblTrain WHERE recType = '%s'"%(self.recType)
        else:
            sql = "SELECT * FROM tblTrain WHERE recType = '%s' AND recID == '%s'"%(self.recType,self.site)
        trainDF = pd.read_sql(sql,con=conn,coerce_float  = True)#This will read in tblTrain and create a pandas dataframe

#        recs = pd.read_sql("SELECT recID from tblMasterReceiver", con = conn).recID.values
#
#        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
#        tbls = c.fetchall()
#        # iterate over the receivers to find the final classification (aka the largest _n)
#        max_iter_dict = {} # receiver:max iter
#        curr_idx = 0
#        for i in recs:
#            max_iter = 1
#            while curr_idx < len(tbls) - 1:
#                for j in tbls:
#                    if i in j[0]:
#                        if int(j[0][-1]) >= max_iter:
#                            max_iter = int(j[0][-1])
#                            max_iter_dict[i] = j[0]
#                    curr_idx = curr_idx + 1
#            curr_idx = 0
#        print (max_iter_dict)
#        del i, j, curr_idx
#        # once we have a hash table of receiver to max classification, extract the classification dataset
#        classDF = pd.DataFrame()
#        for i in max_iter_dict:
#            classDat = pd.read_sql("select test, FreqCode,Power,noiseRatio, lag,lagDiff,conRecLength_A,consDet_A,detHist_A,hitRatio_A,seriesHit_A,conRecLength_M,consDet_M,detHist_M,hitRatio_M,seriesHit_M,postTrue_A,postTrue_M,timeStamp,Epoch,RowSeconds,recID,RecType,ScanTime from %s"%(max_iter_dict[i]),con=conn)
#            #classDat = classDat[classDat.postTrue_A >= classDat.postTrue_M]
#            classDat.drop(['conRecLength_M','consDet_M','detHist_M','hitRatio_M','seriesHit_M'], axis = 1, inplace = True)
#            classDat.rename(columns = {'conRecLength_A':'conRecLength','consDet_A':'consDet','detHist_A':'detHist','hitRatio_A':'hitRatio','seriesHit_A':'seriesHit'}, inplace = True)
#            classDF = classDF.append(classDat)
#
#        trainDF = trainDF[trainDF.Detection==0]
#        classDF = classDF[classDF.test==1]
#        classDF['Channels']=np.repeat(1,len(classDF))
#        classDF.rename(columns={"test":"Detection","RowSeconds":"Seconds","RecType":"recType"},inplace=True)#inplace tells it to replace the existing dataframe
#        trainDF = trainDF.append(classDF)
#

        self.train_stats_data = trainDF
        c.close()

    def train_stats(self):
        '''function reads all classified data, generates summary statistics by receiver type,
        fish, site, classification status and other metrics, as well as generates a number of graphics
        for use in reporting.'''
        print ("")
        print ("Training summary statistics report")
        print ("The algorithm collected %s detections from %s %s receivers"%(len(self.train_stats_data),len(self.train_stats_data.recID.unique()),self.recType))
        print ("----------------------------------------------------------------------------------")
        det_class_count = self.train_stats_data.groupby('Detection')['Detection'].count().to_frame()
        print ("")
        print ("%s detection clas statistics:"%(self.recType) )
        print ("The prior probability that a detection was true was %s"%((round(float(det_class_count.at[1,'Detection'])/float(det_class_count.sum()),3))))
        print ("The prior probability that a detection was false positive was %s"%((round(float(det_class_count.at[0,'Detection'])/float(det_class_count.sum()),3))))
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

#        figSize = (3,2)
#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.hitRatio.values, hitRatioBins, density = True)
#        axs[1].hist(falses.hitRatio.values, hitRatioBins, density = True)
#        axs[0].set_xlabel('Hit Ratio')
#        axs[0].set_title('True')
#        axs[1].set_xlabel('Hit Ratio')
#        axs[1].set_title('False Positive')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#            plt.savefig(os.path.join(self.figureWS,"%s_%s_hitRatioCompare_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#            plt.savefig(os.path.join(self.figureWS,"%s_hitRatioCompare_train.png"%(self.recType)),bbox_inches = 'tight')
#
#        print ("Hit Ratio figure created, check your output workspace")

        # plot signal power histograms by detection class
        minPower = self.train_stats_data.Power.min()//5 * 5
        maxPower = self.train_stats_data.Power.max()//5 * 5
        powerBins =np.arange(minPower,maxPower+20,10)

#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.Power.values, powerBins, density = True)
#        axs[1].hist(falses.Power.values, powerBins, density = True)
#        axs[0].set_xlabel('%s Signal Power'%(self.recType))
#        axs[0].set_title('True')
#        axs[1].set_xlabel('%s Signal Power'%(self.recType))
#        axs[1].set_title('False Positive')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#            plt.savefig(os.path.join(self.figureWS,"%s_%s_powerCompare_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#            plt.savefig(os.path.join(self.figureWS,"%s_powerCompare_train.png"%(self.recType)),bbox_inches = 'tight')
#
#        print ("Signal Power figure created, check your output Workspace")

        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,20)

#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.lagDiff.values, lagBins, density = True)
#        axs[1].hist(falses.lagDiff.values, lagBins, density = True)
#        axs[0].set_xlabel('Lag Differences')
#        axs[0].set_title('True')
#        axs[1].set_xlabel('Lag Differences')
#        axs[1].set_title('False Positive')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#            plt.savefig(os.path.join(self.figureWS,"%s_%s_lagDifferences_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#            plt.savefig(os.path.join(self.figureWS,"%s_lagDifferences_train.png"%(self.recType)),bbox_inches = 'tight')
#
        print ("Lag differences figure created, check your output Workspace")

        # Consecutive Record Length ?
        conBins =np.arange(1,12,1)

#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.conRecLength.values, conBins, density = True)
#        axs[1].hist(falses.conRecLength.values, conBins, density = True)
#        axs[0].set_xlabel('Consecutive Hit Length')
#        axs[0].set_title('True')
#        axs[1].set_xlabel('Consecutive Hit Length')
#        axs[1].set_title('False Positive')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#            plt.savefig(os.path.join(self.figureWS,"%s_%s_conRecLength_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#            plt.savefig(os.path.join(self.figureWS,"%s_conRecLength_train.png"%(self.recType)),bbox_inches = 'tight')
#
#        print ("Consecutive Hit Length figure created, check your output Workspace")

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.noiseRatio.values, noiseBins, density = True)
#        axs[1].hist(falses.noiseRatio.values, noiseBins, density = True)
#        axs[0].set_xlabel('Noise Ratio')
#        axs[0].set_title('True')
#        axs[1].set_xlabel('Noise Ratio')
#        axs[1].set_title('False Positive')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#           plt.savefig(os.path.join(self.figureWS,"%s_%s_noiseRatio_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#           plt.savefig(os.path.join(self.figureWS,"%s_noiseRatio_train.png"%(self.recType)),bbox_inches = 'tight')
#
#        print ("Noise Ratio figure created, check your output Workspace")

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

#
#        # plot fish present
#        minCount = self.train_stats_data.FishCount.min()//10 * 10
#        maxCount = self.train_stats_data.FishCount.max()//10 * 10
#        countBins =np.arange(minCount,maxCount+20,10)
#
#        plt.figure()
#        fig, axs = plt.subplots(1,2,sharey = True, sharex = True, tight_layout = True,figsize = figSize)
#        axs[0].hist(trues.FishCount.values, countBins, density = True)
#        axs[1].hist(falses.FishCount.values, countBins, density = True)
#        axs[0].set_xlabel('Fish Present')
#        axs[0].set_title('True')
#        axs[1].set_xlabel('Fish Present')
#        axs[1].set_title('Fish Present')
#        axs[0].set_ylabel('Probability Density')
#        if self.site != None:
#           plt.savefig(os.path.join(self.figureWS,"%s_%s_fishPresentCompare_train.png"%(self.recType,self.site)),bbox_inches = 'tight')
#        else:
#           plt.savefig(os.path.join(self.figureWS,"%s_fishPresentCompare_train.png"%(self.recType)),bbox_inches = 'tight')

#        print ("Fish Present Figure Created, check output workspace")

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
    def __init__(self,receiver_list,node_to_state,dbDir, input_type = 'query', initial_state_release = False, last_presence_time0 = False, cap_loc = None, rel_loc = None):
        # Import Data using sql
        '''the default input type for this function is query, however

        '''
        print ("Starting extraction of recapture data conforming to the recievers chosen")
        self.dbDir = dbDir
        conn = sqlite3.connect(dbDir)
        c = conn.cursor()
        sql = 'SELECT tblRecaptures.FreqCode, Epoch, timeStamp, Node, TagType, presence_number, overlapping, CapLoc, RelLoc, test FROM tblRecaptures LEFT JOIN tblMasterReceiver ON tblRecaptures.recID = tblMasterReceiver.recID LEFT JOIN tblMasterTag ON tblRecaptures.FreqCode = tblMasterTag.FreqCode WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        self.data = pd.read_sql_query(sql,con = conn)
        self.data['timeStamp'] = pd.to_datetime(self.data.timeStamp)
        self.data = self.data[self.data.TagType == 'Study']
        self.data = self.data[self.data.test == 1]
        self.data = self.data[self.data.overlapping == 0]
        if rel_loc is not None:
            self.data = self.data[self.data.RelLoc == rel_loc]
        if cap_loc is not None:
            self.data = self.data[self.data.CapLoc == cap_loc]
        self.cap_loc = cap_loc
        self.rel_loc = rel_loc

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

            sql = "SELECT FreqCode, TagType, RelDate, RelLoc, CapLoc FROM tblMasterTag WHERE TagType = 'Study'"
            conn = sqlite3.connect(self.dbDir)
            c = conn.cursor()



            relDat = pd.read_sql_query(sql,con = conn, parse_dates = 'RelDate')
            if self.rel_loc is not None:
                relDat = relDat[relDat.RelLoc == rel_loc]
            if self.cap_loc is not None:
                relDat = relDat[relDat.CapLoc == cap_loc]
            relDat['RelDate'] = pd.to_datetime(relDat.RelDate)
            relDat['Epoch'] = (relDat['RelDate'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            relDat.rename(columns = {'RelDate':'timeStamp'}, inplace = True)
            relDat['Node'] = np.repeat('rel',len(relDat))
            relDat['State'] = np.repeat(1,len(relDat))
            relDat['State'] = relDat['State'].astype(np.int32)
            relDat.dropna(inplace =True)
            relDat['Overlapping'] = np.zeros(len(relDat))
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
        #print(self.startTimes.head())

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
            them we still have informative data.  For this to work, we only need to
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
                        presence = presence + 1                                # oh snap new observation for new state
                        rowArr = [i,state,presence,time1,timeDelta,time0,firstObs]  # start a new row
                        row = pd.DataFrame(np.array([rowArr]),columns = columns)
                        stateTable = stateTable.append(row)                    # add the row to the state table data frame
                        time0 = j[1]['Epoch']


                print ("State Table Completed for Fish %s"%(i))

                stateTable['state'] = stateTable['state'].astype(np.int32)
                from_rec = stateTable['state'].shift(1).fillna(stateTable.iloc[0]['state']).astype(np.int32)
                to_rec = stateTable['state'].astype(np.int32)
                trans = tuple(zip(from_rec,to_rec))
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
                        #if state1 != state2 or rowIdx == maxIdx:
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
                # calculate minimum t0 by presence
                min_t0 = exp_stateTable.groupby(['presence'])['t0'].min()#.to_frame().rename({'t0':'min_t0'},inplace = True)
                min_t0 = pd.Series(min_t0, name = 'min_t0')
                min_t0 = pd.DataFrame(min_t0).reset_index()
                # join to exp_stateTable as presence_time_0
                exp_stateTable = pd.merge(left = exp_stateTable, right = min_t0, how = u'left',left_on = 'presence', right_on = 'presence')
                # subtract presence_time_0 from t0 and t1
                exp_stateTable['t0'] = exp_stateTable['t0'] -  exp_stateTable['min_t0']
                exp_stateTable['t1'] = exp_stateTable['t1'] -  exp_stateTable['min_t0']
                # drop presence_time_0 from exp_stateTable

                exp_stateTable['hour'] = pd.DatetimeIndex(exp_stateTable['time0']).hour # get the hour of the day from the current time stamp
                exp_stateTable['qDay'] = exp_stateTable.hour//6                # integer division by 6 to put the day into a quarter
                exp_stateTable['test'] = exp_stateTable.t1 - exp_stateTable.t0 # this is no longer needed, but if t1 is smaller than t0 things are screwed up
                stateTable = exp_stateTable
                del exp_stateTable
                stateTable['transition'] = tuple(zip(stateTable.startState.values.astype(int),stateTable.endState.values.astype(int))) # create transition variable, this is helpful in R
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
        self.fishTransCount = self.master_stateTable.groupby(['FreqCode','transition'])['transition'].count()
        self.fishTransCount = self.fishTransCount.to_frame(name = 'transCount')
        #self.fishTransCount.rename(columns = {'':'transCount'}, inplace = True)
        #self.fishTransCount.reset_index(inplace = True)

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
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            print (tblList)
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            max_iter = 1
            while curr_idx <= len(tblList) - 1:
                for j in tblList:
                    if int(j[-1]) >= max_iter:
                        max_iter = int(j[-1])
                        max_iter_dict[i] = j
                    curr_idx = curr_idx + 1
            curr_idx = 0
            print (max_iter_dict)
            if filtered == False and overlapping == False:
                sql = "SELECT FreqCode, Epoch, timeStamp, recID, test FROM tblClassify_%s_1"%(i)
                dat = pd.read_sql(sql,con = conn)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            elif filtered == True and overlapping == False:
                sql = "SELECT FreqCode, Epoch,  timeStamp, recID, test FROM %s WHERE test = 1"%(max_iter_dict[i])
                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            else:
                sql = "SELECT %s.FreqCode, %s.Epoch,  %s.timeStamp, %s.recID, overlapping, test FROM %s LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID WHERE test = 1"%(max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i])
                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                dat['overlapping'].fillna(0,inplace = True)
                dat = dat[dat.overlapping == 0]
                data = data.append(dat)
                #print data.head()
                #fuck
                print ("Data from reciever %s imported"%(i))
                del sql
        data.drop_duplicates(keep = 'first', inplace = True)
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
        c = conn.cursor()
        for i in receivers:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            max_iter = 1
            while curr_idx <= len(tblList) - 1:
                for j in tblList:
                    if int(j[-1]) >= max_iter:
                        max_iter = int(j[-1])
                        max_iter_dict[i] = j
                    curr_idx = curr_idx + 1
            curr_idx = 0

            # once we have a hash table of receiver to max classification, extract the classification dataset
            for j in max_iter_dict:
                sql = "SELECT FreqCode, Epoch, recID, test FROM %s"%(max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn)                                 # get data for this receiver
                dat = dat[(dat.test == 1)] # query
                dat.drop(['test'],axis = 1, inplace = True)
                data = data.append(dat)

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
            knotDF = pd.DataFrame.from_dict(knot_ssr, orient = 'index').reset_index()
            knotDF.rename(columns = {'index':'knots',0:'SSR'}, inplace = True)
            print(knotDF)
            if len(knotDF) == 0:
                minKnot = 7200
                return minKnot
            else:
                min_knot_idx = knotDF['SSR'].idxmin()
                minKnot = knotDF.iloc[min_knot_idx,0]
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
        c = conn.cursor()
        for i in receivers:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            max_iter = 1
            while curr_idx <= len(tblList) - 1:
                for j in tblList:
                    if int(j[-1]) >= max_iter:
                        max_iter = int(j[-1])
                        max_iter_dict[i] = j
                    curr_idx = curr_idx + 1
            curr_idx = 0
            del j

            # once we have a hash table of receiver to max classification, extract the classification dataset
            for j in max_iter_dict:
                sql = "SELECT FreqCode, Epoch, recID, test FROM %s WHERE FreqCode == '%s'"%(max_iter_dict[j],fish)
                dat = pd.read_sql(sql, con = conn)                                 # get data for this receiver
                dat = dat[(dat.test == 1)] # query
                dat.drop(['test'],axis = 1, inplace = True)
                presence = presence.append(dat)

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
    '''Python class  to reduce redundant dections at overlappin receivers.
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
        Edges always in format of [(from,to)] or [(outer,inner)] or [(child,parent)]'''

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
            for j in node_recs:
                c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                tbls = c.fetchall()
                tblList = []
                for k in tbls:
                    if j in k[0]:
                        tblList.append(k[0])
                del k
                # iterate over the receivers to find the final classification (aka the largest _n)
                max_iter_dict = {} # receiver:max iter
                curr_idx = 0
                max_iter = 1
                while curr_idx <= len(tblList) - 1:
                    for k in tblList:
                        if int(k[-1]) >= max_iter:
                            max_iter = int(k[-1])
                            max_iter_dict[j] = k
                        curr_idx = curr_idx + 1
                curr_idx = 0
                #print (tblList)
                #print (max_iter_dict)
                # once we have a hash table of receiver to max classification, extract the classification dataset
                for k in max_iter_dict:

                    #print "Start selecting classified and presence data that matches the current receiver (%s)"%(j)
                    presence_sql = "SELECT * FROM tblPresence WHERE recID = '%s'"%(j)
                    presenceDat = pd.read_sql(presence_sql, con = conn)
                    recap_sql = "SELECT FreqCode, Epoch, recID, test from %s"%(max_iter_dict[j])
                    recapDat = pd.read_sql(recap_sql, con = conn)
                    recapDat = recapDat[(recapDat.test == 1)]

                    recapDat.drop(labels = ['test'], axis = 1, inplace = True)
                    # now that we have data, we need to summarize it, use group by to get min ans max epoch by freq code, recID and presence_number
                    pres_data = pres_data.append(presenceDat)
                    recap_data = recap_data.append(recapDat)
                    del presence_sql, recap_sql

            dat = pres_data.groupby(['FreqCode','presence_number'])['Epoch'].agg(['min','max'])
            dat.reset_index(inplace = True, drop = False)
            dat.rename(columns = {'min':'min_Epoch','max':'max_Epoch'},inplace = True)
            self.node_pres_dict[i] = dat
            self.node_recap_dict[i] = recap_data
            del pres_data, recap_data, dat, recSQL, receivers, node_recs
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
    # create function that we can vectorize over list of epochs (i)
    def overlap_check(i, min_epoch, max_epoch):
        return np.logical_and(min_epoch >= i, max_epoch < i).any()
    nodeDat = overlap.node_recap_dict[overlap.curr_node]
    fishes = nodeDat.FreqCode.unique()
    for i in fishes:
        overlap.fish = i
        #print "Let's start sifting through fish %s at node %s"%(i, overlap.curr_node)
        children = overlap.G.succ[overlap.curr_node]
        fishDat = nodeDat[nodeDat.FreqCode == i]
        fishDat['overlapping'] = np.zeros(len(fishDat))
        fishDat['successor'] = np.repeat('',len(fishDat))
        fishDat.set_index('Epoch', inplace = True, drop = False)
        if len(children) > 0:                                            # if there is no child node, who cares? there is no overlapping detections, we are at the middle doll
            for j in children:
                child_dat = overlap.node_pres_dict[j]
                if len(child_dat) > 0:
                    child_dat = child_dat[child_dat.FreqCode == i]
                    if len(child_dat) > 0:
                        min_epochs = child_dat.min_Epoch.values
                        max_epochs = child_dat.max_Epoch.values
                        for k in fishDat.Epoch.values:                          # for every row in the fish data
                            print ("Fish %s epoch %s overlap check at child %s"%(i,k,j))
                            if np.logical_and(min_epochs <= k, max_epochs >k).any(): # if the current epoch is within a presence at a child receiver
                                print ("Overlap Found, at %s fish %s was recaptured at both %s and %s"%(k,i,overlap.curr_node,j))
                                fishDat.at[k,'overlapping'] = 1
                                fishDat.at[k,'successor'] = j
        fishDat.reset_index(inplace = True, drop = True)
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



def the_little_merge(outputWS,projectDB, hitRatio_Filter = False, pre_release_Filter = False, rec_list = None, con_rec_filter = None):
    '''function takes classified data, merges across sites and then joins presence
    and overlapping data into one big file for model building.'''
    conn = sqlite3.connect(projectDB)                                              # connect to the database
    if rec_list != None:
        recSQL = "SELECT * FROM tblMasterReceiver WHERE recID = '%s'"%(rec_list[0])
        for i in rec_list[1:]:
            recSQL = recSQL + " OR recID = '%s'"%(i)
    else:
        recSQL = "SELECT * FROM tblMasterReceiver"                                 # SQL code to import data from this node
    receivers = pd.read_sql(recSQL,con = conn)                                 # import data
    receivers = receivers.recID.unique()                                       # get the unique receivers associated with this node
    recapdata = pd.DataFrame(columns = ['FreqCode','Epoch','recID','timeStamp','fileName'])                # set up an empty data frame
    c = conn.cursor()
    c.close()
    for i in receivers:                                                            # for every receiver
        conn = sqlite3.connect(projectDB)                                              # connect to the database
        c = conn.cursor()
        print ("Start selecting and merging data for receiver %s"%(i))
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tbls = c.fetchall()
        tblList = []
        for j in tbls:
            if i in j[0]:
                tblList.append(j[0])
        del j
        # iterate over the receivers to find the final classification (aka the largest _n)
        max_iter_dict = {} # receiver:max iter
        curr_idx = 0
        max_iter = 1
        while curr_idx <= len(tblList) - 1:
            for j in tblList:
                if int(j[-1]) >= max_iter:
                    max_iter = int(j[-1])
                    max_iter_dict[i] = j
                curr_idx = curr_idx + 1
        curr_idx = 0

        # once we have a hash table of receiver to max classification, extract the classification dataset
        for j in max_iter_dict:

            cursor = conn.execute('select * from %s'%(max_iter_dict[j]))
            names = [description[0] for description in cursor.description]


            try:
                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,presence_number, overlapping, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,presence_number, overlapping,test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                dat['overlapping'].fillna(0,inplace = True)
                dat = dat[dat.overlapping == 0]


            except:

                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])

                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver


            dat = dat[dat.test == 1]
            dat['RelDate'] = pd.to_datetime(dat.RelDate)
            dat['timeStamp'] = pd.to_datetime(dat.timeStamp)
            if hitRatio_Filter == True:
                dat = dat[(dat.hitRatio_A > 0.10)]# | (dat.hitRatio_M > 0.10)]
            if con_rec_filter != None:
                dat = dat[(dat.conRecLength_A >= con_rec_filter) | (dat.conRecLength_M >= con_rec_filter)]
            if pre_release_Filter == True:
                dat = dat[(dat.timeStamp >= dat.RelDate)]
            recapdata = recapdata.append(dat)
            del dat
    c.close()

    recapdata.drop_duplicates(keep = 'first', inplace = True)
    return recapdata











def the_big_merge(outputWS,projectDB, hitRatio_Filter = False, pre_release_Filter = False, rec_list = None, con_rec_filter = None):
    '''function takes classified data, merges across sites and then joins presence
    and overlapping data into one big file for model building.'''
    conn = sqlite3.connect(projectDB)                                              # connect to the database
    if rec_list != None:
        recSQL = "SELECT * FROM tblMasterReceiver WHERE recID = '%s'"%(rec_list[0])
        for i in rec_list[1:]:
            recSQL = recSQL + " OR recID = '%s'"%(i)
    else:
        recSQL = "SELECT * FROM tblMasterReceiver"                                 # SQL code to import data from this node
    receivers = pd.read_sql(recSQL,con = conn)                                 # import data
    receivers = receivers.recID.unique()                                       # get the unique receivers associated with this node
    recapdata = pd.DataFrame(columns = ['FreqCode','Epoch','recID','timeStamp','Power','fileName'])                # set up an empty data frame
    c = conn.cursor()

    bouts = False
    for i in receivers:                                                            # for every receiver

        print ("Start selecting and merging data for receiver %s"%(i))
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tbls = c.fetchall()
        tblList = []
        for j in tbls:
            if i in j[0]:
                tblList.append(j[0])
            if j[0] == 'tblPresence' or j[0]== 'tblOverlap':
                bouts = True
        del j
        # iterate over the receivers to find the final classification (aka the largest _n)
        max_iter_dict = {} # receiver:max iter
        curr_idx = 0
        max_iter = 1
        while curr_idx <= len(tblList) - 1:
            for j in tblList:
                if int(j[-1]) >= max_iter:
                    max_iter = int(j[-1])
                    max_iter_dict[i] = j
                curr_idx = curr_idx + 1
        curr_idx = 0

        # once we have a hash table of receiver to max classification, extract the classification dataset
        for j in max_iter_dict:

            cursor = conn.execute('select * from %s'%(max_iter_dict[j]))
            names = [description[0] for description in cursor.description]


            if bouts == True:
                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,Power,presence_number, overlapping, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode
                    LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID
                    LEFT JOIN tblPresence ON %s.FreqCode = tblPresence.FreqCode AND %s.Epoch = tblPresence.Epoch AND %s.recID = tblPresence.recID'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,Power,presence_number, overlapping,test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode
                    LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID
                    LEFT JOIN tblPresence ON %s.FreqCode = tblPresence.FreqCode AND %s.Epoch = tblPresence.Epoch AND %s.recID = tblPresence.recID'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                dat['overlapping'].fillna(0,inplace = True)
                dat = dat[dat.overlapping == 0]


            else:

                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,Power, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])

                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,Power, test, RelDate, fileName
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver


            dat = dat[dat.test == 1]
            dat['RelDate'] = pd.to_datetime(dat.RelDate)
            dat['timeStamp'] = pd.to_datetime(dat.timeStamp)
            if hitRatio_Filter == True:
                dat = dat[(dat.hitRatio_A > 0.10)]# | (dat.hitRatio_M > 0.10)]
            if con_rec_filter != None:
                dat = dat[(dat.conRecLength_A >= con_rec_filter) | (dat.conRecLength_M >= con_rec_filter)]
            if pre_release_Filter == True:
                dat = dat[(dat.timeStamp >= dat.RelDate)]
            recapdata = recapdata.append(dat)
            del dat
    c.close()

    recapdata.drop_duplicates(keep = 'first', inplace = True)
    return recapdata


class cjs_data_prep():
    '''Class creates input files for Cormack Jolly Seber modeling in MARK'''
    def __init__(self,receiver_to_recap, dbDir, input_type = 'query', rel_loc = None, cap_loc = None, initial_recap_release = False):
        # Import Data using sql
        print ("Starting extraction of recapture data conforming to the recievers chosen")
        conn = sqlite3.connect(dbDir)
        c = conn.cursor()
        self.data = pd.DataFrame()                                             # create an empty dataframe
        self.rel_loc = rel_loc
        self.cap_loc = cap_loc

        receiver_list = sorted(list(receiver_to_recap.keys()))
        sql = 'SELECT tblRecaptures.FreqCode, Epoch, timeStamp, tblRecaptures.recID, TagType, overlapping, RelLoc, CapLoc, test FROM tblRecaptures LEFT JOIN tblMasterReceiver ON tblRecaptures.recID = tblMasterReceiver.recID LEFT JOIN tblMasterTag ON tblRecaptures.FreqCode = tblMasterTag.FreqCode WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        recap_data = pd.read_sql_query(sql,con = conn,coerce_float  = True)
        #self.data['timeStamp'] = self.data['timeStamp'].to_datetime()
        recap_data = recap_data[recap_data.TagType == 'Study']
        recap_data = recap_data[recap_data.overlapping == 0.0]
        recap_data = recap_data[recap_data.test == 1.0]
        if rel_loc != None:
            recap_data = recap_data[recap_data.RelLoc == rel_loc]
        if cap_loc != None:
            recap_data = recap_data[recap_data.CapLoc == cap_loc]

        self.data = self.data.append(recap_data)
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

        #  extract the release cohort if required
        if initial_recap_release == True:
            sql = "SELECT FreqCode, TagType, RelDate, RelLoc, CapLoc FROM tblMasterTag WHERE TagType = 'Study'"
            relDat = pd.read_sql_query(sql,con = conn, parse_dates = 'RelDate')
            if self.rel_loc is not None:
                relDat = relDat[relDat.RelLoc == rel_loc]
            if self.cap_loc is not None:
                relDat = relDat[relDat.CapLoc == cap_loc]
            relDat['RelDate'] = pd.to_datetime(relDat.RelDate)
            relDat['Epoch'] = (relDat['RelDate'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            relDat.rename(columns = {'RelDate':'timeStamp'}, inplace = True)
            relDat['RecapOccasion'] = np.repeat('R00',len(relDat))
            #relDat.dropna(inplace =True)
            relDat['overlapping'] = np.zeros(len(relDat))
            self.data = self.data.append(relDat)
        print (self.data.head())
        c.close()
        # Identify first recapture times
        self.startTimes = self.data[self.data.RecapOccasion == "R00"].groupby(['FreqCode'])['Epoch'].min().to_frame()
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
        self.inp = cross_tab
        # Check your work
        print (cross_tab.head(100))
        cross_tab.to_csv(os.path.join(outputWS,'%s_cjs.csv'%(modelName)))

class lrdr_data_prep():
    '''Class creates input files for Live Recapture Dead Recovery modeling in MARK'''
    def __init__(self,receiver_to_recap,mobile_to_recap, dbDir, input_type = 'query', rel_loc = None, cap_loc = None, initial_recap_release = False, time_limit = None):

        # Import Data using sql
        '''the default input type for this function is query, however'''
        print ("Starting extraction of recapture data related to the recievers chosen")
        conn = sqlite3.connect(dbDir)
        c = conn.cursor()
        self.rel_loc = rel_loc
        self.cap_loc = cap_loc
        # get live recapture data
        receiver_list = list(receiver_to_recap.keys())
        sql = 'SELECT tblRecaptures.FreqCode, Epoch, timeStamp, tblRecaptures.recID, TagType, overlapping, RelLoc, CapLoc, test FROM tblRecaptures LEFT JOIN tblMasterReceiver ON tblRecaptures.recID = tblMasterReceiver.recID LEFT JOIN tblMasterTag ON tblRecaptures.FreqCode = tblMasterTag.FreqCode WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        self.live_recap_data = pd.read_sql_query(sql,con = conn,coerce_float  = True)
        #self.live_recap_data['timeStamp'] = self.live_recap_data['timeStamp'].to_datetime()
        self.live_recap_data = self.live_recap_data[self.live_recap_data.TagType == 'Study']
        self.live_recap_data = self.live_recap_data[self.live_recap_data.overlapping == 0.0]
        self.live_recap_data = self.live_recap_data[self.live_recap_data.test == 1.0]
        if rel_loc != None:
            self.live_recap_data = self.live_recap_data[self.live_recap_data.RelLoc == rel_loc]
        if cap_loc != None:
            self.live_recap_data = self.live_recap_data[self.live_recap_data.CapLoc == cap_loc]

        # get dead mobile tracking data
        sql = 'SELECT * FROM tblMobileTracking'
        self.dead_recover_data = pd.read_sql_query(sql,con = conn, coerce_float = True)
        self.dead_recover_data = self.dead_recover_data[(self.dead_recover_data.Alive == '1') | (self.dead_recover_data.Alive == '0')]
        self.dead_recover_data['Alive'] = self.dead_recover_data.Alive.astype(np.int32)
        self.dead_recover_data.loc[self.dead_recover_data.Alive == 1,'Dead'] = 0
        self.dead_recover_data.loc[self.dead_recover_data.Alive == 0,'Dead'] = 1
        self.dead_recover_data['Dead'] = self.dead_recover_data.Dead.astype(np.int32)
        self.dead_recover_data['DateTime'] = pd.to_datetime(self.dead_recover_data.DateTime)
        self.dead_recover_data['Epoch'] = (self.dead_recover_data['DateTime'] - datetime.datetime(1970,1,1)).dt.total_seconds()

        print ("Starting receiver to recapture occasion classification, with %s records, this takes time"%(len(self.live_recap_data)))
        # create function that maps live recapture occasion to row given current receiver and apply
        def live_recap_class(row,receiver_to_recap):
            currRec = row['recID']
            recapOcc = receiver_to_recap[currRec]
            return recapOcc
        self.live_recap_data['RecapOccasion'] = self.live_recap_data.apply(live_recap_class,axis = 1, args = (receiver_to_recap,))

        # create function that maps dead recover occasion to row given current mobile tracking reach and apply
        def dead_recover_class(row,mobile_to_recap):
            currRec = row['mReach']
            recapOcc = mobile_to_recap[currRec]
            return recapOcc
        self.dead_recover_data['RecapOccasion'] = self.dead_recover_data.apply(dead_recover_class,axis = 1, args = (mobile_to_recap,))
        # get initial release states if requirec
        if initial_recap_release == True:
            sql = "SELECT FreqCode, TagType, RelDate, RelLoc, CapLoc FROM tblMasterTag WHERE TagType = 'Study'"
            relDat = pd.read_sql_query(sql,con = conn, parse_dates = 'RelDate')
            if self.rel_loc is not None:
                relDat = relDat[relDat.RelLoc == rel_loc]
            if self.cap_loc is not None:
                relDat = relDat[relDat.CapLoc == cap_loc]
            relDat['RelDate'] = pd.to_datetime(relDat.RelDate)
            relDat['Epoch'] = (relDat['RelDate'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            relDat.rename(columns = {'RelDate':'timeStamp'}, inplace = True)
            relDat['RecapOccasion'] = np.repeat('R00',len(relDat))
            relDat.dropna(inplace =True)
            relDat['overlapping'] = np.zeros(len(relDat))
            self.live_recap_data = self.live_recap_data.append(relDat)

        c.close()
        print ("Finished sql")

        # Identify first recapture times
        self.startTimes = self.live_recap_data[self.live_recap_data.RecapOccasion == "R00"].groupby(['FreqCode'])['Epoch'].min().to_frame()
        self.startTimes.reset_index(drop = False, inplace = True)
        self.startTimes.Epoch = self.startTimes.Epoch.astype(np.int32)
        self.startTimes.rename(columns = {'Epoch':'FirstRecapture'},inplace = True)
        for fish in self.live_recap_data.FreqCode.unique():
            if fish not in self.startTimes.FreqCode.unique():                  # fish never made it to the initial state, remove - we only care about movements from the initial sstate - this is a competing risks model
                self.live_recap_data = self.live_recap_data[self.live_recap_data.FreqCode != fish]

        # identify unique fish
        self.fish = self.live_recap_data.FreqCode.unique()                                # identify unique fish to loop through

        # store some values for safe keeping
        self.mobile_to_recap = mobile_to_recap
        self.receiver_to_recap = receiver_to_recap

        if time_limit != None:
            '''live recapture dead recovery is most useful for understanding the
            true survival of fish after they pass through infrastructure.

            often times performance criteria dictates a certain percentage of
            survival after a certain amount of time.

            If there is a time limit imposed on this performance criteria,
            enter the number of hours.  This function will first find the last
            detection in the initial state - which should be before the infrastructure -
            and will be used as passage time.  Then, this function will remove
            all detections greater than the time limit.

            Mobile tracking may only happen once per week - therefore the mobile
            tracking time limit is extended to 1 week regardless

            The time limit is expressed in decimal hours.
            '''
            # get last recapture in state 00 for all fish
            last_rec_state_00 = self.live_recap_data[self.live_recap_data.RecapOccasion == "R00"].groupby(['FreqCode'])['Epoch'].max().to_frame()
            last_rec_state_00.reset_index(drop = False, inplace = True)
            last_rec_state_00.Epoch = last_rec_state_00.Epoch.astype(np.int32)
            last_rec_state_00.rename(columns = {'Epoch':'LastRecapture_00'},inplace = True)
            print (last_rec_state_00.head())
            # join this dataframe to the live recaptures and dead recovery data
            self.live_recap_data = pd.merge(self.live_recap_data, last_rec_state_00, on = 'FreqCode', how = 'left')#, lsuffix = 'x', rsuffix = 'y')
            self.dead_recover_data = pd.merge(self.dead_recover_data, last_rec_state_00, on = 'FreqCode', how = 'left')#, lsuffix = 'x', rsuffix = 'y')
            # calculate time since last recpature in state 00
            self.live_recap_data['duration'] = self.live_recap_data.Epoch - self.live_recap_data.LastRecapture_00
            self.dead_recover_data['duration'] = self.dead_recover_data.Epoch - self.dead_recover_data.LastRecapture_00
            # convert time limit to seconds
            live_recap_time_limit = time_limit * 3600
            dead_recap_time_limit = 7 * 24 * 3600 # we only mobile track once per week - so let's increase the dead recovery time to be within a week of passage
            # extract only good data below the time limit
            self.live_recap_data = self.live_recap_data[self.live_recap_data.duration <= live_recap_time_limit]
            self.dead_recover_data = self.dead_recover_data[self.dead_recover_data.duration <= dead_recap_time_limit]

    def input_file(self,modelName, outputWS):
        # create cross tabulated data frame of live recapture data
        #Step 1: Create cross tabulated data frame with FreqCode as row index and recap occasion as column.  Identify the min epoch'''
        self.live_recap_cross_tab = pd.pivot_table(self.live_recap_data, values = 'Epoch', index = 'FreqCode', columns = 'RecapOccasion', aggfunc = 'min')
        #Step 2: Fill in those nan values with 0, we can't perform logic on nothing!'''
        self.live_recap_cross_tab.fillna(value = 0, inplace = True)
        #Step 3: Map a simply if else statement, if value > 0,1 else 0'''
        self.live_recap_cross_tab = self.live_recap_cross_tab.applymap(lambda x:1 if x > 0 else 0)
        print(self.live_recap_cross_tab.head())

        # create cross tabulated data frame of dead recovery data
        #Step 1: Create cross tabulated data frame with FreqCode as row index and recap occasion as column.  Identify the min epoch'''
        self.dead_recover_cross_tab = pd.pivot_table(self.dead_recover_data, values = 'Dead', index = 'FreqCode', columns = 'RecapOccasion', aggfunc = 'max')
        #Step 2: Fill in those nan values with 0, we can't perform logic on nothing!'''
        self.dead_recover_cross_tab.fillna(value = 0, inplace = True)
        #Step 3: Map a simply if else statement, if value > 0,1 else 0'''
        self.dead_recover_cross_tab = self.dead_recover_cross_tab.applymap(lambda x:1 if x > 0 else 0)
        print (self.dead_recover_cross_tab.head())

        # join on FreqCode
        self.inp = self.live_recap_cross_tab.join(self.dead_recover_cross_tab, on = 'FreqCode', how = 'left', lsuffix = 'x', rsuffix = 'y')
        self.inp.fillna(value = 0, inplace = True)

        # order by recapture occasion - If we start at R0, that comes after FreqCode - fuck yeah
        self.inp = self.inp.reindex(sorted(self.inp.columns), axis = 1)

        print (self.inp.head())

        # perform that LDLD logic - no alive fish after recovered dead

        # create dictionary describing inp input column and index
        inp_cols = list(self.inp.columns)
        idx = 0
        inp_col_dict = {}
        col_inp_dict = {}
        for i in inp_cols:
            inp_col_dict[i] = idx
            col_inp_dict[idx] = i
            idx = idx + 1

        col_length = len(inp_col_dict)

        # create dictionary describing mobile key index
        keys = sorted(list(self.mobile_to_recap.keys()))
        idx = 0
        mkey_idx_dict = {}
        for i in keys:
            mkey_idx_dict[i] = idx
            idx = idx + 1


        # loop through mobile reaches to find dead fish
        for i in sorted(list(self.mobile_to_recap.keys())):
            occ = self.mobile_to_recap[i]
            for index, row in self.inp.iterrows():
                row_idx = inp_col_dict[occ]
                dead = row[occ]
                if dead == 1:
                    if i != keys[-1]:
                        # this fish is dead, iterate over column indexes and write 0 there - can't be recovered alive or dead anywhere else
                        for j in np.arange(row_idx + 1,col_length,1):
                            self.inp.at[index,col_inp_dict[j]] = 0
                            row.iloc[j] = 0
                        print (row)




        print (self.inp.head())
        # Check your work
        self.inp.to_csv(os.path.join(outputWS,'%s_lrdr.csv'%(modelName)))

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














