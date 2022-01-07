# -*- coding: utf-8 -*-

'''
Module contains all of the functions to create a radio telemetry project.'''

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
    function does not check for inconsistencies with data structures.

    dataFrame = pandas dataframe imported from your structured file.
    dbName = full directory path to project database
    tblName = the name of the data you can import to.  If you are brave, import to
    tblRaw, but really this is meant for tblMasterTag and tblMasterReceiver'''
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    dataFrame.to_sql(tblName,con = conn,index = False, if_exists = 'append')
    conn.commit()
    c.close()

def orionImport(fileName,rxfile,dbName,recName, scanTime = 1, channels = 1, ant_to_rec_dict = None):
    '''Function imports raw Sigma Eight orion data.

    Text parser uses simple column fixed column widths.

    '''
    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()
    study_tags = pd.read_sql('SELECT FreqCode, TagType FROM tblMasterTag',con = conn)
    study_tags = study_tags[study_tags.TagType == 'Study'].FreqCode.values

    recType = 'orion'

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


def lotek_import(fileName,rxfile,dbName,recName,ant_to_rec_dict):
    ''' function imports raw lotek data, reads header data to find receiver parameters
    and automatically locates raw telemetry data.  Import procedure works with
    standardized project database. Database must be created before function can be run'''

    '''to do: in future iterations create a check for project database, if project
    data base does not exist, throw an exception

    inputs:
    fileName = name of raw telemetry data file with full directory and extenstion
    dbName = name of project database with full directory and extension
    recName = official receiver name'''


    recType = 'lotek'
    # create empty dictionary to hold Lotek header data indexed by line number - to be imported to Pandas dataframe
    headerDat = {}
    # create empty array to hold line indices
    lineCounter = []
    # generate a list of header lines - contains all data we need to write to project set up database
    lineList = []
    # open file passed to function
    o_file = open(fileName, encoding='utf-8')
    # start line counter
    counter = 0
    # read first line in file
    line = o_file.readline()[:-1]
    # append the current line counter to the counter array
    lineCounter.append(counter)
    # append the current line of header data to the line list
    lineList.append(line)

    if line == "SRX800 / 800D Information:":
        # find where data begins and header data ends
        with o_file as f:
            for line in f:
                # if this current line signifies the start of the data stream, the data starts three rows down from this
                if "** Data Segment **" in line:
                    counter = counter + 1
                    dataRow = counter + 5
                    # break the loop, we have reached our stop point
                    break
                # else we are still reading header data increase the line counter by 1
                else:
                    counter = counter + 1
                    # append the line counter to the count array
                    lineCounter.append(counter)
                    # append line of data to the data array
                    lineList.append(line)

        # add count array to dictionary with field name 'idx' as key
        headerDat['idx'] = lineCounter
        # add data line array to dictionary with field name 'line' as key
        headerDat['line'] = lineList
        # create pandas dataframe of header data indexed by row number
        headerDF = pd.DataFrame.from_dict(headerDat)
        headerDF.set_index('idx',inplace = True)

        # find scan time
        for row in headerDF.iterrows():
            # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document
            if 'Scan Time' in row[1][0]:
                # get the number value from the row
                scanTimeStr = row[1][0][-7:-1]
                # split the string
                scanTimeSplit = scanTimeStr.split(':')
                # convert the scan time string to float
                scanTime = float(scanTimeSplit[1])
                # stop the loop, we have extracted scan time
                break
        del row, scanTimeStr, scanTimeSplit

        # Find Master Firmware
        firmware = ''
        for row in headerDF.iterrows():
            # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document
            if 'Master Firmware:' in row[1][0]:
                # get the number value from the row
                firmwarestr = row[1][0]
                # split the string
                firmware_split = firmwarestr.split(' ')
                # convert the scan time string to float
                firmware = str(firmware_split[-1]).lstrip().rstrip()
                # stop the loop, we have extracted scan time
                break
        del row, firmwarestr, firmware_split

        # find number of channels and create channel dictionary
        scanChan = []
        channelDict = {}
        counter = 0
        rows = headerDF.iterrows()
        # for every row
        for row in rows:
            # if the first 18 characters say what that says
            if 'Active scan_table:' in row[1][0]:
                # channel dictionary data starts two rows from here
                idx0 = counter + 2
                # while the next row isn't empty
                while next(rows)[1][0] != '\n':
                    # increase the counter
                    counter = counter + 1
                # get index of last data row
                idx1 = counter + 1
                # when the row is empty we have reached the end of channels,
                break
             # else, increase the counter by 1
            else:
                counter = counter + 1
        del row, rows

        # extract channel dictionary data using rows identified earlier
        channelDat = headerDF.iloc[idx0:idx1]
        for row in channelDat.iterrows():
            dat = row[1][0]
            channel = str(dat[0:4]).lstrip()
            frequency = dat[10:17]
            channelDict[channel] = frequency
            # extract that channel ID from the data row and append to array
            scanChan.append(channel)
        channels = len(scanChan)

        # connect to database, read telemetry data using extracted parameters
        conn = sqlite3.connect(dbName, timeout=30.0)
        c = conn.cursor()
        study_tags = pd.read_sql('SELECT FreqCode, TagType FROM tblMasterTag',con = conn)
        study_tags = study_tags[study_tags.TagType == 'Study'].FreqCode.values

        # with our data row, extract information using pandas fwf import procedure
        if firmware == '9.12.5':
            telemDat = pd.read_fwf(fileName,
                                   colspecs = [(0,8),(8,23),(23,33),(33,41),(41,56),(56,64)],
                                   names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                                   skiprows = dataRow,
                                   dtype = {'ChannelID':str,'TagID':str,'Antenna':str})
        else:
            telemDat = pd.read_fwf(fileName,
                                   colspecs = [(0,8),(8,18),(18,28),(28,36),(36,51),(51,59)],
                                   names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                                   skiprows = dataRow,
                                   dtype = {'ChannelID':str,'TagID':str,'Antenna':str})

        telemDat.dropna(inplace = True)

        telemDat['fileName'] = np.repeat(rxfile,len(telemDat))                   # Adding the filename into the dataset...drop the path (note this may cause confusion because above we use filename with path.  Decide what to do and fix)

        def id_to_freq(row,channelDict):
            if row[2] in channelDict:
                return channelDict[row[2]]
            else:
                return '888'
        if len(telemDat) > 0:

            for ant in ant_to_rec_dict:
                site = ant_to_rec_dict[ant]
                telemDat_sub = telemDat[telemDat.Antenna == ant]
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
                c.close()

    else:
        lotek400 = False
        # find where data begins and header data ends
        with o_file as f:
            for line in f:
                if "********************************* Data Segment *********************************" in line:
                    # if this current line signifies the start of the data stream
                    counter = counter + 1
                    # the data starts 5 rows down from this
                    dataRow = counter + 5
                    # break the loop, we have reached our stop point
                    break
                elif line[0:14] == "Code_log data:":
                    # you have a lotek400
                    counter = counter + 1
                    dataRow = counter + 3
                    lotek400 = True
                    break
                else:
                    # if we are still reading header data increase the line counter by 1
                    counter = counter + 1
                    # append the line counter to the count array
                    lineCounter.append(counter)
                    # append line of data to the data array
                    lineList.append(line)
        # add count array to dictionary with field name 'idx' as key
        headerDat['idx'] = lineCounter
        # add data line array to dictionary with field name 'line' as key
        headerDat['line'] = lineList
        # create pandas dataframe of header data indexed by row number
        headerDF = pd.DataFrame.from_dict(headerDat)
        headerDF.set_index('idx',inplace = True)

        # find scan time
        # for every header data row
        for row in headerDF.iterrows():
            if 'scan time' in row[1][0] or 'Scan time' in row[1][0]:
                # get the number value from the row
                scanTimeStr = row[1][0][-7:-1]
                # split the string
                scanTimeSplit = scanTimeStr.split(':')
                # convert the scan time string to float
                scanTime = float(scanTimeSplit[1])
                # stop that loop, we done
                break
        del row
        if lotek400 == True:
            # determine if researchers used CRTO and extract
            for row in headerDF.iterrows():
                if 'Continuous record time-out' in row[1][0]:
                    crto_split = row[1][0].split('=')
                    crto_time = crto_split[1]
                    if crto_time == '0:00.00':
                        crto = False
                    else:
                        crto = True
                    break
            del row

        # find number of channels and create channel dictionary
        # create empty array of channel ID's
        scanChan = []
        # create empty channel ID: frequency dictionary
        channelDict = {}
        # create counter
        counter = 0
        # create row iterator
        rows = headerDF.iterrows()
        for row in rows:
            # if the first 18 characters say Active scan_table
            if 'Active scan_table:' in row[1][0]:
                # channel dictionary data starts two rows from here
                idx0 = counter + 2
                # while the next row isn't empty
                while next(rows)[1][0] != '\n':
                    # increase the counter
                    counter = counter + 1
                # get index of last data row
                idx1 = counter + 1
                break
            else:
                counter = counter + 1
        del row, rows

        # extract channel dictionary data using rows identified earlier
        channelDat = headerDF.iloc[idx0:idx1]
        for row in channelDat.iterrows():
            # extract that channel ID from the data row and append to array
            dat = row[1][0]
            channel = int(dat[0:4])
            frequency = dat[10:17]
            channelDict[channel] = frequency
            scanChan.append(channel)
        channels = len(scanChan)

        # connect to the project database
        conn = sqlite3.connect(dbName, timeout=30.0)
        c = conn.cursor()
        study_tags = pd.read_sql('''SELECT FreqCode
                                 FROM tblMasterTag
                                 WHERE TagType == "Study" OR TagType == "Beacon"''',
                                 con = conn).FreqCode.values

        def id_to_freq(row,channelDict):
            channel = row['ChannelID']
            if np.int(channel) in channelDict:
                return channelDict[np.int(channel)]
            else:
                return '888'

        # with our data row, extract information using pandas fwf import procedure
        if lotek400 == False:
            telemDat = pd.read_fwf(os.path.join(fileName),
                                   colspecs = [(0,5),(5,14),(14,23),(23,31),(31,46),(46,54)],
                                   names = ['DayNumber','Time','ChannelID','TagID','Antenna','Power'],
                                   skiprows = dataRow)
            # remove last two rows, Lotek adds garbage at the end
            telemDat = telemDat.iloc[:-2]
            telemDat.dropna(inplace = True)
            if len(telemDat) > 0:
                telemDat['Frequency'] = telemDat.apply(id_to_freq, axis = 1, args = (channelDict,))
                telemDat = telemDat[telemDat.Frequency != '888']
                telemDat = telemDat[telemDat.TagID != 999]
                telemDat['FreqCode'] = telemDat['Frequency'].astype(str) + ' ' + telemDat['TagID'].astype(int).astype(str)
                telemDat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telemDat))
                telemDat['Date'] = telemDat['day0'] + pd.to_timedelta(telemDat['DayNumber'].astype(int), unit='d')
                telemDat['Date'] = telemDat.Date.astype('str')
                telemDat['timeStamp'] = pd.to_datetime(telemDat['Date'] + ' ' + telemDat['Time'])
                telemDat.drop(['day0','DayNumber'],axis = 1, inplace = True)
                telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                telemDat.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                telemDat['fileName'] = np.repeat(rxfile,len(telemDat))
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
            telemDat = pd.read_fwf(os.path.join(fileName),
                                   colspecs = [(0,6),(6,14),(14,22),(22,27),(27,35),(35,41),(41,48),(48,56),(56,67),(67,80)],
                                   names = ['DayNumber_Start','StartTime','ChannelID','TagID','Antenna','Power','Data','Events','DayNumber_End','EndTime'],
                                   skiprows = dataRow)
            telemDat.dropna(inplace = True)

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
                telemDat['timeStamp'] = pd.to_datetime(telemDat['Date_Start'] + ' ' + telemDat['StartTime'])
                telemDat['time_end'] = pd.to_datetime(telemDat['Date_End'] + ' ' + telemDat['EndTime'])
                telemDat.drop(['day0','DayNumber_Start','DayNumber_End'],axis = 1, inplace = True)
                telemDat['duration'] = (telemDat.time_end - telemDat.timeStamp).astype('timedelta64[s]')
                if crto == True:
                    telemDat = telemDat[telemDat.Events > 1]
                telemDat['events_per_duration'] = telemDat.Events / telemDat.duration
                telemDat['Epoch'] = (telemDat['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                telemDat.drop (['Date_Start','Date_End','time_end','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                telemDat['fileName'] = np.repeat(rxfile,len(telemDat))
                telemDat['recID'] = np.repeat(recName,len(telemDat))
                telemDat['ScanTime'] = np.repeat(scanTime,len(telemDat))
                telemDat['Channels'] = np.repeat(channels,len(telemDat))
                telemDat['RecType'] = np.repeat(recType,len(telemDat))
                telemDat.drop(['StartTime','Data','Events','EndTime','duration','events_per_duration'], axis = 1, inplace = True)
                tuples = zip(telemDat.FreqCode.values,telemDat.recID.values,telemDat.Epoch.values)
                index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                telemDat.set_index(index,inplace = True,drop = False)
                telemDat.to_sql('tblRaw',con = conn,index = False, if_exists = 'append')

    # commit and closee
    conn.commit()
    c.close()

def telemDataImport(site,recType,file_directory,projectDB, scanTime = 1, channels = 1, ant_to_rec_dict = None):
    tFiles = os.listdir(file_directory)
    for f in tFiles:
        print ("start importing file %s"%(f))
        f_dir = os.path.join(file_directory,f)
        rxfile=f
        if recType == 'lotek':
            lotek_import(f_dir,rxfile,projectDB,site,ant_to_rec_dict)
        elif recType == 'orion':
            orionImport(f_dir,rxfile,projectDB,site, scanTime, channels, ant_to_rec_dict)
        else:
            print ("There currently is not an import routine created for this receiver type.  Please try again")
        print ("File %s imported"%(f))
    print ("Raw Telemetry Data Import Completed")