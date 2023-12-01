# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import os
import biotas_refactor.predictors as predictors

def ares(file_name, 
                 db_dir, 
                 rec_id, 
                 study_tags, 
                 scan_time = 1, 
                 channels = 1, 
                 ant_to_rec_dict = None):
    # identify the receiver type 
    rec_type = 'ares'
    
    # what file format is it?  the header row is the key
    o_file =open(file_name, encoding='utf-8')
    header = o_file.readline()[:-1]                                            
    columns = header.split(',')
    o_file.close()
    
    # import ARES data and standardize
    if columns[0] == 'Date':
        telem_dat = pd.read_csv(file_name, 
                                names = ['Date','Time','RxID','Freq','Antenna',
                                         'Protocol','Code','Power','Squelch','Noise Level',
                                         'Pulse Width 1','Pulse Width 2','Pulse Width 3',
                                         'Pulse Interval 1','Pulse Interval 2','Pulse Interval 3'],
                                header = 0,
                                index_col = False)
        
        telem_dat['time_stamp'] = pd.to_datetime(telem_dat['Date'] + ' ' + telem_dat['Time'],errors = 'coerce')
        kHz = telem_dat.Freq * 1000
        kHz = np.round(kHz / 5.0, 0) * 5
        MHz = np.round(kHz / 1000.,3)
        telem_dat['Freq'] = MHz
        telem_dat['Freq'] = telem_dat['Freq'].apply(lambda x: f"{x:.3f}" )
        telem_dat['freq_code'] = telem_dat['Freq'].astype(str) + ' ' + telem_dat['Code'].astype(str)
        telem_dat.rename(columns = {'Power':'power'}, inplace = True)
        telem_dat.drop(columns = ['Date','Time','RxID','Freq','Antenna',
                                 'Protocol','Code','Squelch','Noise Level',
                                 'Pulse Width 1','Pulse Width 2','Pulse Width 3',
                                 'Pulse Interval 1','Pulse Interval 2','Pulse Interval 3'],
                       inplace = True)
    
    # import mitas data and standardize
    else:
        telem_dat = pd.read_csv(file_name,
                                names = ['contactId','decodeTimeUTC-05:00','ReceiverId',
                                         'antenna','frequency','codeNumber','protocol','power'],
                                header = 0,
                                index_col = None)
        
        telem_dat['decodeTimeUTC-05:00'] = pd.to_datetime(telem_dat['decodeTimeUTC-05:00'])
        kHz = telem_dat.frequency * 1000
        kHz = np.round(kHz / 5.0,0) * 5
        MHz = np.round(kHz / 1000., 3)
        telem_dat['frequency'] = MHz
        telem_dat['frequency'] = telem_dat['frequency'].apply(lambda x: f"{x:.3f}" )
        telem_dat['freq_code'] = telem_dat['frequency'].astype(str) + ' ' + telem_dat['codeNumber'].astype(str)
        telem_dat.rename(columns = {'decodeTimeUTC-05:00':'time_stamp'}, inplace = True)
        telem_dat.drop(columns = ['contactId','ReceiverId','antenna',
                                 'frequency','codeNumber','protocol'],
                       inplace = True)
        
    # now do this stuff to files regardless of type
    telem_dat['Epoch'] = (telem_dat['time_stamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()        
    telem_dat['rec_type'] = np.repeat(rec_type,len(telem_dat))
    telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))
    telem_dat['channels'] = np.repeat(channels,len(telem_dat))
    telem_dat['scan_time'] = np.repeat(scan_time, len(telem_dat))
    telem_dat['noise_ratio'] = predictors.noise_ratio(5.0,
                                           telem_dat.freq_code.values,
                                           telem_dat.Epoch.values,
                                           study_tags)
        
    telem_dat = telem_dat.astype({'power':'float32',
                                  'freq_code':'object',
                                  'time_stamp':'datetime64',
                                  'scan_time':'int32',
                                  'channels':'int32',
                                  'rec_type':'object',
                                  'Epoch':'float32',
                                  'noise_ratio':'float32',
                                  'rec_id':'object'})
    
    with pd.HDFStore(db_dir, mode='a') as store:
        store.append(key = 'raw_data',
                     value = telem_dat, 
                     format = 'table', 
                     index = False,
                     min_itemsize = {'freq_code':20,
                                     'rec_type':20,
                                     'rec_id':20},
                     append = True, 
                     chunksize = 1000000)


def orion_import(file_name, 
                 db_dir, 
                 rec_id, 
                 study_tags, 
                 scan_time = 1, 
                 channels = 1, 
                 ant_to_rec_dict = None):
    '''Function imports raw Sigma Eight orion data.

    Text parser uses simple column fixed column widths.

    '''
    # identify the receiver type
    rec_type = 'orion'

    # what orion firmware is it?  the header row is the key
    o_file =open(file_name, encoding='utf-8')
    header = o_file.readline()[:-1]                                            
    columns = str.split(header)
    o_file.close()
    
    if 'Type' in columns:
        # with our data row, extract information using pandas fwf import procedure
        telem_dat = pd.read_fwf(file_name,colspecs = [(0,12),(13,23),(24,30),(31,35),(36,45),(46,54),(55,60),(61,65)],
                                names = ['Date','Time','Site','Ant','Freq','Type','Code','power'],
                                skiprows = 1,
                                dtype = {'Date':str,'Time':str,'Site':np.int32,'Ant':str,'Freq':str,'Type':str,'Code':str,'power':np.float64})
        telem_dat = telem_dat[telem_dat.Type != 'STATUS']
        telem_dat.drop(['Type'], axis = 1, inplace = True)

    else:
        # with our data row, extract information using pandas fwf import procedure
        telem_dat = pd.read_fwf(file_name,colspecs = [(0,11),(11,20),(20,26),(26,30),(30,37),(37,42),(42,48)],
                                names = ['Date','Time','Site','Ant','Freq','Code','power'],
                                skiprows = 1,
                                dtype = {'Date':str,'Time':str,'Site':str,'Ant':str,'Freq':str,'Code':str,'power':str})

    if len(telem_dat) > 0:
        # add file name to data
        #['fileName'] = np.repeat(file_name,len(telem_dat))    #Note I'm going back here to the actual file name without the path.  Is that OK?  I prefer it, but it's a potential source of confusion
        
        # aggregate frequency and code intoa unique identifier
        telem_dat['freq_code'] = telem_dat['Freq'].astype(str) + ' ' + telem_dat['Code'].astype(str)
        
        # concatenate date and time and create timestamp
        telem_dat['time_stamp'] = pd.to_datetime(telem_dat['Date'] + ' ' + telem_dat['Time'],errors = 'coerce')# create timestamp field from date and time and apply to index
        telem_dat['scan_time'] = np.repeat(scan_time,len(telem_dat))
        telem_dat['channels'] = np.repeat(channels,len(telem_dat))
        telem_dat['rec_type'] = np.repeat('orion',len(telem_dat))
        telem_dat = telem_dat[telem_dat.time_stamp.notnull()]
        
        if len(telem_dat) == 0:
            print ("Invalid timestamps in raw data, cannot import")
        else:
            # create epoch
            telem_dat['Epoch'] = (telem_dat['time_stamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            
            # drop unnecessary columns 
            telem_dat.drop (['Date','Time','Freq','Code','Site'],axis = 1, inplace = True)
            
            # calculate noise ratio
            telem_dat['noise_ratio'] = predictors.noise_ratio(5.0,
                                                   telem_dat.freq_code.values,
                                                   telem_dat.Epoch.values,
                                                   study_tags)
            
            # if there is no antenna to receiver dictionary 
            if ant_to_rec_dict == None:
                # drop the antenna column - we don't need it anymore
                telem_dat.drop(['Ant'], axis = 1, inplace = True)
                
                # add receiver id 
                telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))
                
                # create a multindex 
                # tuples = zip(telem_dat.freq_code.values,telem_dat.rec_id.values,telem_dat.Epoch.values)
                # index = pd.MultiIndex.from_tuples(tuples, names=['frqe_code', 'rec_id','Epoch'])
                # telem_dat.set_index(index,inplace = True,drop = False)

                telem_dat = telem_dat.astype({'power':'float32',
                                              'freq_code':'object',
                                              'time_stamp':'datetime64',
                                              'scan_time':'int32',
                                              'channels':'int32',
                                              'rec_type':'object',
                                              'Epoch':'float32',
                                              'noise_ratio':'float32',
                                              'rec_id':'object'})
                
                with pd.HDFStore(db_dir, mode='a') as store:
                    store.append(key = 'raw_data',
                                 value = telem_dat, 
                                 format = 'table', 
                                 index = False,
                                 min_itemsize = {'freq_code':20,
                                                 'rec_type':20,
                                                 'rec_id':20},
                                 append = True, 
                                 chunksize = 1000000)


            # if there is an antenna to receiver dictionary
            else:
                for i in ant_to_rec_dict:
                    # get site from dictionary
                    site = ant_to_rec_dict[i]
                    
                    # get telemetryt data associated with this site
                    telem_dat_sub = telem_dat[telem_dat.Ant == str(i)]
                    
                    # add receiver ID
                    telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                    
                    # index
                    # tuples = zip(telem_dat_sub.FreqCode.values,
                    #              telem_dat_sub.recID.values,
                    #              telem_dat_sub.Epoch.values)
                    # index = pd.MultiIndex.from_tuples(tuples, names=['freq_code', 'rec_id','Epoch'])
                    #telem_dat_sub.set_index(index,inplace = True,drop = False)
                    
                    # remove exctranneous columns
                    telem_dat_sub.drop(['Ant'], axis = 1, inplace = True)
                    #TODO this needs to be switched to HDF
                    # write to HDF
                    
                    telem_dat_sub = telem_dat_sub.astype({'power':'float32',
                                                  'freq_code':'object',
                                                  'time_stamp':'datetime64',
                                                  'scan_time':'int32',
                                                  'channels':'int32',
                                                  'rec_type':'object',
                                                  'Epoch':'float32',
                                                  'noise_ratio':'float32',
                                                  'rec_id':'object'})
                    
                    telem_dat_sub.reset_index(drop = True, inplace = True)

                    with pd.HDFStore(db_dir, mode='a') as store:
                        store.append(key = 'raw_data',
                                     value = telem_dat_sub, 
                                     format = 'table', 
                                     index = False,
                                     min_itemsize = {'freq_code':20,
                                                     'rec_type':20,
                                                     'rec_id':20},
                                     append = True, 
                                     chunksize = 1000000)                    
                    
def vr2_import(file_name,db_dir,study_tags, rec_id):
    '''Function imports raw VEMCO VR2 acoustic data.

    '''

    recType = 'vr2'

    # import csv format and export - pretty simple
    telem_dat = pd.read_csv(file_name)

    # if data isn't empty - hey, things go wrong
    if len(telem_dat) > 0:
        # depending upon header data create a timestep
        try:
            telem_dat['time_stamp'] = pd.to_datetime(telem_dat['Date and Time (UTC)'])
        except KeyError:
            seconds = pd.to_numeric(telem_dat['Unnamed: 2'].str.split(":", n = 1, expand = True)[1])
            telem_dat['time_stamp'] = pd.to_datetime(telem_dat['Date and Time'])
            telem_dat['time_stamp'] = telem_dat.timeStamp + pd.to_timedelta(seconds,unit = 's')
            telem_dat['rec_id'] = telem_dat['Receiver'].str.split("-", expand = True)[0]
            telem_dat['rec_id'] = telem_dat.Receiver + "-" + telem_dat['Receiver.1'].astype('str')
        telem_dat['scan_time'] = np.repeat(1,len(telem_dat))
        telem_dat['channels'] = np.repeat(1,len(telem_dat))
        telem_dat['rec_type'] = np.repeat('vr2',len(telem_dat))
        telem_dat['transmitter'] = telem_dat['transmitter'].str.split("-", n = 2, expand = True)[2]
        telem_dat['transmitter'] = telem_dat.transmitter.astype(str)
        telem_dat.rename(columns = {'Receiver':'rec_id','transmitter':'freq_code'}, inplace = True)
        telem_dat['Epoch'] = (telem_dat['time_stamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
        try:
            telem_dat.drop (['Date and Time (UTC)', 'Transmitter Name','Transmitter Serial','Sensor Value','Sensor Unit','Station Name','Latitude','Longitude','Transmitter Type','Sensor Precision'],axis = 1, inplace = True)
        except KeyError:
            telem_dat.drop (['Unnamed: 0', 'Date and Time', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Receiver.1', 'Unnamed: 7', 'Unnamed: 8', 'Transmitter Name', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',],axis = 1, inplace = True)

        # tuples = zip(telem_dat.FreqCode.values,telem_dat.recID.values,telem_dat.Epoch.values)
        # index = pd.MultiIndex.from_tuples(tuples, names=['freq_code', 'rec_id','Epoch'])
        # telem_dat.set_index(index,inplace = True,drop = False)
        
        telem_dat = telem_dat.astype({'power':'np.float32',
                                      'frea_code':'object',
                                      'time_stamp':'datetime64',
                                      'scan_time':'int32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'Epoch':'float32',
                                      'noise_ratio':'float32',
                                      'rec_id':'object'})
        
        with pd.HDFStore(db_dir, mode='a') as store:
            store.append(key = 'raw_data',
                         value = telem_dat, 
                         format = 'table', 
                         index = False,
                         min_itemsize = {'freq_code':20,
                                         'rec_type':20,
                                         'rec_id':20},
                         append = True, 
                         chunksize = 1000000)        
def srx1200(file_name,
             db_dir,
             rec_id, 
             study_tags, 
             scan_time = 1, 
             channels = 1, 
             ant_to_rec_dict = None):
    rec_type = 'srx1200'
    
    # create empty dictionary to hold Lotek header data indexed by line number - to be imported to Pandas dataframe
    header_dat = {}
    
    # create empty array to hold line indices
    line_counter = []
    
    # generate a list of header lines - contains all data we need to write to project set up database
    line_list = []
    
    # open file passed to function
    o_file = open(file_name, encoding='utf-8')
    
    # start line counter
    counter = 0
    
    # start end of file 
    eof = 0
    line_counter.append(counter)
    
    # read first line in file
    line = o_file.readline()[:-1]
    
    # append the current line of header data to the line list
    line_list.append(line)
    
    # determine file format 
    file_header = True
    if 'SRX1200 Data File Export' not in line:
        file_header = False
    
    # if the srx1200 file has header information
    if file_header == True:
        # find where data begins and header data ends
        with o_file as f:
            for line in f:
                # if this current line signifies the start of the data stream, the data starts three rows down from this
                if "All Detection Records" in line:
                    counter = counter + 1
                    dataRow = counter + 5
                    # break the loop, we have reached our stop point
                    
                # else we are still reading header data increase the line counter by 1
                elif "Battery Report" in line:
                    counter = counter + 1
                    dataEnd = counter - 2
                else:
                    counter = counter + 1
                    # append the line counter to the count array
                    line_counter.append(counter)
                    # append line of data to the data array
                    line_list.append(line)
                    eof = counter

        # add count array to dictionary with field name 'idx' as key
        header_dat['idx'] = line_counter
        # add data line array to dictionary with field name 'line' as key
        header_dat['line'] = line_list
        # create pandas dataframe of header data indexed by row number
        header_df = pd.DataFrame.from_dict(header_dat)
        header_df.set_index('idx',inplace = True)
            
        # find scan time
        for row in header_df.iterrows():
            # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document
            if 'Scan Period' in row[1][0]:
                # get the number value from the row
                scanTimeStr = row[1][0]
                #scanTimeStr = scanTimeStr.rstrip()
                scanTimeStr = scanTimeStr[:-4]
                # split the string
                new_split = None
                scanTimeSplit = scanTimeStr.split(':')
                if 'sec' in scanTimeSplit[1]:
                    new_split = scanTimeSplit[1].split(' ')
                    scanTimeSplit[1] = new_split[8]
                # convert the scan time string to float
                scanTime = float(scanTimeSplit[1])
                # stop the loop, we have extracted scan time
                del row, scanTimeStr, scanTimeSplit
                break
        
        # find number of channels and create channel dictionary
        scanChan = []
        channelDict = {}
        counter = 0
        rows = header_df.iterrows()
        # for every row
        for row in rows:
            # if the first 18 characters say what that says
            if 'Channel Settings' in row[1][0]:
                # channel dictionary data starts two rows from here
                idx0 = counter + 5
                # while the next row isn't empty
                while 'Antenna Configuration' not in next(rows)[1][0]:
                    # increase the counter
                    counter = counter + 1
                # get index of last data row
                idx1 = counter
                # when the row is empty we have reached the end of channels,
                break
             # else, increase the counter by 1
            else:
                counter = counter + 1
        del row, rows

        # extract channel dictionary data using rows identified earlier
        channelDat = header_df.iloc[idx0:idx1]
        for row in channelDat.iterrows():
            dat = row[1][0]
            channel = float(dat[0:6])
            frequency = dat[16:23]
            channelDict[channel] = frequency
            # extract that channel ID from the data row and append to array
            scanChan.append(channel)
        channels = len(scanChan)
        
        del row

        # read in telemetry data
        if new_split == None:
            telem_dat = pd.read_fwf(file_name,
                                   colspecs = [(0,6),(6,20),(20,32),(32,40),(40,55),(55,64),(64,72),(72,85),(85,93),(93,102)],
                                   names = ['Index','Date','Time','[uSec]','Tag/BPM','Freq [MHz]','Codeset','Antenna','Gain','RSSI'],
                                   skiprows = dataRow, 
                                   skipfooter = eof - dataEnd)
            telem_dat.drop(columns = ['Index'], inplace = True)

        else:
            telem_dat = pd.read_csv(file_name,
                                   names = ['Index','Date','Time','[uSec]','Tag/BPM','Freq [MHz]','Codeset','Antenna','','Gain','RSSI'],
                                   skiprows = dataRow, 
                                   skipfooter = eof - dataEnd)
            telem_dat.drop(columns = ['', 'Index'], inplace = True)
            telem_dat.dropna(inplace = True)
            telem_dat = telem_dat.astype({'[uSec]':'int32'})

        
        # create a timestamp and conver to datetime object
        telem_dat['time_stamp'] = telem_dat.Date + ' ' + telem_dat.Time + '.' + telem_dat['[uSec]'].astype(str)

        telem_dat['time_stamp'] = pd.to_datetime(telem_dat.time_stamp)
        
        # calculate Epoch
        telem_dat['Epoch'] = (telem_dat['time_stamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()       
        
        # format frequency code
        telem_dat['FreqNo'] = telem_dat['Freq [MHz]'].apply(lambda x: f"{x:.3f}" )
        telem_dat = telem_dat[telem_dat['Tag/BPM'] != 999]
        telem_dat['freq_code'] = telem_dat['FreqNo'] + ' ' + telem_dat['Tag/BPM'].astype(np.str)
        
        # calculate 
        telem_dat['noise_ratio'] = predictors.noise_ratio(600,
                                              telem_dat.freq_code.values,
                                              telem_dat.Epoch.values,
                                              study_tags)

        # write scan time, channels, rec type and recID to data
        telem_dat['scan_time'] = np.repeat(scan_time,len(telem_dat))
        telem_dat['channels'] = np.repeat(channels,len(telem_dat))
        telem_dat['rec_type'] = np.repeat(rec_type,len(telem_dat))
        telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))
        telem_dat['noise_ratio'] = telem_dat.noise_ratio.values.astype(np.float32)
       
        # remove unnecessary columns
        telem_dat.drop(columns = ['Date','Time','[uSec]','Freq [MHz]','FreqNo', 'Codeset', 'Gain','Tag/BPM','Antenna'], inplace = True)
        telem_dat.rename(columns = {'RSSI':'power'}, inplace = True)
        telem_dat['power'] = telem_dat.power.values.astype(np.float32)
        telem_dat['noise_ratio'] = telem_dat.noise_ratio.values.astype(np.float32)

        telem_dat.reset_index(inplace = True)
        
        telem_dat = telem_dat.astype({'power':'float32',
                                      'freq_code':'object',
                                      'time_stamp':'datetime64',
                                      'scan_time':'int32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'Epoch':'float32',
                                      'noise_ratio':'float32',
                                      'rec_id':'object'})
        
        if new_split != None:
            telem_dat.drop(columns = ['index'], inplace = True)
            print ('fuck')
                
        with pd.HDFStore(db_dir, mode='a') as store:
            store.append(key = 'raw_data',
                         value = telem_dat, 
                         format = 'table', 
                         index = False, 
                         append = True, 
                         min_itemsize = {'freq_code':20,
                                         'rec_type':20,
                                         'rec_id':20},
                         chunksize = 1000000)
        
    # if the data doesn't have a header
    else:
        stop = 0
        eof = 0
        # find where data ends
        with o_file as f:
            for line in f:
                if "Battery Report" in line:
                    stop = counter - 1

                else:
                    counter = counter + 1
                    eof = counter
        
        # import data using specs defined above
        telem_dat = pd.read_csv(file_name, skiprows = 3, skipfooter = eof - stop)
        
        # create a timestamp and conver to datetime object
        telem_dat['time_stamp'] = telem_dat.Date + ' ' + telem_dat.Time + '.' + telem_dat['[uSec]'].astype(str)
        telem_dat['time_stamp'] = pd.to_datetime(telem_dat.time_stamp)
        
        # calculate Epoch
        telem_dat['Epoch'] = (telem_dat['time_stamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()       
        
        # format frequency code
        telem_dat['FreqNo'] = telem_dat['Freq [MHz]'].apply(lambda x: f"{x:.3f}" )
        telem_dat = telem_dat[telem_dat['TagID/BPM'] != 999]

        telem_dat['freq_code'] = telem_dat['FreqNo'] + ' ' + telem_dat['TagID/BPM'].astype(np.str)
        
        # calculate 
        telem_dat['noise_ratio'] = predictors.noise_ratio(600,
                                              telem_dat.freq_code.values,
                                              telem_dat.Epoch.values,
                                              study_tags)

        # write scan time, channels, rec type and recID to data
        telem_dat['scan_time'] = np.repeat(scan_time,len(telem_dat))
        telem_dat['channels'] = np.repeat(channels,len(telem_dat))
        telem_dat['rec_type'] = np.repeat(rec_type,len(telem_dat))
        telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))
                
        # remove unnecessary columns
        telem_dat.drop(columns = ['Index','Date','Time','[uSec]','Freq [MHz]','FreqNo', 'Codeset', 'Gain','TagID/BPM','Antenna'], inplace = True)
        telem_dat.rename(columns = {'RSSI':'power'}, inplace = True)
        telem_dat['noise_ratio'] = telem_dat.noise_ratio.values.astype(np.float32)
        telem_dat['power'] = telem_dat.power.values.astype(np.float32)
        
        telem_dat.reset_index(inplace = True, drop = True)
        
        telem_dat = telem_dat.astype({'power':'float32',
                                      'freq_code':'object',
                                      'time_stamp':'datetime64',
                                      'scan_time':'int32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'Epoch':'float32',
                                      'noise_ratio':'float32',
                                      'rec_id':'object'})
                
        with pd.HDFStore(db_dir, mode='a') as store:
            store.append(key = 'raw_data',
                         value = telem_dat, 
                         format = 'table', 
                         index = False, 
                         min_itemsize = {'freq_code':20,
                                         'rec_type':20,
                                         'rec_id':20},
                         append = True, 
                         chunksize = 1000000)    
def srx800(file_name,
             db_dir,
             rec_id, 
             study_tags, 
             scan_time = 1, 
             channels = 1, 
             ant_to_rec_dict = None):
    
    rec_type = 'srx800'
    
    # create empty dictionary to hold Lotek header data indexed by line number - to be imported to Pandas dataframe
    header_dat = {}
    
    # create empty array to hold line indices
    line_counter = []
    
    # generate a list of header lines - contains all data we need to write to project set up database
    line_list = []
    
    # open file passed to function
    o_file = open(file_name, encoding='utf-8')
    
    # start line counter
    counter = 0
    
    # start end of file 
    eof = 0
    line_counter.append(counter)
    
    # read first line in file
    line = o_file.readline()[:-1]
    
    # append the current line of header data to the line list
    line_list.append(line)
    
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
                line_counter.append(counter)
                # append line of data to the data array
                line_list.append(line)

    # add count array to dictionary with field name 'idx' as key
    header_dat['idx'] = line_counter
    # add data line array to dictionary with field name 'line' as key
    header_dat['line'] = line_counter
    # create pandas dataframe of header data indexed by row number
    headerDF = pd.DataFrame.from_dict(header_dat)
    headerDF.set_index('idx',inplace = True)
    
    data_format = 'primary'
    for row in  headerDF.iterrows():
        idx = row[0]
        if 'Environment History:' in row[1][0]:
            data_format = 'alternate'
        if idx > 0:
            break
        
    # find scan time
    for row in headerDF.iterrows():
        # if the first 9 characters of the line say 'Scan Time' = we have found the scan time in the document
        if 'Scan Time' in row[1][0] or 'scan time' in row[1][0]:
            # get the number value from the row
            scanTimeStr = row[1][0][-7:-1]
            # split the string
            scanTimeSplit = scanTimeStr.split(':')
            # convert the scan time string to float
            scanTime = float(scanTimeSplit[1])
            # stop the loop, we have extracted scan time
            del row, scanTimeStr, scanTimeSplit
            break

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
            del row, firmwarestr, firmware_split
            break
    

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
    #del row, rows

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

    # with our data row, extract information using pandas fwf import procedure
    if firmware == '9.12.5':
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,8),(8,23),(23,33),(33,41),(41,56),(56,64)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})
    elif data_format == 'alternate':
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,5),(5,14),(14,23),(23,31),(31,46),(46,53)],
                               names = ['DayNumber','Time','ChannelID','TagID','Antenna','Power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})
        telem_dat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telem_dat))
        telem_dat['Date'] = telem_dat['day0'] + pd.to_timedelta(telem_dat['DayNumber'].astype(int), unit='d')
        telem_dat['Date'] = telem_dat.Date.astype('str')
        telem_dat.drop(columns = ['DayNumber','day0'], inplace = True)
    else:
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,8),(8,18),(18,28),(28,36),(36,51),(51,59)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})

    telem_dat.dropna(inplace = True)

    def id_to_freq(row,channelDict):
        if row[2] in channelDict:
            return channelDict[row[2]]
        else:
            return '888'
    
    if len(telem_dat) > 0:
        for ant in ant_to_rec_dict:
            site = ant_to_rec_dict[ant]
            telem_dat_sub = telem_dat[telem_dat.Antenna == ant]
            
            # get frequency from channel
            telem_dat_sub['Frequency'] = telem_dat_sub['ChannelID'].map(channelDict)

            # remove extraneous data
            telem_dat_sub = telem_dat_sub[telem_dat_sub.Frequency != '888']
            #telemDat_sub = telemDat_sub[telemDat_sub.TagID != 999]
            
            # create FreqCode
            telem_dat_sub['freq_code'] = telem_dat_sub['Frequency'].astype(str) + ' ' + telem_dat_sub['TagID'].astype(int).astype(str)
            
            # create timestamp
            telem_dat_sub['time_stamp'] = pd.to_datetime(telem_dat_sub['Date'] + ' ' + telem_dat_sub['Time'])# create timestamp field from date and time and apply to index
            
            # get UNIX epoch
            telem_dat_sub['Epoch'] = (telem_dat_sub['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            
            # calculate noise ratio
            telem_dat_sub = predictors.noise_ratio(5.0,telem_dat_sub,study_tags)
            
            # remove extraneous columns
            telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
            
            # write scan time, channels, rec type and recID to data
            telem_dat_sub['scan_time'] = np.repeat(scan_time,len(telem_dat_sub))
            telem_dat_sub['channels'] = np.repeat(channels,len(telem_dat_sub))
            telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))
            telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
            telem_dat_sub.reset_index(inplace = True)
            telem_dat_sub['noise_ratio'] = telem_dat_sub.noise_ratio.values.astype(np.float32)

            # write to hdf
            
            telem_dat_sub = telem_dat_sub.astype({'power':'np.float32',
                                          'frea_code':'object',
                                          'time_stamp':'datetime64',
                                          'scan_time':'int32',
                                          'channels':'int32',
                                          'rec_type':'object',
                                          'Epoch':'float32',
                                          'noise_ratio':'float32',
                                          'rec_id':'object'})
                
            with pd.HDFStore(db_dir, mode='a') as store:
                store.append(key = 'raw_data',
                             value = telem_dat_sub, 
                             format = 'table', 
                             index = False,
                             min_itemsize = {'freq_code':20,
                                             'rec_type':20,
                                             'rec_id':20},
                             append = True, 
                             chunksize = 1000000)

def srx600(file_name,
             db_dir,
             rec_id, 
             study_tags, 
             scan_time = 1, 
             channels = 1, 
             ant_to_rec_dict = None):
    
    rec_type = 'srx600'
    
    # create empty dictionary to hold Lotek header data indexed by line number - to be imported to Pandas dataframe
    header_dat = {}
    
    # create empty array to hold line indices
    line_counter = []
    
    # generate a list of header lines - contains all data we need to write to project set up database
    line_list = []
    
    # open file passed to function
    o_file = open(file_name, encoding='utf-8')
    
    # start line counter
    counter = 0
    
    # start end of file 
    eof = 0
    line_counter.append(counter)
    
    # read first line in file
    line = o_file.readline()[:-1]
    
    # append the current line of header data to the line list
    line_list.append(line)
    
    with o_file as f:
        for line in f:
            if "** Data Segment **" in line:
                # if this current line signifies the start of the data stream
                counter = counter + 1
                # the data starts 5 rows down from this
                dataRow = counter + 5
                # break the loop, we have reached our stop point
                break

            else:
                # if we are still reading header data increase the line counter by 1
                counter = counter + 1
                # append the line counter to the count array
                line_counter.append(counter)
                # append line of data to the data array
                line_list.append(line)
    
    # add count array to dictionary with field name 'idx' as key
    header_dat['idx'] = line_counter
    
    # add data line array to dictionary with field name 'line' as key
    header_dat['line'] = line_list
    
    # create pandas dataframe of header data indexed by row number
    headerDF = pd.DataFrame.from_dict(header_dat)
    headerDF.set_index('idx',inplace = True)
    
    # create a variable for header type because apparently Lotek makes its mind up
    header_type = 'standard'

    # gotta love multiple header types 
    for row in headerDF.iterrows():
        if row[1][0] == 'Environment History:':
            header_type = 'alternate'
            break
    del row
        
    # find scan time
    # for every header data row
    for row in headerDF.iterrows():   
        if 'scan time' in row[1][0] or 'Scan Time' in row[1][0]:
            # get the number value from the row
            scanTimeStr = row[1][0][-7:-1]
            # split the string
            scanTimeSplit = scanTimeStr.split(':')
            # convert the scan time string to float
            scanTime = float(scanTimeSplit[1])
            # stop that loop, we done
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

    def id_to_freq(row,channelDict):
        channel = row['ChannelID']
        if np.int(channel) in channelDict:
            return channelDict[np.int(channel)]
        else:
            return '888'
        
    if header_type == 'alternate':

        # with our data row, extract information using pandas fwf import procedure
        telem_dat = pd.read_fwf(os.path.join(file_name),
                               colspecs = [(0,5),(5,14),(14,20),(20,26),(26,30),(30,35)],
                               names = ['DayNumber','Time','ChannelID','Power','Antenna','TagID'],
                               skiprows = dataRow)
        
        # remove last two rows, Lotek adds garbage at the end
        telem_dat = telem_dat.iloc[:-2]
        telem_dat.dropna(inplace = True)
        
        if len(telem_dat) > 0:
            for ant in ant_to_rec_dict:
                site = ant_to_rec_dict[ant]
                telem_dat_sub = telem_dat[telem_dat.Antenna == ant]
                
                # calculate frequency with channel ID
                telem_dat_sub['Frequency'] = telem_dat_sub['ChannelID'].map(channelDict) #(id_to_freq, axis = 1, args = (channelDict,))
        
                # remove some extraneous data
                telem_dat_sub = telem_dat_sub[telem_dat_sub.Frequency != '888']
                telem_dat_sub = telem_dat_sub[telem_dat_sub.TagID != 999]
                
                # create freq code column
                telem_dat_sub['freq_code'] = telem_dat_sub['Frequency'].astype(str) + ' ' + telem_dat_sub['TagID'].astype(int).astype(str)
                
                # deal with that pesky number date and clean up
                telem_dat_sub['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telem_dat_sub))
                telem_dat_sub['Date'] = telem_dat_sub['day0'] + pd.to_timedelta(telem_dat_sub['DayNumber'].astype(int), unit='d')
                telem_dat_sub['Date'] = telem_dat_sub.Date.astype('str')
                telem_dat_sub['time_stamp'] = pd.to_datetime(telem_dat_sub['Date'] + ' ' + telem_dat_sub['Time'])
                telem_dat_sub.drop(['day0','DayNumber'],axis = 1, inplace = True)
                
                # calculate unix epoch
                telem_dat_sub['Epoch'] = (telem_dat_sub['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                
                # clean up some more
                telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                
                # add file name to data
                #telemDat_sub['fileName'] = np.repeat(rxfile,len(telemDat_sub))
                
                # add recID
                telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                
                # calculate noise ratio
                telem_dat_sub = predictors.noise_ratio(5.0,
                                           telem_dat_sub.freq_code.values,
                                           telem_dat_sub.Epoch.values,
                                           study_tags)
                
                # add in scan time, channels, rec type
                telem_dat_sub['scan_time'] = np.repeat(scan_time,len(telem_dat_sub))
                telem_dat_sub['channels'] = np.repeat(channels,len(telem_dat_sub))
                telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))
                
                # make multiindex for fast indexing
                # tuples = zip(telem_dat_sub.FreqCode.values,telem_dat_sub.recID.values,telem_dat_sub.Epoch.values)
                # index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                # telem_dat_sub.set_index(index,inplace = True,drop = False)
                telem_dat_sub.reset_index(inplace = True)
                telem_dat_sub['noise_ratio'] = telem_dat_sub.noise_ratio.values.astype(np.float32)

                # write to SQL
                
                telem_dat_sub = telem_dat_sub.astype({'power':'np.float32',
                                              'frea_code':'object',
                                              'time_stamp':'datetime64',
                                              'scan_time':'int32',
                                              'channels':'int32',
                                              'rec_type':'object',
                                              'Epoch':'float32',
                                              'noise_ratio':'float32',
                                              'rec_id':'object'})
                
                with pd.HDFStore(db_dir, mode='a') as store:
                    store.append(key = 'raw_data',
                                 value = telem_dat_sub, 
                                 format = 'table', 
                                 index = False,
                                 min_itemsize = {'freq_code':20,
                                                 'rec_type':20,
                                                 'rec_id':20},
                                 append = True, 
                                 chunksize = 1000000)    
    else:
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,9),(9,19),(19,29),(29,36),(36,44),(44,52)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                               skiprows = dataRow)
        
        # remove last two rows, Lotek adds garbage at the end
        telem_dat = telem_dat.iloc[:-2]
        telem_dat.dropna(inplace = True)
        
        if len(telem_dat_sub) > 0:
            for ant in ant_to_rec_dict:
                site = ant_to_rec_dict[ant]
                telem_dat_sub = telem_dat[telem_dat.Antenna == ant]
                
                # calculate frequency with channel ID
                telem_dat_sub['Frequency'] = telem_dat_sub['ChannelID'].map(channelDict)
                
                # remove extra data
                telem_dat_sub = telem_dat_sub[telem_dat_sub.Frequency != '888']
                telem_dat_sub = telem_dat_sub[telem_dat_sub.TagID != 999]
                
                
                # calculate freqcode
                telem_dat_sub['freqcode'] = telem_dat_sub['Frequency'].astype(str) + ' ' + telem_dat_sub['TagID'].astype(int).astype(str)
                
                # get timestamp
                telem_dat_sub['time_stamp'] = pd.to_datetime(telem_dat_sub['Date'] + ' ' + telem_dat_sub['Time'])# create timestamp field from date and time and apply to index
                
                # calculate UNIX epoch 
                telem_dat_sub['Epoch'] = (telem_dat_sub['timeStamp'] - datetime.datetime(1970,1,1)).dt.total_seconds()
                
                # calculate noise ratio
                telem_dat_sub = predictors.noise_ratio(5.0,
                                           telem_dat_sub.freq_code.values,
                                           telem_dat_sub.Epoch.values,
                                           study_tags)
                
                # clean up some more
                telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                
                # add in scan time, channels, rec type
                telem_dat_sub['scan_time'] = np.repeat(scan_time,len(telem_dat_sub))
                telem_dat_sub['channels'] = np.repeat(channels,len(telem_dat_sub))
                telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))
                telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                
                # make multiindex for fast indexing
                # tuples = zip(telem_dat_sub.FreqCode.values,telem_dat_sub.recID.values,telem_dat_sub.Epoch.values)
                # index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','Epoch'])
                # telem_dat_sub.set_index(index,inplace = True,drop = False)
                telem_dat_sub.reset_index(inplace = True)
                telem_dat_sub['noise_ratio'] = telem_dat.noise_ratio.values.astype(np.float32)

                telem_dat_sub = telem_dat_sub.astype({'power':'np.float32',
                                              'frea_code':'object',
                                              'time_stamp':'datetime64',
                                              'scan_time':'int32',
                                              'channels':'int32',
                                              'rec_type':'object',
                                              'Epoch':'float32',
                                              'noise_ratio':'float32',
                                              'rec_id':'object'})

                # write to SQL
                with pd.HDFStore(db_dir, mode='a') as store:
                    store.append(key = 'raw_data',
                                 value = telem_dat_sub, 
                                 format = 'table', 
                                 index = False,
                                 min_itemsize = {'freq_code':20,
                                                 'rec_type':20,
                                                 'rec_id':20},
                                 append = True, 
                                 chunksize = 1000000)
    
    
    
    
    
    
    