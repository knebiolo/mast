# -*- coding: utf-8 -*-
"""
Data parsers for radio telemetry receiver file formats.

This module provides parser functions to import raw detection data from various
radio telemetry receiver manufacturers into the MAST HDF5 database. Each parser
handles manufacturer-specific file formats and standardizes the data into a common
schema for downstream processing.

Supported Receiver Types
------------------------
- **ARES**: Lotek Advanced Radio Telemetry Systems
- **Orion**: Sigma Eight Orion receivers
- **SRX-1200**: Lotek SRX 1200 receivers (fixed-width format)
- **SRX-800**: Lotek SRX 800 receivers (fixed-width format)
- **SRX-600**: Lotek SRX 600 receivers (fixed-width format)
- **VR2**: Vemco VR2 acoustic receivers (CSV format)
- **PIT**: Passive Integrated Transponder readers

Common Data Pipeline
--------------------
All parsers follow this workflow:
1. Read raw receiver file (CSV, fixed-width, or vendor format)
2. Parse timestamps, frequencies, codes, power, antenna information
3. Calculate derived fields: epoch, noise_ratio
4. Standardize column names and data types
5. Append to HDF5 `/raw_data` table

Standardized Output Schema
--------------------------
All parsers produce these columns:
- `time_stamp` : datetime64 - Detection timestamp
- `epoch` : float32 - Seconds since 1970-01-01
- `freq_code` : object - Frequency + code (e.g., "166.380 7")
- `power` : float32 - Signal power (dB or raw)
- `rec_id` : object - Receiver identifier
- `rec_type` : object - Receiver type (ares, orion, srx1200, etc.)
- `channels` : int32 - Number of receiver channels
- `scan_time` : float32 - Scan duration per channel (seconds)
- `noise_ratio` : float32 - Ratio of miscoded to total detections

Typical Usage
-------------
>>> import pymast.parsers as parsers
>>> 
>>> # Import ARES data
>>> parsers.ares(
...     file_name='receiver_001.csv',
...     db_dir='project.h5',
...     rec_id='REC001',
...     study_tags=['166.380 7', '166.380 12'],
...     scan_time=1.0,
...     channels=1
... )
>>> 
>>> # Import SRX-1200 data
>>> parsers.srx1200(
...     file_name='srx_detections.txt',
...     db_dir='project.h5',
...     rec_id='SRX123',
...     study_tags=['166.380 7'],
...     scan_time=2.5,
...     channels=1
... )

Notes
-----
- Frequency values are rounded to nearest 5 kHz then converted to MHz with 3 decimal precision
- Noise ratio calculated using 5-minute moving window (see `predictors.noise_ratio`)
- All parsers append to existing HDF5 `/raw_data` table (mode='a')
- Timestamps assumed to be in UTC or project-specific timezone
- PIT readers have different schemas due to antenna-based detection logic

See Also
--------
radio_project.import_data : High-level batch import interface
predictors.noise_ratio : Miscoded detection ratio calculation
"""

import pandas as pd
import numpy as np
import datetime
import os
import pymast.predictors as predictors
import sys

def ares(file_name, 
                 db_dir, 
                 rec_id, 
                 study_tags, 
                 scan_time = 1, 
                 channels = 1, 
                 ant_to_rec_dict = None):
    """
    Import Lotek ARES receiver data into MAST HDF5 database.
    
    Parses CSV format detection files from Lotek Advanced Radio Telemetry Systems
    (ARES) receivers. Automatically detects file format variant based on header row
    and standardizes data into common schema.
    
    Parameters
    ----------
    file_name : str
        Absolute path to ARES CSV file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique receiver identifier (e.g., 'REC001', 'SITE_A')
    study_tags : list of str
        List of valid freq_code tags deployed in study (e.g., ['166.380 7', '166.380 12'])
        Used to calculate noise_ratio
    scan_time : float, optional
        Scan duration per channel in seconds (default: 1.0)
    channels : int, optional
        Number of receiver channels (default: 1)
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs (not currently used)
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Handles two ARES file format variants (detected via header row)
    - Frequencies rounded to nearest 5 kHz, formatted as 3-decimal MHz
    - Calculates noise_ratio using 5-minute moving window
    - All timestamps converted to epoch (seconds since 1970-01-01)
    
    Examples
    --------
    >>> import pymast.parsers as parsers
    >>> parsers.ares(
    ...     file_name='C:/data/ares_001.csv',
    ...     db_dir='C:/project/study.h5',
    ...     rec_id='ARES001',
    ...     study_tags=['166.380 7', '166.380 12', '166.380 19'],
    ...     scan_time=1.0,
    ...     channels=1
    ... )
    
    See Also
    --------
    radio_project.import_data : High-level batch import
    predictors.noise_ratio : Noise ratio calculation
    """
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
    telem_dat['epoch'] = np.round((telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
    telem_dat['rec_type'] = np.repeat(rec_type,len(telem_dat))
    telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))
    telem_dat['channels'] = np.repeat(channels,len(telem_dat))
    telem_dat['scan_time'] = np.repeat(scan_time, len(telem_dat))
    telem_dat['noise_ratio'] = predictors.noise_ratio(5.0,
                                           telem_dat.freq_code.values,
                                           telem_dat.epoch.values,
                                           study_tags)
        
    telem_dat = telem_dat.astype({'power':'float32',
                                  'freq_code':'object',
                                  'time_stamp':'datetime64[ns]',
                                  'scan_time':'float32',
                                  'channels':'int32',
                                  'rec_type':'object',
                                  'epoch':'float32',
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
                 scan_time = 1., 
                 channels = 1, 
                 ant_to_rec_dict = None):
    """
    Import Sigma Eight Orion receiver data into MAST HDF5 database.
    
    Parses fixed-width format detection files from Sigma Eight Orion receivers.
    Automatically detects firmware version based on header row and adjusts
    column parsing accordingly.
    
    Parameters
    ----------
    file_name : str
        Absolute path to Orion fixed-width text file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique receiver identifier (e.g., 'ORION_01')
    study_tags : list of str
        List of valid freq_code tags deployed in study
    scan_time : float, optional
        Scan duration per channel in seconds (default: 1.0)
    channels : int, optional
        Number of receiver channels (default: 1)
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs (not currently used)
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Handles two Orion firmware variants: with/without 'Type' column
    - Fixed-width column parsing using pandas read_fwf
    - Filters out 'STATUS' messages (firmware-specific)
    - Frequencies formatted as 3-decimal MHz
    
    Examples
    --------
    >>> parsers.orion_import(
    ...     file_name='C:/data/orion_site1.txt',
    ...     db_dir='C:/project/study.h5',
    ...     rec_id='ORION_SITE1',
    ...     study_tags=['166.380 7'],
    ...     scan_time=1.0,
    ...     channels=1
    ... )
    
    See Also
    --------
    ares : Similar parser for Lotek ARES receivers
    srx1200 : Parser for Lotek SRX 1200 receivers
    """
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
                                skiprows = 1)#,
                                #dtype = {'Date':str,'Time':str,'Site':np.int32,'Ant':str,'Freq':str,'Type':str,'Code':str,'power':np.float64})
        telem_dat = telem_dat[telem_dat.Type != 'STATUS']
        telem_dat['Freq'] = telem_dat.Freq.astype('float32')

        telem_dat['Freq'] = telem_dat['Freq'].apply(lambda x: f"{x:.3f}")
        telem_dat['Ant'] = telem_dat.Ant.astype('object')
        telem_dat.drop(['Type'], axis = 1, inplace = True)

    else:
        # with our data row, extract information using pandas fwf import procedure
        telem_dat = pd.read_fwf(file_name,colspecs = [(0,11),(11,20),(20,26),(26,30),(30,37),(37,42),(42,48)],
                                names = ['Date','Time','Site','Ant','Freq','Code','power'],
                                skiprows = 1)#,
                                #dtype = {'Date':str,'Time':str,'Site':str,'Ant':str,'Freq':str,'Code':str,'power':str})
        telem_dat['Ant'] = telem_dat.Ant.astype('object')
        telem_dat['Freq'] = telem_dat.Freq.astype('float32')
        telem_dat['Freq'] = telem_dat['Freq'].apply(lambda x: f"{x:.3f}")


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
            telem_dat['epoch'] = np.round((telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
            
            # drop unnecessary columns 
            telem_dat.drop (['Date','Time','Freq','Code','Site'],axis = 1, inplace = True)
            
            # calculate noise ratio
            telem_dat['noise_ratio'] = predictors.noise_ratio(5.0,
                                                   telem_dat.freq_code.values,
                                                   telem_dat.epoch.values,
                                                   study_tags)
            
            # if there is no antenna to receiver dictionary 
            if ant_to_rec_dict == None:
                # drop the antenna column - we don't need it anymore
                telem_dat.drop(['Ant'], axis = 1, inplace = True)
                
                # add receiver id 
                telem_dat['rec_id'] = np.repeat(rec_id,len(telem_dat))

                telem_dat = telem_dat.astype({'power':'float32',
                                              'freq_code':'object',
                                              'time_stamp':'datetime64[ns]',
                                              'scan_time':'float32',
                                              'channels':'int32',
                                              'rec_type':'object',
                                              'epoch':'float32',
                                              'noise_ratio':'float32',
                                              'rec_id':'object'})
                
                telem_dat = telem_dat[['power', 
                                        'time_stamp',
                                        'epoch',
                                        'freq_code',
                                        'noise_ratio',
                                        'scan_time',
                                        'channels', 
                                        'rec_id',
                                        'rec_type']]
                
                with pd.HDFStore(db_dir, mode='a') as store:
                    store.append(key = 'raw_data',
                                 value = telem_dat, 
                                 format = 'table', 
                                 index = False, 
                                 min_itemsize = {'freq_code':20,
                                                 'rec_type':20,
                                                 'rec_id':20},
                                 append = True, 
                                 chunksize = 1000000,
                                 data_columns = True)  
                
            # if there is an antenna to receiver dictionary
            else:
                for i in ant_to_rec_dict.keys():
                    # get site from dictionary
                    site = ant_to_rec_dict[i]
                    
                    # get telemetryt data associated with this site
                    telem_dat_sub = telem_dat[telem_dat.Ant == 1]
                    
                    # add receiver ID
                    telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                    
                    # remove exctranneous columns
                    telem_dat_sub.drop(['Ant'], axis = 1, inplace = True)
                    
                    telem_dat_sub = telem_dat_sub.astype({'power':'float32',
                                                          'freq_code':'object',
                                                          'time_stamp':'datetime64[ns]',
                                                          'scan_time':'float32',
                                                          'channels':'int32',
                                                          'rec_type':'object',
                                                          'epoch':'float32',
                                                          'noise_ratio':'float32',
                                                          'rec_id':'object'})
                    
                    telem_dat_sub = telem_dat_sub[['power',
                                                   'time_stamp',
                                                   'epoch', 
                                                   'freq_code', 
                                                   'noise_ratio',
                                                   'scan_time', 
                                                   'channels',
                                                   'rec_id',
                                                   'rec_type']]
                    
                    with pd.HDFStore(db_dir, mode='a') as store:
                        store.append(key = 'raw_data',
                                     value = telem_dat_sub, 
                                     format = 'table', 
                                     index = False, 
                                     min_itemsize = {'freq_code':20,
                                                     'rec_type':20,
                                                     'rec_id':20},
                                     append = True, 
                                     chunksize = 1000000,
                                     data_columns = True)  
    else:
        raise ValueError("Invalid import parameters, no data returned")
        sys.exit()
        
                    
def vr2_import(file_name,db_dir,study_tags, rec_id):
    """
    Import Vemco VR2 acoustic receiver data into MAST HDF5 database.
    
    Parses CSV format detection files from Vemco VR2 acoustic receivers.
    VR2 data uses acoustic tags instead of radio frequencies, with different
    field names and data structure.
    
    Parameters
    ----------
    file_name : str
        Absolute path to VR2 CSV file
    db_dir : str
        Absolute path to project HDF5 database
    study_tags : list of str
        List of valid acoustic tag codes deployed in study
    rec_id : str
        Unique receiver identifier
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Acoustic receivers use different schema than radio receivers
    - VR2 files typically have standardized CSV format from Vemco software
    - Converts acoustic tag IDs to freq_code format for consistency
    
    Examples
    --------
    >>> parsers.vr2_import(
    ...     file_name='C:/data/vr2_001.csv',
    ...     db_dir='C:/project/acoustic_study.h5',
    ...     study_tags=['A69-1601-12345', 'A69-1601-12346'],
    ...     rec_id='VR2_001'
    ... )
    
    See Also
    --------
    ares : Parser for radio telemetry receivers
    """

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
        telem_dat['epoch'] = np.round((telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
        try:
            telem_dat.drop (['Date and Time (UTC)', 'Transmitter Name','Transmitter Serial','Sensor Value','Sensor Unit','Station Name','Latitude','Longitude','Transmitter Type','Sensor Precision'],axis = 1, inplace = True)
        except KeyError:
            telem_dat.drop (['Unnamed: 0', 'Date and Time', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Receiver.1', 'Unnamed: 7', 'Unnamed: 8', 'Transmitter Name', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',],axis = 1, inplace = True)

        # tuples = zip(telem_dat.FreqCode.values,telem_dat.recID.values,telem_dat.Epoch.values)
        # index = pd.MultiIndex.from_tuples(tuples, names=['freq_code', 'rec_id','Epoch'])
        # telem_dat.set_index(index,inplace = True,drop = False)
        
        telem_dat = telem_dat.astype({'power':'float32',
                                      'freq_code':'object',
                                      'time_stamp':'datetime64[ns]',
                                      'scan_time':'float32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'epoch':'float32',
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
             ant_to_rec_dict = None,
             ka_format = False):
    """
    Import Lotek SRX-1200 receiver data into MAST HDF5 database.
    
    Parses fixed-width format detection files from Lotek SRX-1200 receivers.
    Supports both standard Lotek format and custom Kleinschmidt Associates (KA) format.
    
    Parameters
    ----------
    file_name : str
        Absolute path to SRX-1200 fixed-width text file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique receiver identifier (e.g., 'SRX1200_001')
    study_tags : list of str
        List of valid freq_code tags deployed in study
    scan_time : float, optional
        Scan duration per channel in seconds (default: 1.0)
    channels : int, optional
        Number of receiver channels (default: 1)
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs for multi-antenna setups
    ka_format : bool, optional
        If True, parse Kleinschmidt Associates custom format (default: False)
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Fixed-width column parsing optimized for SRX-1200 output
    - Handles multi-antenna configurations via ant_to_rec_dict
    - KA format includes additional metadata fields
    - Power values typically in dB
    
    Examples
    --------
    >>> parsers.srx1200(
    ...     file_name='C:/data/srx1200_site1.txt',
    ...     db_dir='C:/project/study.h5',
    ...     rec_id='SRX1200_SITE1',
    ...     study_tags=['166.380 7', '166.380 12'],
    ...     scan_time=2.5,
    ...     channels=1,
    ...     ka_format=False
    ... )
    
    See Also
    --------
    srx800 : Parser for SRX-800 receivers
    srx600 : Parser for SRX-600 receivers
    ares : Parser for ARES receivers
    """
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
            
        # get setups, scan times, firmware, and active scan tables
        setups = {}
        scan_times = {}
        firmware = ''
        setup_no = 0
        scan_channel = []
        scan_table_setups = {}
        scan_table_number = 0
        scan_time_number = 0
        new_split = None
        for row in header_df.iterrows():
            if 'SCAN SETTINGS' in row[1][0]:
                next_row = header_df.loc[row[0]+1]
                scanTimeStr = next_row.loc['line'][-7:-1]
                scanTimeSplit = scanTimeStr.split(':')
                if 'sec' in scanTimeSplit[1]:
                    new_split = scanTimeSplit[1].split(' ')
                    scanTimeSplit[1] = new_split[8]
                else:
                    scan_time = float(scanTimeSplit[1])
                    scan_times[scan_time_number] = scan_time
                    scan_time_number = scan_time_number + 1
                    
            if 'Environment History:' in row[1][0]:
                next_row = header_df.loc[row[0]+1]
                change_str = next_row.loc['line']
                setups[setup_no] = change_str[8:-1]
                setup_no = setup_no + 1
            
            if 'Master Firmware:' in row[1][0]:
                firmwarestr = row[1][0]
                firmware_split = firmwarestr.split(' ')
                firmware = str(firmware_split[-1]).lstrip().rstrip()
                
            if 'Active scan_table:' in row[1][0]:
                idx0 = row[0] + 2
                row_count = 1
                next_row = header_df.loc[row[0] + 1]
                while next_row[0] != '\n':
                    row_count = row_count + 1
                    next_row = header_df.loc[row[0] + row_count]
                    
                idx1 = idx0 + row_count - 2
                scan_table_setups[scan_table_number] = (idx0,idx1)
                scan_table_number = scan_table_number + 1
                
        scan_times_df = pd.DataFrame.from_dict(scan_times,
                                               orient = 'index',
                                               columns = ['scan_time'])
        scan_times_df.reset_index(inplace = True, drop = False)
        scan_times_arr = np.column_stack((scan_times_df.index, scan_times_df.scan_time))
        
        setup_df = pd.DataFrame.from_dict(setups, 
                                          orient = 'index', 
                                          columns = ['change_date'])
        setup_df['change_date'] = pd.to_datetime(setup_df.change_date)
        setup_df['epoch'] = (setup_df.change_date - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')        
        setup_df.reset_index(inplace = True, drop = False)
        setup_df.rename(columns = {'index':'setup'}, inplace = True)
    
        # extract channel dictionary data using rows identified 
        channel_dict = {}
        idx = 0
        for setup in scan_table_setups:
            idx0 = scan_table_setups[setup][0]
            idx1 = scan_table_setups[setup][1]
            channel_dat = header_df.iloc[idx0:idx1]
            for row in channel_dat.iterrows():
                dat = row[1][0]
                channel = str(dat[0:4]).lstrip()
                frequency = dat[10:17]
                channel_dict[idx] = [setup, channel, frequency]
                idx = idx + 1
            
        freq_scan_df = pd.DataFrame.from_dict(channel_dict, 
                                              orient = 'index', 
                                              columns = ['setup','channel','frequency'])
        # convert to numpy for vectorizing 
        freq_scan_arr = np.column_stack((freq_scan_df.setup.values,
                                         freq_scan_df.channel.values,
                                         freq_scan_df.frequency.values))
        
        print ('parsing SRX1200 header complete')

        # read in telemetry data
        if new_split == None:
            if ka_format == False:
                telem_dat = pd.read_fwf(file_name,
                                       colspecs = [(0,7),(7,25),(25,35),(35,46),(46,57),(57,68),(68,80),(80,90),(90,102),(102,110),(110,130),(130,143),(143,153)],
                                       names = ['Index','Rx Serial Number','Date','Time','[uSec]','Tag/BPM','Freq [MHz]','Codeset','Antenna','Gain','RSSI','Latitude','Longitude'],
                                       skiprows = dataRow, 
                                       skipfooter = eof - dataEnd)
                telem_dat.drop(columns = ['Index'], inplace = True)
            else:
                telem_dat = pd.read_fwf(file_name,
                                       colspecs = [(0,5),(6,20),(20,32),(32,43),(43,53),(53,65),(65,72),(72,85),(85,93),(93,101)],
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
        telem_dat['epoch'] = (telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
        
        # format frequency code
        telem_dat['FreqNo'] = telem_dat['Freq [MHz]'].apply(lambda x: f"{x:.3f}" )
        telem_dat = telem_dat[telem_dat['Tag/BPM'] != 999]
        telem_dat['freq_code'] = telem_dat['FreqNo'] + ' ' + telem_dat['Tag/BPM'].astype(str)
        
        # calculate 
        telem_dat['noise_ratio'] = predictors.noise_ratio(600,
                                              telem_dat.freq_code.values,
                                              telem_dat.epoch.values,
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
                                      'time_stamp':'datetime64[ns]',
                                      'scan_time':'float32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'epoch':'float32',
                                      'noise_ratio':'float32',
                                      'rec_id':'object'})
        
        telem_dat = telem_dat[['power', 
                                'time_stamp',
                                'epoch',
                                'freq_code',
                                'noise_ratio',
                                'scan_time',
                                'channels', 
                                'rec_id',
                                'rec_type']]
        
        # Write the DataFrame to the HDF5 file without the index
        with pd.HDFStore(db_dir, mode='a') as store:
            store.append(key='raw_data',
                         value=telem_dat,
                         format='table',
                         index=False,  # Ensure index is not written
                         min_itemsize={'freq_code': 20,
                                       'rec_type': 20,
                                       'rec_id': 20},
                         append=True,
                         chunksize=1000000,
                         data_columns=True)
        
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
        telem_dat['epoch'] = np.round((telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
                
        # format frequency code
        telem_dat['FreqNo'] = telem_dat['Freq [MHz]'].apply(lambda x: f"{x:.3f}" )
        telem_dat = telem_dat[telem_dat['TagID/BPM'] != 999]

        telem_dat['freq_code'] = telem_dat['FreqNo'] + ' ' + telem_dat['TagID/BPM'].astype(str)
        
        # calculate 
        telem_dat['noise_ratio'] = predictors.noise_ratio(600,
                                              telem_dat.freq_code.values,
                                              telem_dat.epoch.values,
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
                                      'time_stamp':'datetime64[ns]',
                                      'scan_time':'float32',
                                      'channels':'int32',
                                      'rec_type':'object',
                                      'epoch':'float32',
                                      'noise_ratio':'float32',
                                      'rec_id':'object'})
        
        telem_dat = telem_dat[['power', 
                                'time_stamp',
                                'epoch',
                                'freq_code',
                                'noise_ratio',
                                'scan_time',
                                'channels', 
                                'rec_id',
                                'rec_type']]
        
        # Write the DataFrame to the HDF5 file without the index
        with pd.HDFStore(db_dir, mode='a') as store:
            store.append(key='raw_data',
                         value=telem_dat,
                         format='table',
                         index=False,  # Ensure index is not written
                         min_itemsize={'freq_code': 20,
                                       'rec_type': 20,
                                       'rec_id': 20},
                         append=True,
                         chunksize=1000000,
                         data_columns=True)
            
def srx800(file_name,
             db_dir,
             rec_id, 
             study_tags, 
             scan_time = 1, 
             channels = 1, 
             ant_to_rec_dict = None):
    """
    Import Lotek SRX-800 receiver data into MAST HDF5 database.
    
    Parses fixed-width format detection files from Lotek SRX-800 receivers.
    Similar to SRX-1200 but with different column widths and firmware-specific
    header parsing.
    
    Parameters
    ----------
    file_name : str
        Absolute path to SRX-800 fixed-width text file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique receiver identifier
    study_tags : list of str
        List of valid freq_code tags deployed in study
    scan_time : float, optional
        Scan duration per channel in seconds (default: 1.0)
    channels : int, optional
        Number of receiver channels (default: 1)
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Parses SRX-800 specific header format for scan configuration
    - Fixed-width column parsing adjusted for SRX-800 output
    - Handles multi-antenna configurations
    
    Examples
    --------
    >>> parsers.srx800(
    ...     file_name='C:/data/srx800_detections.txt',
    ...     db_dir='C:/project/study.h5',
    ...     rec_id='SRX800_001',
    ...     study_tags=['166.380 7'],
    ...     scan_time=2.0,
    ...     channels=1
    ... )
    
    See Also
    --------
    srx1200 : Parser for SRX-1200 receivers
    srx600 : Parser for SRX-600 receivers
    """
    
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

            # else we are still reading header data increase the line counter by 1
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
    headerDF = pd.DataFrame.from_dict(header_dat)
    headerDF.set_index('idx',inplace = True)
    
    data_format = 'primary'
    if 'Environment History:' in headerDF.at[0,'line']:
        data_format = 'alternate'

    # get setups, scan times, firmware, and active scan tables
    setups = {}
    scan_times = {}
    firmware = ''
    setup_no = 0
    scan_channel = []
    scan_table_setups = {}
    scan_table_number = 0
    scan_time_number = 0
    for row in headerDF.iterrows():
        if 'Environment History:' in row[1][0]:
            next_row = headerDF.loc[row[0]+1]
            change_str = next_row.loc['line']
            setups[setup_no] = change_str[8:-1]
            setup_no = setup_no + 1
        
        if 'Master Firmware:' in row[1][0]:
            firmwarestr = row[1][0]
            firmware_split = firmwarestr.split(' ')
            firmware = str(firmware_split[-1]).lstrip().rstrip()
            
        if 'SCAN SETTINGS:' in row[1][0]:
            next_row = headerDF.loc[row[0]+1]
            scanTimeStr = next_row.loc['line'][-7:-1]
            scanTimeSplit = scanTimeStr.split(':')
            scan_time = float(scanTimeSplit[1])
            scan_times[scan_time_number] = scan_time
            scan_time_number = scan_time_number + 1  
            
        if 'scan time' in row[1][0]:
            scanTimeStr = row[1]['line'][-7:-1]
            scanTimeSplit = scanTimeStr.split(':')
            scan_time = float(scanTimeSplit[1])
            scan_times[scan_time_number] = scan_time
            scan_time_number = scan_time_number + 1             
            
        if 'Active scan_table:' in row[1][0]:
            idx0 = row[0] + 2
            row_count = 1
            next_row = headerDF.loc[row[0] + 1]
            while next_row[0] != '\n':
                row_count = row_count + 1
                next_row = headerDF.loc[row[0] + row_count]
                
            idx1 = idx0 + row_count - 2
            scan_table_setups[scan_table_number] = (idx0,idx1)
            scan_table_number = scan_table_number + 1
            
    scan_times_df = pd.DataFrame.from_dict(scan_times,
                                           orient = 'index',
                                           columns = ['scan_time'])
    scan_times_df.reset_index(inplace = True, drop = False)
    scan_times_arr = np.column_stack((scan_times_df.index, scan_times_df.scan_time))
    
    setup_df = pd.DataFrame.from_dict(setups, 
                                      orient = 'index', 
                                      columns = ['change_date'])
    if data_format == 'alternate':
        split = setup_df['change_date'].str.split(' ', expand=True)
        setup_df['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(setup_df))
        setup_df['Date'] = setup_df['day0'] + pd.to_timedelta(split[1].astype(int), unit='d')
        setup_df['change_date'] = setup_df.Date.astype(str) + ' ' + split[2]
        

    setup_df['change_date'] = pd.to_datetime(setup_df.change_date)
    setup_df['epoch'] = (setup_df['change_date'] - datetime.datetime(1970,1,1)).dt.total_seconds()
    setup_df.reset_index(inplace = True, drop = False)
    setup_df.rename(columns = {'index':'setup'}, inplace = True)

    # extract channel dictionary data using rows identified 
    channel_dict = {}
    idx = 0
    for setup in scan_table_setups:
        idx0 = scan_table_setups[setup][0]
        idx1 = scan_table_setups[setup][1]
        channel_dat = headerDF.iloc[idx0:idx1]
        for row in channel_dat.iterrows():
            dat = row[1][0]
            channel = str(dat[0:4]).lstrip()
            frequency = dat[10:17]
            channel_dict[idx] = [setup, channel, frequency]
            idx = idx + 1
        
    freq_scan_df = pd.DataFrame.from_dict(channel_dict, 
                                          orient = 'index', 
                                          columns = ['setup','channel','frequency'])
    # convert to numpy for vectorizing 
    freq_scan_arr = np.column_stack((freq_scan_df.setup.values,
                                     freq_scan_df.channel.values,
                                     freq_scan_df.frequency.values))
    
    print ('parsing SRX800 header complete')
    
    # with our data row, extract information using pandas fwf import procedure
    if firmware == '9.12.5':
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,8),(8,23),(23,33),(33,41),(41,56),(56,64)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})
    elif data_format == 'alternate':
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,5),(5,14),(14,23),(23,31),(31,46),(46,53)],
                               names = ['DayNumber','Time','ChannelID','TagID','Antenna','power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})
        telem_dat = telem_dat.iloc[:-1]
        telem_dat['day0'] = np.repeat(pd.to_datetime("1900-01-01"),len(telem_dat))
        telem_dat['Date'] = telem_dat['day0'] + pd.to_timedelta(telem_dat['DayNumber'].astype(int), unit='d')
        telem_dat['Date'] = telem_dat.Date.astype('str')
        telem_dat.drop(columns = ['DayNumber','day0'], inplace = True)
    else:
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,8),(8,18),(18,28),(28,36),(36,51),(51,59)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','power'],
                               skiprows = dataRow,
                               dtype = {'ChannelID':str,'TagID':str,'Antenna':str})

    telem_dat.dropna(inplace = True)
    print ('data import complete')
    
    # create some vectorizable functions that get setups, frequencies, channels, and scan times
    def setup_class(curr_date, setups):
        # Find indices where the condition is True
        indices = np.where(curr_date > setups)
        
        # Get the last index that meets the condition
        last_index = indices[0][-1] if indices[0].size > 0 else setups.shape[0]-1

        return last_index
    
    get_setup = np.vectorize(setup_class, excluded = [1])
    
    def freq_class(curr_setup, curr_channel, freq_scan_arr):
        # Condition: input_value1 > column 1 AND input_value2 < column 2
        condition = (freq_scan_arr[:, 0] == curr_setup) & (freq_scan_arr[:, 1] == curr_channel)
        
        # Find indices where the condition is True
        indices = np.where(condition)
        
        freq = freq_scan_arr[indices][:, 2][0]
        return freq
            
    get_frequency = np.vectorize(freq_class, excluded = [2])
    
    def count_channels(curr_setup, freq_scan_arr):
        condition = freq_scan_arr[:, 0] == curr_setup
        channels = condition.sum()
        return channels
    
    get_channels = np.vectorize(count_channels, excluded = [1])
    
    def scan_time_class(curr_setup, scan_times_arr):
        condition = scan_times_arr[:, 0] == curr_setup
        indices = np.where(condition)
        scan_time = scan_times_arr[indices][:, 1][0]
        return scan_time
        
    get_scan_time = np.vectorize(scan_time_class, excluded = [1])
    
    if len(telem_dat) > 0:
        for ant in ant_to_rec_dict:
            site = ant_to_rec_dict[ant]
            telem_dat_sub = telem_dat[telem_dat.Antenna == ant]
            
            # create timestamp
            telem_dat_sub['time_stamp'] = pd.to_datetime(telem_dat_sub['Date'] + ' ' + telem_dat_sub['Time'])
            
            # get UNIX epoch
            telem_dat_sub['epoch'] = np.round((telem_dat_sub.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
            
            # get setup number for every row
            try:
                telem_dat_sub['setup'] = get_setup(telem_dat_sub.epoch.values,
                                                   setup_df.epoch.values)
            except:
                print ('why you fail?')
            
            # get frequency from channel
            telem_dat_sub['Frequency'] = get_frequency(telem_dat_sub.setup.values,
                                                       telem_dat_sub.ChannelID.values,
                                                       freq_scan_arr)

            # remove extraneous data
            telem_dat_sub = telem_dat_sub[telem_dat_sub.Frequency != '888']
            #telemDat_sub = telemDat_sub[telemDat_sub.TagID != 999]
            
            # create FreqCode
            telem_dat_sub['freq_code'] = telem_dat_sub['Frequency'].astype(str) + ' ' + telem_dat_sub['TagID'].astype(int).astype(str)
            
            if np.any(pd.isnull(telem_dat_sub.Frequency.values)):
                print ('debug - why is frequency not working?')
            
            # calculate noise ratio
            telem_dat_sub['noise_ratio'] = predictors.noise_ratio(5.0,
                                                                  telem_dat_sub.freq_code,
                                                                  telem_dat_sub.epoch,
                                                                  study_tags)
            
            # write scan time, channels, rec type and recID to data
            telem_dat_sub['scan_time'] = get_scan_time(telem_dat_sub.setup.values, scan_times_arr)
            telem_dat_sub['channels'] = get_channels(telem_dat_sub.setup.values,freq_scan_arr)
            telem_dat_sub['scan_time'] = np.where(telem_dat_sub.channels == 1, 1, scan_time)
            telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
            telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))

            
            # remove extraneous columns
            telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna','setup'],axis = 1, inplace = True)
            telem_dat_sub.reset_index(inplace = True)
            
            # write to hdf
            telem_dat_sub = telem_dat_sub.astype({'power':'float32',
                                                  'freq_code':'object',
                                                  'time_stamp':'datetime64[ns]',
                                                  'scan_time':'float32',
                                                  'channels':'int32',
                                                  'rec_type':'object',
                                                  'epoch':'float32',
                                                  'noise_ratio':'float32',
                                                  'rec_id':'object'})
                
            telem_dat_sub.reset_index(inplace = True, drop = True)
            
            telem_dat_sub = telem_dat_sub[['power', 
                                           'time_stamp',
                                           'epoch',
                                           'freq_code',
                                           'noise_ratio',
                                           'scan_time',
                                           'channels', 
                                           'rec_id',
                                           'rec_type']]
            
            telem_dat_sub.to_hdf(db_dir,
                                 key = 'raw_data',
                                 mode = 'a',
                                 append = True,
                                 format = 'table',
                                 index = False,
                                 min_itemsize = {'freq_code':20,
                                                 'rec_type':20,
                                                 'rec_id':20},
                                 data_columns = True)
            

            print ('Data standardized and exported to hdf')

def srx600(file_name,
             db_dir,
             rec_id, 
             study_tags, 
             scan_time = 1, 
             channels = 1, 
             ant_to_rec_dict = None):
    """
    Import Lotek SRX-600 receiver data into MAST HDF5 database.
    
    Parses fixed-width format detection files from Lotek SRX-600 receivers.
    Similar to SRX-800/1200 but with SRX-600 specific column widths and
    header structure.
    
    Parameters
    ----------
    file_name : str
        Absolute path to SRX-600 fixed-width text file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique receiver identifier
    study_tags : list of str
        List of valid freq_code tags deployed in study
    scan_time : float, optional
        Scan duration per channel in seconds (default: 1.0)
    channels : int, optional
        Number of receiver channels (default: 1)
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Parses SRX-600 specific header format
    - Fixed-width column parsing adjusted for SRX-600 output
    - Older receiver model with slightly different data structure
    
    Examples
    --------
    >>> parsers.srx600(
    ...     file_name='C:/data/srx600_detections.txt',
    ...     db_dir='C:/project/study.h5',
    ...     rec_id='SRX600_001',
    ...     study_tags=['166.380 7'],
    ...     scan_time=1.5,
    ...     channels=1
    ... )
    
    See Also
    --------
    srx1200 : Parser for SRX-1200 receivers
    srx800 : Parser for SRX-800 receivers
    """
    
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
                telem_dat['epoch'] = np.round((telem_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
                
                # clean up some more
                telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                
                # add file name to data
                #telemDat_sub['fileName'] = np.repeat(rxfile,len(telemDat_sub))
                
                # add recID
                telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                
                # calculate noise ratio
                telem_dat_sub = predictors.noise_ratio(600,
                                           telem_dat_sub.freq_code.values,
                                           telem_dat_sub.epoch.values,
                                           study_tags)
                
                # add in scan time, channels, rec type
                telem_dat_sub['scan_time'] = np.repeat(scan_time,len(telem_dat_sub))
                telem_dat_sub['channels'] = np.repeat(channels,len(telem_dat_sub))
                telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))
                
                # make multiindex for fast indexing
                # tuples = zip(telem_dat_sub.FreqCode.values,telem_dat_sub.recID.values,telem_dat_sub.Epoch.values)
                # index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','epoch'])
                # telem_dat_sub.set_index(index,inplace = True,drop = False)
                telem_dat_sub.reset_index(inplace = True)
                telem_dat_sub['noise_ratio'] = telem_dat_sub.noise_ratio.values.astype(np.float32)

                # write to SQL
                
                telem_dat_sub = telem_dat_sub.astype({'power':'float32',
                                                      'freq_code':'object',
                                                      'time_stamp':'datetime64[ns]',
                                                      'scan_time':'float32',
                                                      'channels':'int32',
                                                      'rec_type':'object',
                                                      'epoch':'float32',
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
                                 chunksize = 1000000,
                                 data_columns = True)    
    else:
        telem_dat = pd.read_fwf(file_name,
                               colspecs = [(0,9),(9,19),(19,29),(29,36),(36,44),(44,52)],
                               names = ['Date','Time','ChannelID','TagID','Antenna','Power'],
                               skiprows = dataRow)
        
        # remove last two rows, Lotek adds garbage at the end
        telem_dat = telem_dat.iloc[:-2]
        telem_dat.dropna(inplace = True)
        
        if len(telem_dat) > 0:
            for ant in ant_to_rec_dict:
                site = ant_to_rec_dict[ant]
                telem_dat_sub = telem_dat[telem_dat.Antenna == ant]
                
                # calculate frequency with channel ID
                telem_dat_sub['Frequency'] = telem_dat_sub['ChannelID'].map(channelDict)
                
                # remove extra data
                telem_dat_sub = telem_dat_sub[telem_dat_sub.Frequency != '888']
                telem_dat_sub = telem_dat_sub[telem_dat_sub.TagID != 999]
                
                
                # calculate freqcode
                telem_dat_sub['freq_code'] = telem_dat_sub['Frequency'].astype(str) + ' ' + telem_dat_sub['TagID'].astype(int).astype(str)
                
                # get timestamp
                telem_dat_sub['time_stamp'] = pd.to_datetime(telem_dat_sub['Date'] + ' ' + telem_dat_sub['Time'])# create timestamp field from date and time and apply to index
                
                # calculate UNIX epoch 
                telem_dat_sub['epoch'] = np.round((telem_dat_sub.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
                
                # calculate noise ratio
                telem_dat_sub = predictors.noise_ratio(5.0,
                                           telem_dat_sub.freq_code.values,
                                           telem_dat_sub.epoch.values,
                                           study_tags)
                
                # clean up some more
                telem_dat_sub.drop (['Date','Time','Frequency','TagID','ChannelID','Antenna'],axis = 1, inplace = True)
                
                # add in scan time, channels, rec type
                telem_dat_sub['scan_time'] = np.repeat(scan_time,len(telem_dat_sub))
                telem_dat_sub['channels'] = np.repeat(channels,len(telem_dat_sub))
                telem_dat_sub['rec_type'] = np.repeat(rec_type,len(telem_dat_sub))
                telem_dat_sub['rec_id'] = np.repeat(site,len(telem_dat_sub))
                
                # make multiindex for fast indexing
                # tuples = zip(telem_dat_sub.FreqCode.values,telem_dat_sub.recID.values,telem_dat_sub.epoch.values)
                # index = pd.MultiIndex.from_tuples(tuples, names=['FreqCode', 'recID','epoch'])
                # telem_dat_sub.set_index(index,inplace = True,drop = False)
                telem_dat_sub.reset_index(inplace = True)
                telem_dat_sub['noise_ratio'] = telem_dat.noise_ratio.values.astype(np.float32)

                telem_dat_sub = telem_dat_sub.astype({'power':'float32',
                                                      'freq_code':'object',
                                                      'time_stamp':'datetime64[ns]',
                                                      'scan_time':'float32',
                                                      'channels':'int32',
                                                      'rec_type':'object',
                                                      'epoch':'float32',
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
    
    
    
    
    
def PIT(file_name,
        db_dir,
        rec_id=None,
        study_tags=None,
        skiprows=6,
        scan_time=0,
        channels=0,
        rec_type="PIT",
        ant_to_rec_dict=None):
    """
    Import PIT (Passive Integrated Transponder) reader data into MAST HDF5 database.
    
    Parses detection files from PIT tag readers. PIT systems use different
    technology (RFID) than radio telemetry but data can be analyzed with
    similar methods.
    
    Parameters
    ----------
    file_name : str
        Absolute path to PIT reader CSV/text file
    db_dir : str
        Absolute path to project HDF5 database
    rec_id : str
        Unique reader identifier
    study_tags : list of str
        List of valid PIT tag IDs deployed in study
    skiprows : int, optional
        Number of header rows to skip (default: 6)
    scan_time : float, optional
        Not used for PIT readers (default: 0)
    channels : int, optional
        Not used for PIT readers (default: 0)
    rec_type : str, optional
        Reader type identifier (default: 'PIT_Array')
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to reader IDs for multi-antenna arrays
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - PIT readers have antenna-based detection logic (different from radio receivers)
    - Tag IDs converted to freq_code format for consistency with radio data
    - Typically used at fixed locations (weirs, ladders, bypass systems)
    - scan_time and channels not applicable to PIT technology
    
    Examples
    --------
    >>> parsers.PIT(
    ...     file_name='C:/data/pit_reader_001.csv',
    ...     db_dir='C:/project/pit_study.h5',
    ...     rec_id='PIT_WEIR_01',
    ...     study_tags=['3D9.1BF3C5A8B2', '3D9.1BF3C5A8C1'],
    ...     skiprows=6,
    ...     rec_type='PIT_Array'
    ... )
    
    See Also
    --------
    PIT_Multiple : Parser for multi-antenna PIT arrays
    """
    
    import pandas as pd
    import re
    
    # Determine mode based on parameters
    is_multi_antenna = ant_to_rec_dict is not None
    mode_str = "multi-antenna" if is_multi_antenna else "single antenna"
    print(f"Parsing PIT file ({mode_str}): {file_name}")
    
    # Function to find columns by pattern matching
    def find_column_by_patterns(df, patterns):
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    # First, analyze the file to determine format
    def analyze_file_format(file_name):
        """Dynamically determine PIT file format and header structure"""
        with open(file_name, 'r') as file:
            lines = []
            for i in range(20):  # Read first 20 lines to analyze format
                try:
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line.rstrip('\n'))
                except:
                    break
        
        # Check if CSV format (look for commas in sample lines)
        csv_indicators = 0
        for line in lines[max(0, len(lines)-10):]:  # Check last 10 lines for data
            if line.count(',') > 3:  # More than 3 commas suggests CSV
                csv_indicators += 1
        
        is_csv = csv_indicators > 2  # If most lines have commas, it's CSV
        
        # For CSV, look for header row
        actual_skiprows = 0
        if is_csv:
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Look for column headers (must contain text headers, not just data)
                if any(header in line_lower for header in ['tag', 'time', 'date', 'antenna', 'detected', 'site', 'reader']):
                    if ',' in line and not re.search(r'^\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', line.strip()):
                        # It's a header row (has keywords but no date pattern at start)
                        actual_skiprows = 0  # Keep headers, don't skip them
                        break
            
            # If no header found, assume no header (skiprows = 0)
        else:
            # For fixed-width, look for data start
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if 'version' in line_lower or 'ver' in line_lower:
                    print(f"Found version info: {line}")
                
                # Look for data start indicators
                if any(indicator in line_lower for indicator in ['scan date', 'date', 'timestamp', 'tag id']):
                    if i > 0:  # If this looks like a header row
                        actual_skiprows = i + 1
                        break
                
                # Check if this looks like a data line
                if re.search(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', line):
                    actual_skiprows = i
                    break
        
        return is_csv, actual_skiprows, lines

    # Analyze file format
    is_csv_format, detected_skiprows, sample_lines = analyze_file_format(file_name)
    
    # Use detected skiprows for CSV, keep provided for fixed-width
    if is_csv_format:
        skiprows = detected_skiprows
        print(f"Detected CSV format, using skiprows: {skiprows}")
    else:
        print(f"Detected fixed-width format, using skiprows: {skiprows}")

    # Parse the file based on detected format
    if is_csv_format:
        # CSV Format Parsing
        try:
            # Read CSV - if skiprows is 0, pandas will automatically use first row as headers
            telem_dat = pd.read_csv(file_name, dtype=str)
            print(f"Auto-detected columns: {list(telem_dat.columns)}")
            
        except Exception as e:
            print(f"CSV auto-detection failed: {e}")
            # Fallback to predefined column names
            col_names = [
                "FishId", "Tag1Dec", "Tag1Hex", "Tag2Dec", "Tag2Hex", "FloyTag", "RadioTag",
                "Location", "Source", "FishSpecies", "TimeStamp", "Weight", "Length",
                "Antennae", "Latitude", "Longitude", "SampleDate", "CaptureMethod",
                "LocationDetail", "Type", "Recapture", "Sex", "GeneticSampleID", "Comments"
            ]
            telem_dat = pd.read_csv(file_name, names=col_names, header=0, skiprows=skiprows, dtype=str)

        # Find timestamp column dynamically
        timestamp_col = find_column_by_patterns(telem_dat, ['timestamp', 'time stamp', 'date', 'scan date', 'detected'])
        if timestamp_col:
            print(f"Found timestamp column: {timestamp_col}")
            # Try multiple datetime formats
            for fmt in ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d", None]:
                try:
                    if fmt:
                        telem_dat["time_stamp"] = pd.to_datetime(telem_dat[timestamp_col], format=fmt, errors="coerce")
                    else:
                        telem_dat["time_stamp"] = pd.to_datetime(telem_dat[timestamp_col], errors="coerce")
                    
                    # Check if parsing was successful
                    if not telem_dat["time_stamp"].isna().all():
                        print(f"Successfully parsed timestamps using format: {fmt or 'auto-detect'}")
                        break
                except:
                    continue
        else:
            raise ValueError("Could not find timestamp column")

        # Find tag ID columns dynamically
        hex_tag_col = find_column_by_patterns(telem_dat, ['hex', 'tag1hex', 'tag id', 'tagid', 'tag'])
        dec_tag_col = find_column_by_patterns(telem_dat, ['dec', 'tag1dec', 'decimal'])
        
        if hex_tag_col:
            print(f"Found HEX tag column: {hex_tag_col}")
            telem_dat["freq_code"] = telem_dat[hex_tag_col].astype(str).str.strip()
        elif dec_tag_col:
            print(f"Found DEC tag column: {dec_tag_col}")
            telem_dat["freq_code"] = telem_dat[dec_tag_col].astype(str).str.strip()
        else:
            raise ValueError("Could not find tag ID column")

        # Handle antenna mapping for multi-antenna files
        if is_multi_antenna:
            antenna_col = find_column_by_patterns(telem_dat, ['antenna', 'antennae', 'ant'])
            if antenna_col:
                print(f"Found antenna column: {antenna_col}")
                # Convert antenna column to integer and apply mapping
                telem_dat["antenna_clean"] = telem_dat[antenna_col].astype(str).str.extract(r'(\d+)')[0]
                telem_dat["antenna_clean"] = pd.to_numeric(telem_dat["antenna_clean"], errors='coerce').astype("Int64")
                telem_dat["rec_id"] = telem_dat["antenna_clean"].map(ant_to_rec_dict)
                # Drop rows where antenna values don't match known receivers
                telem_dat = telem_dat.dropna(subset=["rec_id"])
            else:
                raise ValueError("Multi-antenna mode requires antenna column, but none found")
        else:
            # Single antenna mode - use provided rec_id
            telem_dat["rec_id"] = rec_id

    else:
        # Fixed-Width Format Parsing (original logic)
        
        # Read header information for format detection
        with open(file_name, 'r') as file:
            header_lines = []
            for _ in range(max(skiprows, 10)):
                try:
                    line = file.readline()
                    if not line:
                        break
                    header_lines.append(line.rstrip('\n'))
                except:
                    break
            header_text = " ".join(header_lines).lower()

        # Define colspecs for different fixed-width formats
        if 'latitude' in header_text or 'longitude' in header_text:
            colspecs = [(0, 12), (12, 26), (26, 41), (41, 56), (56, 59), (66, 70), 
                       (79, 95), (95, 112), (113, 120), (120, 131), (138, 145), 
                       (145, 155), (155, 166), (166, 175)]
            col_names = ["Scan Date", "Scan Time", "Download Date", "Download Time", 
                        "Reader ID", "Antenna ID", "HEX Tag ID", "DEC Tag ID", 
                        "Temperature_C", "Signal_mV", "Is Duplicate", "Latitude", 
                        "Longitude", "File Name"]
            print("Using format with latitude/longitude")
        else:
            colspecs = [(0, 12), (12, 26), (26, 41), (41, 56), (56, 62), (62, 73), 
                       (73, 89), (89, 107), (107, 122), (122, 132), (136, 136)]
            col_names = ["Scan Date", "Scan Time", "Download Date", "Download Time", 
                        "S/N", "Reader ID", "HEX Tag ID", "DEC Tag ID", 
                        "Temperature_C", "Signal_mV", "Is Duplicate"]
            print("Using format without latitude/longitude")

        # Try different encodings if UTF-8 fails
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        telem_dat = None
        
        for encoding in encodings_to_try:
            try:
                print(f"Attempting to read file with encoding: {encoding}")
                telem_dat = pd.read_fwf(
                    file_name,
                    colspecs=colspecs,
                    names=col_names,
                    skiprows=skiprows,
                    encoding=encoding
                )
                print(f"Successfully read with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                print(f"Failed with {encoding}, trying next...")
                continue
        
        if telem_dat is None:
            raise ValueError(f"Could not read file with any supported encoding: {encodings_to_try}")
        
        print(f"Fixed-width parsing complete. Shape: {telem_dat.shape}")
        
        # Build timestamp from Scan Date + Scan Time
        telem_dat["time_stamp"] = pd.to_datetime(
            telem_dat["Scan Date"] + " " + telem_dat["Scan Time"],
            errors="coerce"
        )
        
        # Use HEX Tag ID as freq_code
        telem_dat["freq_code"] = telem_dat["HEX Tag ID"].str.strip()
        
        # For fixed-width, always use single antenna mode
        telem_dat["rec_id"] = rec_id

    # Data cleaning - remove invalid entries
    print(f"\nCleaning data - original records: {len(telem_dat)}")
    
    before_cleanup = len(telem_dat)
    
    # Remove header artifacts
    header_patterns = ['HEX Tag ID', 'DEC Tag ID', '----', '====', 'Tag ID', 'Scan Date']
    for pattern in header_patterns:
        telem_dat = telem_dat[telem_dat['freq_code'] != pattern]
    
    # Remove separator lines
    telem_dat = telem_dat[~telem_dat['freq_code'].str.match(r'^-+$', na=False)]
    
    # Remove rows with invalid timestamps
    telem_dat = telem_dat[~telem_dat['time_stamp'].isna()]
    
    # Remove rows with invalid freq_codes
    telem_dat = telem_dat[telem_dat['freq_code'].str.len() > 3]
    telem_dat = telem_dat[~telem_dat['freq_code'].isna()]
    
    with pd.HDFStore(db_dir, 'r') as store:
        print("Store keys after append:", store.keys())


def PIT_Multiple(
    file_name,
    db_dir,
    study_tags=None,
    skiprows=0,
    scan_time=0,
    channels=0,
    rec_type="PIT_Multiple",
    ant_to_rec_dict=None
):
    """
    Import multi-antenna PIT array data into MAST HDF5 database.
    
    Parses detection files from PIT reader arrays with multiple antennas at
    a single location. Handles antenna-to-receiver mapping and converts
    multi-antenna detections to individual receiver records.
    
    Parameters
    ----------
    file_name : str
        Absolute path to PIT array CSV file
    db_dir : str
        Absolute path to project HDF5 database
    study_tags : list of str, optional
        List of valid PIT tag IDs deployed in study
    skiprows : int, optional
        Number of header rows to skip (default: 0)
    scan_time : float, optional
        Not used for PIT readers (default: 0)
    channels : int, optional
        Not used for PIT readers (default: 0)
    rec_type : str, optional
        Reader type identifier (default: 'PIT_Multiple')
    ant_to_rec_dict : dict, optional
        Mapping of antenna IDs to receiver IDs (REQUIRED for multi-antenna arrays)
    
    Returns
    -------
    None
        Data appended directly to HDF5 `/raw_data` table
    
    Notes
    -----
    - Designed for PIT arrays with multiple antennas at single location
    - Uses ant_to_rec_dict to assign detections to virtual "receivers" per antenna
    - Processes fish metadata (species, weight, length, capture method)
    - Handles both decimal and hexadecimal tag formats
    
    Examples
    --------
    >>> ant_map = {
    ...     'Antenna1': 'PIT_WEIR_DOWNSTREAM',
    ...     'Antenna2': 'PIT_WEIR_UPSTREAM',
    ...     'Antenna3': 'PIT_WEIR_LADDER'
    ... }
    >>> parsers.PIT_Multiple(
    ...     file_name='C:/data/pit_array_detections.csv',
    ...     db_dir='C:/project/pit_study.h5',
    ...     study_tags=['3D9.1BF3C5A8B2'],
    ...     rec_type='PIT_Multiple',
    ...     ant_to_rec_dict=ant_map
    ... )
    
    See Also
    --------
    PIT : Parser for single PIT readers
    """
    # Define column names based on the expected structure of the CSV
    col_names = [
        "FishId", "Tag1Dec", "Tag1Hex", "Tag2Dec", "Tag2Hex", "FloyTag", "RadioTag",
        "Location", "Source", "FishSpecies", "TimeStamp", "Weight", "Length",
        "Antennae", "Latitude", "Longitude", "SampleDate", "CaptureMethod",
        "LocationDetail", "Type", "Recapture", "Sex", "GeneticSampleID", "Comments"
    ]

    # Read the CSV into a DataFrame, skipping rows if needed
    telem_dat = pd.read_csv(file_name, names=col_names, header=0, skiprows=skiprows, dtype=str)

    # Convert "TimeStamp" to datetime with explicit format
    telem_dat["time_stamp"] = pd.to_datetime(telem_dat["TimeStamp"], format="%m/%d/%Y %H:%M", errors="coerce")

    # Ensure "Tag1Dec" and "Tag1Hex" are treated as strings (avoid scientific notation issues)
    telem_dat["Tag1Dec"] = telem_dat["Tag1Dec"].astype(str)
    telem_dat["Tag1Hex"] = telem_dat["Tag1Hex"].astype(str)

    if after_cleanup == 0:
        raise ValueError(f"No valid records found in {file_name}")

    # Standardize columns
    telem_dat["power"] = 0.0
    telem_dat["noise_ratio"] = 0.0
    telem_dat["scan_time"] = scan_time
    telem_dat["channels"] = channels
    telem_dat["rec_type"] = rec_type

    # Calculate epoch time
    telem_dat["epoch"] = (
        telem_dat["time_stamp"] - pd.Timestamp("1970-01-01")
    ) / pd.Timedelta("1s")

    # Convert to standard data types
    telem_dat = telem_dat.astype({
        "power": "float32",
        "freq_code": "object",
        "time_stamp": "datetime64[ns]",
        "scan_time": "float32",
        "channels": "int32",
        "rec_type": "object",
        "epoch": "float32",
        "noise_ratio": "float32",
        "rec_id": "object"
    })

    # Keep only standard columns
    telem_dat = telem_dat[
        ["power", "time_stamp", "epoch", "freq_code", "noise_ratio",
         "scan_time", "channels", "rec_id", "rec_type"]
    ]

    # Append to HDF5 store
    with pd.HDFStore(db_dir, mode='a') as store:
        store.append(
            key="raw_data",
            value=telem_dat,
            format="table",
            index=False,
            min_itemsize={"freq_code": 20, "rec_type": 20, "rec_id": 20},
            append=True,
            chunksize=1000000,
            data_columns=True
        )

    print(f"\nSuccessfully parsed {file_name} and appended to {db_dir}!")
    print(f"Imported {len(telem_dat)} records in {mode_str} mode")

    with pd.HDFStore(db_dir, 'r') as store:
        print("Store keys after append:", store.keys())





 
    