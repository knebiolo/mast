# -*- coding: utf-8 -*-
"""
Modules contains all of the functions and classes required to format radio
telemetry data for statistical testing.

currently we can set up models for Cormack Jolly Seber mark recapture, Time-to-
Event modeling, and Live Recapture Dead Recovery Mark Recapture.
"""

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
        sql = 'SELECT * FROM tblRecaptures WHERE tblRecaptures.recID = "%s"'%(receiver_list[0])
        for i in receiver_list[1:]:
            sql = sql + ' OR tblRecaptures.recID = "%s"'%(i)
        self.data = pd.read_sql_query(sql,con = conn)
        rec_dat = pd.read_sql_query('SELECT recID, Node FROM tblMasterReceiver', con = conn)
        tag_dat = pd.read_sql_query('SELECT FreqCode, TagType, CapLoc, RelLoc FROM tblMasterTag', con = conn)
        self.data = self.data.merge(rec_dat,how = 'left', on = 'recID')
        self.data = self.data.merge(tag_dat,how = 'left', on = 'FreqCode')
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