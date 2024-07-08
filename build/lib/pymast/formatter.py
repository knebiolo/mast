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
from matplotlib import rcParams
from scipy import interpolate
from pymast.radio_project import radio_project


font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

class cjs_data_prep():
    '''Class creates input files for Cormack Jolly Seber modeling in MARK'''
    def __init__(self, 
                 receiver_to_recap, 
                 project, 
                 input_type = 'query',
                 species = None, 
                 rel_loc = None, 
                 cap_loc = None, 
                 initial_recap_release = False):
        
        # Import Data From Project HDF
        self.rel_loc = rel_loc
        self.cap_loc = cap_loc

        # get recaptures, but first build a query
        query_parts = []
        for key in receiver_to_recap:
            query_parts.append(f"rec_id == '{key}'")   
        qry = " & ".join(query_parts)

        recap_data = pd.read_hdf(project.db,
                             'recaptures',
                             where = qry)
        
        self.recap_data.set_index('freq_code', inplace = True)
        project.tags.set_index('freq_code', inplace = True)
        self.recap_data = pd.merge(recap_data, project.tags, how = 'left')
        self.recap_data.reset_index(drop = False, inplace = True)
        project.tags.reset_index(drop = False, inplace = True)
        
        # filter out tag data we don't want mucking up our staistical model
        if species != None:
            self.recap_data = self.recap_data[self.recap_data.Species == species]
        if rel_loc != None:
            self.recap_data = self.recap_data[self.recap_data.RelLoc == rel_loc]
        if cap_loc != None:
            self.recap_data = self.recap_data[self.recap_data.CapLoc == cap_loc]

        # map recap occasions
        self.recap_data['recap_occasion'] = self.recap_data['rec_id'].map(receiver_to_recap)

        if initial_recap_release == False:
            '''When initial recap release is true, the first recapture occasion
            in the rec_list query becomes our release location - this is helpful
            when modeling downstream migratory behavior of anadromous fish like
            American Shad.  We don't care about what happens on their upstream
            migration, we only care about when they turn around and come back down.

            When this argument is false, we are modeling fish from their initial
            release time and we add in R00 as our release location, otherwise R00
            should appear on the input receiver to recapture event dictionary.
            '''

            print ("Adding release time for this fish")
            rel_dat = project.tags
            if self.rel_loc is not None:
                rel_dat = rel_dat[rel_dat.rel_loc == rel_loc]
            if self.cap_loc is not None:
                rel_dat = rel_dat[rel_dat.cap_loc == cap_loc]
            rel_dat['rel_date'] = pd.to_datetime(rel_dat.rel_date)
            rel_dat['epoch'] = (rel_dat['rel_date'] - datetime.datetime(1970,1,1)).dt.total_seconds()
            rel_dat.rename(columns = {'rel_date':'time_stamp'}, inplace = True)
            rel_dat['recap_occasion'] = np.repeat('R00',len(rel_dat))
            rel_dat['overlapping'] = np.zeros(len(rel_dat))
            self.recap_data = pd.concat([self.recap_data,rel_dat],axis = 0)

        else:
            print ("Starting Initial Recap Release Procedure")
            # Identify first recapture times
            start_times = self.recap_data[self.recap_data.recap_occasion == "R00"].groupby(['freq_code'])['epoch'].min().to_frame()
            start_times.reset_index(drop = False, inplace = True)
            start_times.set_index('freq_code',inplace = True)
            start_times.rename(columns = {'epoch':'first_recapture'},inplace = True)

            # make sure fish start from the first recapture occassion and that there are no recaps before release
            for fish in self.recap_data.freq_code.unique():

                if fish not in start_times.index.values:
                    # fish never made it to the initial state
                    self.recap_dat.drop(self.recap_dat[self.recap_data.freq_code == fish].index, inplace = True)
                else:
                    # fish arrived at the initial state but their may be recaptures before arrival at initial state
                    t = start_times.at[fish,'first_recapture']
                    self.recap_data.drop(self.recap_data[(self.recap_data.freq_code == fish) & (self.recap_data.epoch < t)].index, inplace = True)

        print (self.recap_data.head())

    def input_file(self,model_name, output_ws):
        #Step 1: Create cross tabulated data frame with FreqCode as row index and recap occasion as column
        cross_tab = pd.pivot_table(self.recap_data, values = 'epoch', index = 'freq_code', columns = 'recap_occasion', aggfunc = 'min')

        #Step 2: Fill in those nan values with 0, we can't perform logic on nothing!'''
        cross_tab.fillna(value = 0, inplace = True)

        #Step 3: Map a simply if else statement, if value > 0,1 else 0'''
        cross_tab = cross_tab.applymap(lambda x:1 if x > 0 else 0)

        # put together the input string
        inp = "/* " + cross_tab.index.astype(str) + " */  "
        for i in np.arange(0,len(cross_tab.columns),1):
            inp = inp + cross_tab.iloc[:,i].astype(str)
        inp = inp + "     1;"

        self.inp = inp
        self.cross = cross_tab
        # Check your work
        print (cross_tab.head(100))
        cross_tab.to_csv(os.path.join(output_ws,'%s_cjs.csv'%(model_name)))

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


        c.close()
        print ("Finished sql")

        if initial_recap_release == False:
            '''When initial recap release is true, the first recapture occasion
            in the rec_list query becomes our release location - this is helpful
            when modeling downstream migratory behavior of anadromous fish like
            American Shad.  We don't care about what happens on their upstream
            migration, we only care about when they turn around and come back down.

            When this argument is false, we are modeling fish from their initial
            release time and we add in R00 as our release location, otherwise R00
            should appear on the input receiver to recapture event dictionary.
            '''
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
        startTimes = self.data[self.data.RecapOccasion == "R00"].groupby(['FreqCode'])['Epoch'].min().to_frame()
        startTimes.reset_index(drop = False, inplace = True)
        startTimes.set_index('FreqCode',inplace = True)
        #self.startTimes.Epoch = self.startTimes.Epoch.astype(np.int32)
        startTimes.rename(columns = {'Epoch':'FirstRecapture'},inplace = True)
        print(startTimes)
        for fish in self.data.FreqCode.unique():
            if fish not in startTimes.index.values:                  # fish never made it to the initial state, remove - we only care about movements from the initial sstate - this is a competing risks model
                self.data = self.data[self.data.FreqCode != fish]
            else:
                t = startTimes.at[fish,'FirstRecapture']
                print (t)
                self.data = self.data.drop(self.data[(self.data.FreqCode == fish) & (self.data.Epoch < t)].index, inplace = True)

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
    def __init__(self, 
                 receiver_to_state,
                 project, 
                 input_type = 'query', 
                 initial_state_release = False, 
                 last_presence_time0 = False, 
                 cap_loc = None,
                 rel_loc = None, 
                 species = None):
        # Import Data From Project HDF
        self.rel_loc = rel_loc
        self.cap_loc = cap_loc

        # get recaptures, but first build a query
        query_parts = []
        for key in receiver_to_state:
            query_parts.append(f"rec_id == '{key}'")   
        qry = " | ".join(query_parts)

        self.recap_data = pd.read_hdf(project.db,
                                      'recaptures',
                                      where = qry)
        
        self.recap_data.drop(columns = ['power',
                                        'noise_ratio',
                                        'det_hist',
                                        'hit_ratio',
                                        'cons_det',
                                        'cons_length',
                                        'likelihood_T',
                                        'likelihood_F'],
                             axis = 'columns',
                             inplace = True)
        
        self.recap_data.set_index('freq_code', inplace = True)
        self.recap_data = pd.merge(self.recap_data,
                                   project.tags, 
                                   how = 'left',
                                   left_index = True,
                                   right_index = True)
        
        self.recap_data.reset_index(drop = False, inplace = True)
        self.recap_data.drop(columns = ['pulse_rate',
                                        'tag_type',
                                        'rel_date',
                                        'length'],
                             axis = 'columns',
                             inplace = True)
        
        # filter out tag data we don't want mucking up our staistical model
        if species != None:
            self.recap_data = self.recap_data[self.recap_data.Species == species]
        if rel_loc != None:
            self.recap_data = self.recap_data[self.recap_data.RelLoc == rel_loc]
        if cap_loc != None:
            self.recap_data = self.recap_data[self.recap_data.CapLoc == cap_loc]

        self.recap_data['state'] = self.recap_data.rec_id.map(receiver_to_state)
        
        self.recap_data = self.recap_data.astype({'freq_code':'object', 
                                                  'rec_id':'object', 
                                                  'epoch':'float32', 
                                                  'time_stamp':'datetime64', 
                                                  'lag':'float32', 
                                                  'cap_loc':'object', 
                                                  'rel_loc':'object', 
                                                  'state':'int32'})
        
        print ("Unique states in Returned data:%s"%(self.recap_data.state.unique()))

        if initial_state_release == True:
            '''we are modeling movement from the initial release location rather
            than the initial location in our competing risks model.  This allows
            us to quantify fall back.  If a fish never makes it to the intial
            spoke, then its fall back.

            If we care about modeling from the release point, we need to query
            release times of each fish, morph data into a recaptures file and
            merge it to self.data'''

            # get data   
            release_dat = project.tags
            
            # do some data management
            release_dat['rel_date'] = pd.to_datetime(release_dat.rel_date)
            release_dat['epoch'] = np.round((release_dat.rel_date - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
            release_dat.rename(columns = {'rel_date':'time_stamp'}, inplace = True)
            release_dat['rec_id'] = np.repeat('rel', len(release_dat))
            release_dat['state'] = np.zeros(len(release_dat)) 
            release_dat['presence_number'] = np.zeros(len(release_dat))
            

            # filter out tag data we don't want mucking up our staistical model
            if species != None:
                release_dat = release_dat[release_dat.Species == species]
            if rel_loc != None:
                release_dat = release_dat[release_dat.RelLoc == rel_loc]
            if cap_loc != None:
                release_dat = release_dat[release_dat.CapLoc == cap_loc]            
            
            self.recap_data = self.recap_data.append(release_dat)

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
            self.recap_data['presence_number'] = pd.to_numeric(self.recap_data['presence_number'], 
                                                              errors='coerce')
            self.recap_data.dropna(axis = 0,
                                  how = 'any',
                                  subset = ['presence_number'], 
                                  inplace = True)
            
            last_presence = self.recap_data.groupby(['freq_code','state'])\
                ['presence_number'].max().\
                    to_frame().\
                        reset_index()
                
            last_presence.rename(columns = {'presence_number':'max_presence'},inplace = True)
            last_presence = last_presence[last_presence.state == 1]
            fish_at_start = last_presence.freq_code.unique()
            all_fish = self.recap_data.freq_code.unique()

            # now that we have the last presence, iterate over fish
            for i in last_presence.iterrows():
                fish = i[1]['freq_code']
                max_pres = i[1]['max_presence']

                # get the first recap_data at the last presence in the initial state for this fish
                recap_0 = self.recap_data[(self.recap_data.freq_code == fish) & \
                                        (self.recap_data.state == 1) & \
                                            (self.recap_data.presence_number == max_pres)]
                min_epoch = recap_0.epoch.min()

                # drop rows using boolean indexing
                self.recap_data.drop(self.recap_data[(self.recap_data.freq_code == fish) & \
                                                   (self.recap_data.epoch < min_epoch)].index, 
                                    inplace = True)
            for i in all_fish:
                if i not in fish_at_start:
                    self.recap_data.drop(self.recap_data[self.recap_data.freq_code == i].index, 
                                        inplace = True)

        # Identify first recapture times
        self.start_times = self.recap_data[self.recap_data.state == 1].\
            groupby(['freq_code'])['epoch'].min().\
                to_frame()
                    
        self.start_times.rename(columns = {'epoch':'first_recapture'},
                                inplace = True)

    	# Clean Up stuff that doesn't make sense
        for fish in self.recap_data.freq_code.unique():
            # we only care about movements from the initial sstate - this is a competing risks model
            if fish not in self.start_times.index:
                self.recap_data = self.recap_data[self.recap_data.freq_code != fish]

        # identify unique fish to loop through
        self.fish = self.recap_data.freq_code.unique()

    def data_prep(self, 
                  project,
                  time_dependent_covariates = False, 
                  unknown_state = None, 
                  bucket_length_min = 15,
                  adjacency_filter = None):
        
        if unknown_state != None:
            '''It may be beneficial to allow fish to enter into an unknown state
            rather than become censored at their last recapture in the initial state.
            This way the Nelson-Aalen will match empirical expectations.  If we have
            a lot of censored fish we lose information from the denominator and
            numerator.  If we put fish into an unknown state rather than censoring
            them we still have informative data.  For this to work, we only need to
            know the last recapture of any fish in the initial state.  We will
            assess absorbption into the unknown state with a Boolean statement later on.'''

            last_epoch = self.recap_data[self.recap_data.state == 1].epoch.max()

        if time_dependent_covariates == False:
            '''This option will produce data appropriate for construction of
            Nelson-Aalen cumulative incidence functions and to produce the state
            tables.

            This option is not appropriate if we wish to perform Cox Proportional
            Hazards Regression modeling as we will not be able to join to time-
            dependent covariates in R.
            '''
            columns = ['freq_code','state','presence','epoch','time_delta','time_0','first_obs']    # create columns
            self.master_state_table = pd.DataFrame()
            for i in self.fish:
                # get data for this fish
                fish_dat = self.recap_data[self.recap_data.freq_code == i]
                
                # get first recapture in state
                fish_dat.sort_values(by = 'epoch', 
                                     ascending = True, 
                                     inplace = True) # sort by exposure time
                
                # get previous state and fill nans with current state
                fish_dat['prev_state'] = fish_dat['state'].shift(1)  
                fish_dat.at[0,'prev_state']= fish_dat.state.values[0]
                
                # create some counters
                presence = 1
                first_obs = 1
                
                # create empty data frame
                state_table = pd.DataFrame(columns = columns)
                
                # get release time and the first epoch after release time
                time_0 = self.start_times.at[i,'first_recapture']
                fish_dat = fish_dat[fish_dat.epoch >= time_0]
                time_1 = fish_dat.epoch.iloc[0]
                
                # calculate seconds from release
                time_delta = time_1 - time_0  

                # creat a row and add it to the state table                   
                row_arr = [i,
                           fish_dat.state.values[0],
                           presence,
                           time_1,
                           time_delta,
                           time_0,
                           first_obs] # create initial row for state table
                
                row = pd.DataFrame(np.array([row_arr]),columns = columns)       
                state_table = state_table.append(row)                            
                first_obs = 0 # no other recapture can be the first
                
                # give rows an arbitrary index and find the index of our last row
                fish_dat['idx'] = np.arange(0,len(fish_dat),1)                  
                max_idx = fish_dat.idx.iloc[-1]
                
                # loop through rows, if it is a new state enter transition data                 
                for j in fish_dat.iterrows():                                   
                    row_idx = j[1]['idx']                                       
                    state = j[1]['state']                               
                    prev_state = j[1]['prev_state']  

                    if state != prev_state  or row_idx == max_idx:                
                        time_1 = j[1]['epoch']                                  
                        if unknown_state != None \
                            and row_idx == max_idx \
                                and state == 1 \
                                    and time_1 < last_epoch:
                            state = unknown_state
                            
                        time_delta = time_1 - time_0                             
                        presence = presence + 1                             
                        row_arr = [i,state,presence,time_1,time_delta,time_0,first_obs]
                        
                        row = pd.DataFrame(row_arr,columns = columns)
                        row['state'] = pd.to_numeric(row.state)
                        row = row.astype({'freq_code':'object',
                                          'state':'int32',
                                          'presence':'int32',
                                          'epoch':'float32',
                                          'time_delta':'float32',
                                          'time_0': 'float32',
                                          'first_obs':'int32'})
                        
                        state_table = state_table.append(row)                    
                        time_0 = j[1]['epoch']

                print ("State Table Completed for Fish %s"%(i))
                
                # identify transitions and write to the state table
                from_rec = state_table['state'].shift(1)
                        
                to_rec = state_table['state'].astype(np.int32)
                trans = tuple(zip(from_rec,to_rec))
                state_table['transition'] = trans
                state_table['start_state'] = from_rec
                state_table['end_state'] = to_rec
                
                # get time all sorted out
                state_table['t0'] = np.zeros(len(state_table))
                state_table['t1'] = state_table['epoch'] - state_table['time_0']
                
                # write state table to master state table
                self.master_state_table = self.master_state_table.append(state_table)

            del i,j
        else:
            columns = ['FreqCode','startState','endState','presence','time_stamp','firstObs','t0','t1']    # create columns
            
            self.master_state_table = pd.DataFrame()
            self.bucket_length = bucket_length_min
            
            for i in self.fish:
                # get fish and sort by epoch
                fish_dat = self.recap_data[self.recap_data.freq_code == i]                   # get data for this fish
                fish_dat.sort_values(by = 'epoch', 
                                     ascending = True, 
                                     inplace = True)   # sort by exposure time
                
                # identify previous state and fill in nans
                fish_dat['prev_state'] = fish_dat['state'].shift(1)                      # get previous state
                fish_dat['prev_state'].fillna(fish_dat.state.values[0], inplace = True)  # fill NaN states with current state - for first record in data frame
                
                # initialize some counters
                presence = 0
                first_obs = 1
                
                # create an empty state table 
                state_table = pd.DataFrame(columns = columns)   

                # get initial start and end times and filter dataset                 
                time_0 = self.start_times.at[i,'first_recapture'] 
                fish_dat = fish_dat[fish_dat.epoch >= time_0]
                time_1 = fish_dat.epoch.iloc[0]
                
                # calculate seconds since releaes
                time_delta = time_1 - time_0    

                # creat a row and add it to the state table                   
                row_arr = [i,
                           0,
                           fish_dat.state.values[0],
                           presence,
                           fish_dat.time_stamp.values[-1],
                           first_obs,
                           time_0,
                           time_1] # create initial row for state table
                
                row = pd.DataFrame(np.array([row_arr]),columns = columns)       
                state_table = state_table.append(row) 
                
                # create arbitrary index and get the maximum
                fish_dat['idx'] = np.arange(0,len(fish_dat),1)                  
                max_idx = fish_dat.idx.iloc[-1]                                 
                
                # for each row, if it's a new presence add data to state table
                for j in fish_dat.iterrows():                                   # for every row in fish data
                    row_idx = j[1]['idx']                                       # what's the row number?
                    state_1 = int(j[1]['prev_state'])                            # what's the state
                    state_2 = int(j[1]['state'])                                # what was the previous state
                    ts = j[1]['time_stamp']
                    
                    # if it's a new state or the end add a row
                    if state_1 != state_2 or row_idx == max_idx:                   # if the present state does not equal the previous state or if we reach the end of the dataframe...
                        time_1 = j[1]['epoch']                                  # what time is it?
                        time_delta = time_1 - time_0                              # calculate difference in seconds between current time and release                                             # if it's a new state
                        presence = presence + 1                            # oh snap new observation for new state
                        row_arr = [i,
                                   state_1,
                                   state_2,
                                   presence,
                                   ts,
                                   first_obs,
                                   time_0,
                                   time_1]  # start a new row
                        row = pd.DataFrame(np.array([row_arr]),
                                           columns = columns)
                        state_table = state_table.append(row)                    # add the row to the state table data frame
                        time_0 = j[1]['epoch']
                        first_obs = 0
                        
                print ("State Table Completed for Fish %s"%(i))

                state_table.sort_values(by = 'time_0', 
                                        ascending = True, 
                                        inplace = True) # sort by exposure time
                
                # put time into increments to match time series variables
                time_bucket = self.bucket_length*60*1000000000                  # time bucket in nanoseconds
                state_table['flow_period'] = (state_table['time_0'].\
                                              astype(np.int64)//time_bucket+1) * time_bucket # round to nearest 15 minute period
                state_table['flow_period'] = pd.to_datetime(state_table['flow_period']) # turn it into a datetime object so we can use pandas to expand and fill
                
                # create arbitrary index
                row_num = np.arange(0,len(state_table),1)
                state_table['row_num'] = row_num
                
                # create an expanded state table
                exp_state_table = pd.DataFrame()
                
                # build expanded state table
                for row in state_table.iterrows():
                    row_idx = row[1]['row_num']                                 
                    t0 = row[1]['flow_period']                                
                    t1 = row[1]['t1']                                         
                    
                    # try expanding, if interval not large enough return nothing 
                    try:
                        expand = pd.date_range(t0,
                                               t1,
                                               freq = '%smin'%(self.bucket_length)) 
                    except ValueError:
                        expand = []
                    except AttributeError:
                        expand = []
                        
                    # if we can expand create intervals 
                    if len(expand) > 0:
                        # create a series using expanded time stamps
                        series = pd.Series(expand, 
                                           index = expand, 
                                           name = 'flow_period') 
                        
                        # convert series to invterval dataframe and perform data management 
                        intervals = series.to_frame()                          
                        intervals.reset_index(inplace = True, drop = True)     
                        intervals['t0'] = row[1]['t0']                         
                        intervals['t1'] = row[1]['t1']
                        intervals['startState'] = row[1]['startState']
                        intervals['endState'] = row[1]['endState']
                        intervals['timeStamp'] = row[1]['timeStamp']
                        intervals['FreqCode'] = row[1]['FreqCode']
                        intervals['presence'] = row[1]['presence']
                        newRowArr = np.array([row[1]['FreqCode'],
                                              row[1]['startState'],
                                              row[1]['endState'],
                                              row[1]['timeStamp'],
                                              row[1]['flowPeriod'],
                                              row[1]['t0'],
                                              row[1]['t1'],
                                              row[1]['presence']])
                        newRow = pd.DataFrame(np.array([newRowArr]),columns = ['FreqCode',
                                                                               'startState',
                                                                               'endState',
                                                                               'timeStamp',
                                                                               'flowPeriod',
                                                                               't0',
                                                                               't1',
                                                                               'presence']) # add first, non expanded row to new state table
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
                        exp_state_table = exp_state_table.append(newRow)         # now add all that stuff to the state table dataframe
                        del newRow, intervals, newRowArr, expand
                    else:
                        newRowArr = np.array([row[1]['FreqCode'],
                                              row[1]['startState'],
                                              row[1]['endState'],
                                              row[1]['timeStamp'],
                                              row[1]['flowPeriod'],
                                              row[1]['t0'],
                                              row[1]['t1'],
                                              row[1]['presence']])
                        newRow = pd.DataFrame(np.array([newRowArr]),columns = ['FreqCode',
                                                                               'startState',
                                                                               'endState',
                                                                               'timeStamp',
                                                                               'flowPeriod',
                                                                               't0',
                                                                               't1',
                                                                               'presence']) # add first, non expanded row to new state table
                        exp_state_table = exp_state_table.append(newRow)
                        del newRow, newRowArr
                # exp_state_table.sort_values(by = 't0', ascending = True, inplace = True)     # sort by exposure time
                # exp_state_table['time0'] = pd.to_datetime(exp_state_table['t0']) # create new time columns
                # exp_state_table['time1'] = pd.to_datetime(exp_state_table['t1'])
                # exp_state_table['t0'] = (pd.to_datetime(exp_state_table['t0']) - initialTime)/np.timedelta64(1,'s')
                # exp_state_table['t1'] = (pd.to_datetime(exp_state_table['t1']) - initialTime)/np.timedelta64(1,'s')
                # # calculate minimum t0 by presence
                # min_t0 = exp_stateTable.groupby(['presence'])['t0'].min()#.to_frame().rename({'t0':'min_t0'},inplace = True)
                # min_t0 = pd.Series(min_t0, name = 'min_t0')
                # min_t0 = pd.DataFrame(min_t0).reset_index()
                # # join to exp_stateTable as presence_time_0
                # exp_stateTable = pd.merge(left = exp_stateTable, right = min_t0, how = u'left',left_on = 'presence', right_on = 'presence')
                # # subtract presence_time_0 from t0 and t1
                # exp_stateTable['t0'] = exp_stateTable['t0'] -  exp_stateTable['min_t0']
                # exp_stateTable['t1'] = exp_stateTable['t1'] -  exp_stateTable['min_t0']
                # # drop presence_time_0 from exp_stateTable

                # exp_stateTable['hour'] = pd.DatetimeIndex(exp_stateTable['time0']).hour # get the hour of the day from the current time stamp
                # exp_stateTable['qDay'] = exp_stateTable.hour//6                # integer division by 6 to put the day into a quarter
                # exp_stateTable['test'] = exp_stateTable.t1 - exp_stateTable.t0 # this is no longer needed, but if t1 is smaller than t0 things are screwed up
                # stateTable = exp_stateTable
                # del exp_stateTable
                # stateTable['transition'] = tuple(zip(stateTable.startState.values.astype(int),stateTable.endState.values.astype(int))) # create transition variable, this is helpful in R
                # self.master_stateTable = self.master_stateTable.append(stateTable)
                # export
            self.master_stateTable.drop(labels = ['nextFlowPeriod'],axis = 1, inplace = True)


        if adjacency_filter is not None:
            '''When the truth value of a detection is assessed, a detection
            may be valid for a fish that is not present.

            In some instances, especially when using Yagi antennas, back-lobes
            may develop where a fish in the tailrace of a powerhouse is
            detected in the forebay antenna.  In these instances, a downstream
            migrating fish would not have migrated up through the powerhouse.

            From a false positive perspective, these records are valid detections.
            However, from a movement perspective, these series of detections
            could not occur and should be removed.

            This function repeatedly removes rows with 'illegal' movements
            until there are none left.  Rows with 'illegal' transitions are
            identified with a list that is passed to the function.

            input = list of illegal transitions stored as (from, to) tuples
            '''
            fish = self.master_stateTable.FreqCode.unique()

            for i in fish:
                fishDat =  self.master_stateTable[self.master_stateTable.FreqCode == i]
                self.master_stateTable = self.master_stateTable[self.master_stateTable.FreqCode != i]

                # create a condition, we're running this filter because we know illogical movements are present
                bad_moves_present = True

                # while there are illogical movements, keep filtering
                while bad_moves_present == True:
                    # let's keep count of the number of rows we are filtering
                    filtered_rows = 0.0

                    # for every known bad movement
                    for j in adjacency_filter:
                        print ("Starting %s filter"%(i))
                        # find those rows where this movement exists
                        fishDat['transition_filter'] = np.where(fishDat.transition == j,1,0)
                        fishDat.set_index(['time0'], inplace = True)

                        if fishDat.transition_filter.sum() > 0:
                            # add up those rows
                            filtered_rows = filtered_rows + fishDat.transition_filter.sum()
                            print ('%s rows found with %s movements'%(fishDat.transition_filter.sum(),j))

                            # do some data management, we need to take the start state and t0 of the affected rows and place them on the subsequent row
                            idx = fishDat.index[fishDat['transition_filter']==1]

                            for k in idx:
                                idx_int = fishDat.index.get_loc(k)
                                t0_col = fishDat.columns.get_loc('t0')
                                start_col = fishDat.columns.get_loc('startState')

                                # get start time and start state
                                start = fishDat.iloc[idx_int]['startState']
                                t0 = fishDat.iloc[idx_int]['t0']

                                # write it to next row
                                try:
                                    idx1 = idx_int + 1
                                except:
                                    start = fishDat.iloc[idx_int].index[0]
                                    idx1 = start + 1
                                try:
                                    fishDat.iloc[idx1, start_col] = start
                                    fishDat.iloc[idx1, t0_col] = t0
                                except IndexError:
                                    # when this occurs, there is no extra row - this last row will be deleted
                                    continue

                            # remove those rows
                            fishDat = fishDat[fishDat.transition_filter != 1]

                            # create a new transition field
                            fishDat['transition'] = tuple(zip(fishDat.startState.values.astype(int),
                                                              fishDat.endState.values.astype(int)))

                            fishDat.reset_index(inplace = True)
                        else:
                            print ("No illegal movements identified")
                            fishDat.reset_index(inplace = True)

                    if filtered_rows == 0.0:
                        print ("All illegal movements for fish %s removed"%(i))
                        # stop that loop
                        bad_moves_present = False

                    else:
                        # i feel bad for you son
                        print ("%s illegal movements present in iteration, go again"%(filtered_rows))

                fishDat.drop(labels = ['transition_filter'], axis = 1, inplace = True)
                self.master_stateTable = self.master_stateTable.append(fishDat)

        #self.master_stateTable = self.master_stateTable[self.master_stateTable.firstObs == 0]
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
        self.master_stateTable['transition'] = self.master_stateTable.transition.astype(str)
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
        print ("The maximum number of times each transition was made by each fish:")
        max_transCount = self.fishTransCount.groupby(['FreqCode','transition'])['transCount'].max()
        #max_transCount.to_csv(os.path.join(r'C:\Users\Kevin Nebiolo\Desktop','maxCountByFreqCode.csv'))
        print (max_transCount)
        print ("")
        print ("Movement summaries - Duration between states in seconds")
        self.master_stateTable['dur'] = (self.master_stateTable.t1 - self.master_stateTable.t0)
        move_summ = self.master_stateTable.groupby('transition')['dur'].describe().round(decimals = 3)
        print (move_summ)
