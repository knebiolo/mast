# -*- coding: utf-8 -*-
"""
Statistical Model Data Formatting Module
========================================

This module contains classes for formatting cleaned telemetry data
into input files for various statistical survival and movement models.

Classes
-------
cjs_data_prep : Cormack-Jolly-Seber (CJS) mark-recapture formatting
    Creates encounter history matrices for Program MARK survival analysis.
    
lrdr_data_prep : Live Recapture Dead Recovery (LRDR) formatting
    Combines live detections with mobile tracking mortality surveys.
    Used for post-passage survival estimation with recovery data.
    
time_to_event : Multi-state time-to-event (competing risks) formatting
    Creates counting-process style data for multi-state survival models.
    Most commonly used class for fish passage and movement studies.

Typical Usage
-------------
>>> import pymast
>>> project = pymast.radio_project('path/to/project')
>>> 
>>> # Time-to-event model (most common)
>>> receiver_to_state = {'R01': 1, 'R02': 2, 'R03': 9}  # Map receivers to states
>>> tte = pymast.formatter.time_to_event(receiver_to_state, project)
>>> tte.data_prep(project, adjacency_filter=[(9,1), (9,2)])  # Remove illegal movements
>>> stats = tte.summary()  # Print movement statistics
>>> tte.master_state_table.to_csv('output.csv')  # Export for R/msm
>>>
>>> # CJS model
>>> receiver_to_recap = {'R01': 'R00', 'R02': 'R01', 'R03': 'R02'}
>>> cjs = pymast.formatter.cjs_data_prep(receiver_to_recap, project)
>>> cjs.input_file('model_name', 'output_dir')  # Creates .inp for MARK

Notes
-----
- All classes expect data from a pymast.radio_project with complete pipeline:
  1. Data imported (parsers)
  2. Classified (naive_bayes)
  3. Bouts detected (overlap_removal.bout)
  4. Overlaps removed (overlap_removal.overlap_reduction)
  5. Recaptures built (radio_project.make_recaptures_table)
  
- The time_to_event class automatically filters:
  - Overlapping detections (overlapping == 1)
  - Ambiguous overlaps (ambiguous_overlap == 1)
  - Small bouts (< 3 detections)
  
- Adjacency filter removes biologically impossible state transitions
  based on study site geography (e.g., downstream → upstream without ladder).

See Also
--------
pymast.radio_project : Project management and database
pymast.overlap_removal : Bout detection and overlap resolution
pymast.naive_bayes : Classification of true detections
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

        self.recap_data = pd.read_hdf(project.db,
                             'recaptures')#, 
                             #where = qry)
        
        #self.recap_data.set_index('freq_code', inplace = True)
        #project.tags.reset_index('freq_code', inplace = True)
        self.recap_data = pd.merge(self.recap_data, 
                                   project.tags,
                                   left_on = 'freq_code', 
                                   right_on = 'freq_code',
                                   how = 'left')
        self.recap_data.reset_index(inplace = True)
        #project.tags.reset_index(drop = False, inplace = True)
        
        # filter out tag data we don't want mucking up our staistical model
        if species != None:
            self.recap_data = self.recap_data[self.recap_data.species == species]
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
            # Check if the index is the default integer index
            has_default_index = isinstance(rel_dat.index, pd.RangeIndex)
            
            # If the DataFrame has a default index, reset it
            if not has_default_index:
                rel_dat.reset_index(inplace = True)            
            self.recap_data = pd.concat([self.recap_data,rel_dat])

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
                    self.recap_data.drop(self.recap_data[self.recap_data.freq_code == fish].index, inplace = True)
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

class time_to_event():
    """
    Multi-state Time-to-Event Data Formatter
    
    Formats radio telemetry recapture data into counting-process style records
    for multi-state survival models (competing risks, Cox proportional hazards).
    
    This is the primary class for fish passage and movement studies using
    state-based survival models in R (msm, mstate packages).
    
    Parameters
    ----------
    receiver_to_state : dict
        Mapping of receiver IDs to integer state codes.
        Example: {'R_forebay': 1, 'R_powerhouse': 2, 'R_tailrace': 9}
        
    project : pymast.radio_project
        Project object with completed data pipeline (imported, classified,
        bouts detected, overlaps removed, recaptures built).
        
    input_type : str, optional
        Legacy parameter, always uses 'query' from HDF5. Default 'query'.
        
    initial_state_release : bool, optional
        If True, models movement from release point (state 0).
        If False, models from first recapture at state 1. Default False.
        
    last_presence_time0 : bool, optional
        If True, uses last presence at initial state as time_0 (for
        downstream migration studies where fish linger upstream). Default False.
        
    cap_loc : str, optional
        Filter by capture location. Default None (all fish).
        
    rel_loc : str, optional
        Filter by release location. Default None (all fish).
        
    species : str, optional
        Filter by species code. Default None (all species).
        
    rel_date : str, optional
        Filter fish released after this date (YYYY-MM-DD). Default None.
        
    recap_date : str, optional
        Filter recaptures after this date (YYYY-MM-DD). Default None.
    
    Attributes
    ----------
    recap_data : pd.DataFrame
        Loaded recapture data after all filtering (overlapping, ambiguous,
        bout size, species/location filters).
        
    master_state_table : pd.DataFrame
        Final formatted output with columns:
        - freq_code : Fish ID
        - species : Species code
        - start_state : State at time_0 (integer)
        - end_state : State at time_1 (integer)
        - presence : Presence number for this fish
        - time_stamp : Datetime of transition
        - time_delta : Duration in state (seconds)
        - first_obs : Binary flag for first observation
        - time_0 : Entry time (epoch seconds)
        - time_1 : Exit time (epoch seconds)
        - transition : Tuple (start_state, end_state)
        
    fish : np.ndarray
        Array of unique fish IDs in dataset.
        
    start_times : pd.DataFrame
        First recapture time per fish (for time_0 initialization).
    
    Methods
    -------
    data_prep(project, unknown_state=None, bucket_length_min=15, adjacency_filter=None)
        Process recapture data into state transitions. Applies adjacency filter
        if provided to remove biologically impossible movements.
        
    summary()
        Calculate and print movement statistics (transition counts, durations).
        Returns dict with statistics and saves CSV files to output directory.
    
    Examples
    --------
    >>> import pymast
    >>> project = pymast.radio_project('path/to/project')
    >>> 
    >>> # Define state mapping
    >>> receiver_to_state = {
    ...     'R_release': 0,      # Release location
    ...     'R_forebay': 1,      # Upstream of dam
    ...     'R_powerhouse': 2,   # At turbines
    ...     'R_tailrace': 9      # Downstream of dam
    ... }
    >>> 
    >>> # Initialize formatter
    >>> tte = pymast.formatter.time_to_event(
    ...     receiver_to_state,
    ...     project,
    ...     species='Chinook',
    ...     initial_state_release=True
    ... )
    >>> 
    >>> # Define illegal movements (downstream can't go upstream without ladder)
    >>> adjacency_filter = [
    ...     (9, 1),  # Tailrace to forebay
    ...     (9, 2),  # Tailrace to powerhouse
    ...     (2, 1),  # Powerhouse to forebay
    ... ]
    >>> 
    >>> # Process data
    >>> tte.data_prep(project, adjacency_filter=adjacency_filter)
    >>> 
    >>> # Get statistics
    >>> stats = tte.summary()
    >>> 
    >>> # Export for R analysis
    >>> tte.master_state_table.to_csv('tte_data.csv', index=False)
    >>> 
    >>> # In R:
    >>> # library(msm)
    >>> # data <- read.csv('tte_data.csv')
    >>> # model <- msm(end_state ~ time_1, subject=freq_code,
    >>> #              data=data, qmatrix=Q, ...)
    
    Notes
    -----
    **Automatic Filtering**:
    
    The class automatically removes:
    - Detections with overlapping == 1 (removed during overlap resolution)
    - Detections with ambiguous_overlap == 1 (couldn't resolve receiver conflict)
    - Bouts with < 3 detections (likely false positives)
    
    **Adjacency Filter**:
    
    Removes biologically impossible state transitions based on site geography.
    Iteratively processes each fish until all illegal movements are removed.
    
    Works by:
    1. Finding rows with illegal transitions
    2. Carrying forward the start_state and time_0 to next row
    3. Deleting the illegal row
    4. Recalculating transitions
    5. Repeating until clean
    
    **State Coding**:
    
    Common conventions:
    - 0: Release location (if initial_state_release=True)
    - 1: Initial study area state
    - 2-8: Intermediate states
    - 9: Terminal/downstream state
    
    **Time Format**:
    
    All times are Unix epoch (seconds since 1970-01-01 00:00:00 UTC).
    Convert in R: `as.POSIXct(time_0, origin='1970-01-01')`
    
    See Also
    --------
    pymast.radio_project.make_recaptures_table : Builds input recaptures table
    pymast.overlap_removal.overlap_reduction : Marks overlapping detections
    cjs_data_prep : For standard CJS survival models in Program MARK
    """
    def __init__(self, 
                 receiver_to_state,
                 project, 
                 input_type = 'query', 
                 initial_state_release = False, 
                 last_presence_time0 = False, 
                 cap_loc = None,
                 rel_loc = None,
                 species = None,
                 rel_date = None,
                 recap_date = None):
        # Import Data From Project HDF
        self.rel_loc = rel_loc
        self.cap_loc = cap_loc
        
        self.initial_state_release = initial_state_release
        # get recaptures, but first build a query
        query_parts = []
        for key in receiver_to_state:
            query_parts.append(f"rec_id == '{key}'")   
        qry = " | ".join(query_parts)

        self.recap_data = pd.read_hdf(project.db,
                                      'recaptures',
                                      where = qry)#"rec_id == 'R11'")
        
        # Debug: check if ambiguous_overlap column exists
        print(f"[TTE] Columns in recaptures table: {self.recap_data.columns.tolist()}")
        print(f"[TTE] Has ambiguous_overlap: {'ambiguous_overlap' in self.recap_data.columns}")
        
        self.recap_data = self.recap_data[self.recap_data.hit_ratio > 0.1]
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
        # remove any row where df['your_column'] is NaN
        self.recap_data.dropna(subset=['rel_date'], inplace=True)
        df = self.recap_data
        
        # Filter out overlapping detections (keep only overlapping=0)
        self.recap_data = self.recap_data[self.recap_data.overlapping == 0]
        
        # Filter out ambiguous overlaps if column exists (keep only ambiguous_overlap=0)
        # For time-to-event models, fish can only be in one place at once
        if 'ambiguous_overlap' in self.recap_data.columns:
            before_ambig_filter = len(self.recap_data)
            self.recap_data = self.recap_data[self.recap_data.ambiguous_overlap == 0]
            after_ambig_filter = len(self.recap_data)
            print(f"[TTE] Filtered {before_ambig_filter - after_ambig_filter} ambiguous overlap detections")
        
        # Filter out detections from small bouts (likely false positives)
        # Calculate bout size for each bout and filter
        if 'bout_no' in self.recap_data.columns:
            # Count detections per bout (per fish, per receiver, per bout_no)
            bout_sizes = self.recap_data.groupby(['freq_code', 'rec_id', 'bout_no']).size().reset_index(name='bout_size')
            self.recap_data = self.recap_data.merge(bout_sizes, on=['freq_code', 'rec_id', 'bout_no'], how='left')
            
            # Filter out bouts with < 3 detections (single spurious detections)
            min_bout_size = 3
            before_bout_filter = len(self.recap_data)
            self.recap_data = self.recap_data[self.recap_data['bout_size'] >= min_bout_size]
            after_bout_filter = len(self.recap_data)
            print(f"[TTE] Filtered {before_bout_filter - after_bout_filter} detections from bouts with < {min_bout_size} detections")
            
            # Drop the bout_size column (no longer needed)
            self.recap_data = self.recap_data.drop(columns=['bout_size'])
        
        self.recap_data['rel_date'] = pd.to_datetime(self.recap_data.rel_date)
        
        self.recap_data.drop(columns = ['pulse_rate',
                                        'tag_type',
                                        'length'],
                             axis = 'columns',
                             inplace = True)
        
        if "species" not in self.recap_data.columns:
            self.recap_data["species"] = np.nan  # or any default value you want
            print ("Species Column Not In Recap Data, defaulting to Nan")
        
        # filter out tag data we don't want mucking up our staistical model
        if species != None:
            self.recap_data = self.recap_data[self.recap_data.species == species]
        if rel_loc != None:
            self.recap_data = self.recap_data[self.recap_data.rel_loc == rel_loc]
        if cap_loc != None:
            self.recap_data = self.recap_data[self.recap_data.cap_loc == cap_loc]
        if rel_date != None:
            self.recap_data = self.recap_data[self.recap_data.rel_date >= pd.to_datetime(rel_date)]
        if recap_date != None:
            self.recap_data = self.recap_data[self.recap_data.time_stamp >= pd.to_datetime(recap_date)]

        
        self.recap_data['state'] = self.recap_data.rec_id.map(receiver_to_state)
        self.recap_data.reset_index(inplace = True)

        self.recap_data = self.recap_data.astype({'freq_code':'object', 
                                                  'rec_id':'object', 
                                                  'epoch':'float32', 
                                                  'time_stamp':'datetime64[ns]', 
                                                  'lag':'float32', 
                                                  'cap_loc':'object', 
                                                  'rel_loc':'object', 
                                                  'state':'int32'})
        
        print ("Unique states in Returned data:%s"%(self.recap_data.state.unique()))
        
        # identify unique fish to loop through
        self.fish = self.recap_data.freq_code.unique()

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
                    
        if initial_state_release == True:
            '''we are modeling movement from the initial release location rather
            than the initial location in our competing risks model.  This allows
            us to quantify fall back.  If a fish never makes it to the intial
            spoke, then its fall back.

            If we care about modeling from the release point, we need to query
            release times of each fish, morph data into a recaptures file and
            merge it to self.data'''

            # get data   
            release_dat = project.tags.copy()
            
            # do some data management
            release_dat['rel_date'] = pd.to_datetime(release_dat.rel_date)
            release_dat['epoch'] = np.round((release_dat.rel_date - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'),6)
            release_dat.rename(columns = {'rel_date':'time_stamp'}, inplace = True)
            release_dat['rec_id'] = np.repeat('rel', len(release_dat))
            release_dat['state'] = np.zeros(len(release_dat)) 
            release_dat['presence_number'] = np.zeros(len(release_dat))
            
            
            # filter out tag data we don't want mucking up our staistical model
            if species != None:
                release_dat = release_dat[release_dat.species == species]
            if rel_loc != None:
                release_dat = release_dat[release_dat.rel_loc == rel_loc]
            if cap_loc != None:
                release_dat = release_dat[release_dat.cap_loc == cap_loc]
            if rel_date != None:
                release_dat = release_dat[release_dat.time_stamp >= pd.to_datetime(rel_date)]
            if recap_date != None:
                release_dat = release_dat[release_dat.time_stamp >= pd.to_datetime(recap_date)]
                
            release_dat.reset_index(inplace = True)
            release_dat = release_dat[release_dat['freq_code'].isin(self.fish)]

            # add to recaptures table and create a start times table
            self.recap_data = pd.concat([self.recap_data, release_dat], axis=0, ignore_index=True)
            self.start_times = release_dat[['freq_code','epoch']]
            self.start_times.rename(columns = {'epoch':'first_recapture'},
                                    inplace = True)
            self.start_times.set_index('freq_code', inplace = True) 
        else:
            # movement from state 1
            self.start_times = self.recap_data[self.recap_data.state == 1].\
                groupby(['freq_code'])['epoch'].min().\
                    to_frame()
            
            # movement from first recapture        
            # self.start_times = (
            #     self.recap_data
            #     .groupby(['freq_code'])['epoch']
            #     .min()
            #     .to_frame()
            # )

            self.start_times.rename(columns = {'epoch':'first_recapture'},
                                    inplace = True)

            # Clean Up stuff that doesn't make sense
            for fish in self.recap_data.freq_code.unique():
                # we only care about movements from the initial sstate - this is a competing risks model
                if fish not in self.start_times.index:
                    self.recap_data = self.recap_data[self.recap_data.freq_code != fish]

    def data_prep(self, project, unknown_state=None, bucket_length_min=15, adjacency_filter=None):
        self.project = project
        if unknown_state is not None:
            last_epoch = self.recap_data[self.recap_data.state == 1].epoch.max()
    
        columns = ['freq_code','species', 'start_state', 'end_state', 'presence', 'time_stamp',
                   'time_delta', 'first_obs', 'time_0', 'time_1', 'transition']  # Include 'transition' here
    
        self.master_state_table = pd.DataFrame()
        self.bucket_length = bucket_length_min
    
        # Sorting recap_data by freq_code and epoch for efficient processing
        self.recap_data.sort_values(by=['freq_code', 'epoch'], ascending=True, inplace=True)
    
        # if self.initial_state_release == True:
        #     # Merge start_times into recap_data based on freq_code
        #     self.recap_data = self.recap_data.merge(
        #         self.start_times[['first_recapture']].reset_index(),
        #         on='freq_code',
        #         how='left'
        #     )
            
        self.recap_data = self.recap_data.merge(
            self.start_times[['first_recapture']].reset_index(),
            on='freq_code',
            how='left')
        
        # Create a boolean array to mark the start of a new fish
        fish_start_mask = self.recap_data['freq_code'] != self.recap_data['freq_code'].shift(1)
    
        # Initialize state tracking columns
        self.recap_data['prev_state'] = self.recap_data.groupby('freq_code')['state'].shift(1).fillna(0).astype(int)
        if self.initial_state_release == False:
            self.recap_data = self.recap_data[self.recap_data.prev_state > 0]
        # Set time_0 to the previous epoch or first_recapture if it's the first observation
        self.recap_data['time_0'] = self.recap_data.groupby('freq_code')['epoch'].shift(1)
        self.recap_data['time_0'].fillna(self.recap_data['first_recapture'], inplace=True)
    
        self.recap_data['time_delta'] = self.recap_data['epoch'] - self.recap_data['time_0']
    
        # Identify the rows where state changes or the fish changes (new fish)
        state_change_mask = self.recap_data['state'] != self.recap_data['prev_state']
        last_recapture_mask = self.recap_data.groupby('freq_code')['epoch'].transform('max') == self.recap_data['epoch']
        mask = state_change_mask | last_recapture_mask
    
        # Filter rows to keep only those where state changes or it's the last record for the fish
        state_table = self.recap_data[mask].copy()
    
        # Fill in the remaining columns
        state_table['start_state'] = state_table['prev_state'].astype('int32')
        state_table['end_state'] = state_table['state'].astype('int32')
        # drop duplicates 
        state_table = state_table.drop_duplicates(subset=['time_0', 'end_state', 'start_state'])

        state_table['presence'] = state_table.groupby('freq_code').cumcount()
        state_table['first_obs'] = fish_start_mask.astype(int)
        state_table['time_1'] = state_table['epoch']
    
        # Create the 'transition' column by zipping 'start_state' and 'end_state'
        state_table['transition'] = list(zip(state_table['start_state'].astype('int32'), state_table['end_state'].astype('int32')))
    
        # Add flow period for time-dependent variables
        state_table['flow_period'] = state_table['time_stamp'].dt.round('30min')
    
        
        # Write state table to master state table
        self.master_state_table = pd.concat([self.master_state_table, state_table[columns]], axis=0, ignore_index=True)

        # --- Fixes and rules for initial_state_release = True ---
        # Rules:
        # 1) If initial_state_release == True we only include fish that were
        #    recaptured at least once after release (i.e., have a state > 0).
        # 2) Fish must never transition into the release state (0). If a row
        #    would transition TO state 0, drop that transition.
        # 3) If a fish's last observed state is some S>0, ensure there is an
        #    explicit diagonal row (S->S) representing last-seen at that
        #    receiver with zero duration (instantaneous) so counts appear on
        #    the diagonal of the transition matrix.
        if self.initial_state_release:
            # Keep only fish that have at least one non-release observation
            fish_with_nonrelease = self.master_state_table[self.master_state_table['end_state'] > 0]['freq_code'].unique()
            self.master_state_table = self.master_state_table[self.master_state_table['freq_code'].isin(fish_with_nonrelease)].copy()

            # Convert any transitions that end at the release state (end_state == 0)
            # into a stay-in-place diagonal entry: set end_state = start_state,
            # zero the duration, and set time_1 == time_0. This preserves the
            # fish's presence at the prior state rather than sending it to 0.
            if 'end_state' in self.master_state_table.columns:
                mask_end0 = self.master_state_table['end_state'] == 0
                count_end0 = int(mask_end0.sum())
                if count_end0 > 0:
                    print(f"[TTE] Converting {count_end0} transitions ending at release state (0) into stay-in-place rows")
                    idxs = self.master_state_table.index[mask_end0].tolist()
                    for ii in idxs:
                        try:
                            start_state = int(self.master_state_table.at[ii, 'start_state'])
                        except Exception:
                            # fallback if missing
                            start_state = int(self.master_state_table.loc[ii, 'start_state'])
                        # set end_state to start_state
                        self.master_state_table.at[ii, 'end_state'] = start_state
                        # zero the time_delta and make time_1 equal time_0 (instantaneous)
                        self.master_state_table.at[ii, 'time_delta'] = 0
                        self.master_state_table.at[ii, 'time_1'] = self.master_state_table.at[ii, 'time_0']
                        # update transition tuple
                        self.master_state_table.at[ii, 'transition'] = (start_state, start_state)

            # For each fish, ensure last-seen state has a diagonal row
            diag_rows = []
            for fish in self.master_state_table['freq_code'].unique():
                fish_rows = self.master_state_table[self.master_state_table['freq_code'] == fish]
                if fish_rows.empty:
                    continue
                # find row with max time_1 (last observation)
                last_row = fish_rows.loc[fish_rows['time_1'].idxmax()]
                last_state = int(last_row['end_state'])
                last_time = int(last_row['time_1'])

                # if last_state > 0, check if there already is a diagonal at that time
                diag_exists = ((fish_rows['start_state'] == last_state) & (fish_rows['end_state'] == last_state) & (fish_rows['time_1'] == last_time)).any()
                if not diag_exists:
                    # create an instantaneous diagonal transition (duration 0)
                    diag = {
                        'freq_code': fish,
                        'species': last_row.get('species', np.nan),
                        'start_state': last_state,
                        'end_state': last_state,
                        'presence': last_row.get('presence', 0),
                        'time_stamp': last_row.get('time_stamp', pd.to_datetime(last_time, unit='s')),
                        'time_delta': 0,
                        'first_obs': 0,
                        'time_0': last_time,
                        'time_1': last_time,
                        'transition': (last_state, last_state)
                    }
                    diag_rows.append(diag)

            if diag_rows:
                diag_df = pd.DataFrame(diag_rows)
                # Ensure dtype consistency
                diag_df = diag_df.astype({k: v for k, v in self.master_state_table.dtypes.items() if k in diag_df.columns})
                self.master_state_table = pd.concat([self.master_state_table, diag_df], axis=0, ignore_index=True)

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
            fish = self.master_state_table.freq_code.unique()
            filtered = pd.DataFrame()
            for i in fish:
                fish_dat =  self.master_state_table[self.master_state_table.freq_code == i]

                # create a condition, we're running this filter because we know illogical movements are present
                bad_moves_present = True

                # while there are illogical movements, keep filtering
                while bad_moves_present == True:
                    # let's keep count of the number of rows we are filtering
                    filtered_rows = 0.0

                    # for every known bad movement
                    for j in adjacency_filter:
                        # find those rows where this movement exists
                        try:
                            fish_dat['transition_filter'] = np.where(fish_dat.transition == j,1,0)
                        except:
                            bad_moves_present = False  # Fixed: was == instead of =
                            break
                        #fish_dat.set_index(['time_0'], inplace = True)

                        if fish_dat.transition_filter.sum() > 0:
                            # add up those rows
                            filtered_rows = filtered_rows + fish_dat.transition_filter.sum()
                            print ('%s rows found with %s movements'%(fish_dat.transition_filter.sum(),j))

                            # do some data management, we need to take the start state and t0 of the affected rows and place them on the subsequent row
                            idx = fish_dat.index[fish_dat['transition_filter']==1]
                            time0 = fish_dat.iloc[0]['time_0']

                            for k in idx:
                                idx_int = fish_dat.index.get_loc(k)
                                t0_col = fish_dat.columns.get_loc('time_0')
                                start_col = fish_dat.columns.get_loc('start_state')

                                # get start time and start state
                                start = fish_dat.iloc[idx_int]['start_state']
                                t0 = fish_dat.iloc[idx_int]['time_0']

                                # write it to next row
                                try:
                                    idx1 = idx_int + 1
                                except:
                                    start = fish_dat.iloc[idx_int].index[0]
                                    idx1 = start + 1
                                try:
                                    fish_dat.iloc[idx1, start_col] = start
                                    fish_dat.iloc[idx1, t0_col] = t0
                                except IndexError:
                                    # when this occurs, there is no extra row - this last row will be deleted
                                    continue

                            # remove those rows
                            fish_dat = fish_dat[fish_dat.transition_filter != 1]
                            fish_dat.time_0 = time0

                            # create a new transition field
                            fish_dat['transition'] = list(zip(fish_dat.start_state.values.astype(int),
                                                              fish_dat.end_state.values.astype(int)))  # Fixed: was tuple, should be list
                            
                            #fish_dat.reset_index(inplace = True)
                        else:
                            pass  # No illegal movements for this transition type
                            #fish_dat.reset_index(inplace = True)

                    if filtered_rows == 0.0:
                        # stop that loop
                        bad_moves_present = False
                    else:
                        pass  # Continue filtering
                
                # we can only have 1 transmission to point of no return - let's grab the last recapture
                equal_rows = fish_dat[fish_dat['start_state'] == fish_dat['end_state']]

                # Step 2: Get the index of the last occurrence where column1 equals column2
                try:
                    last_index = fish_dat.index[-1]

                    # Step 3: Drop all rows where column1 equals column2 except the last one
                    if len(equal_rows) > 1 and equal_rows.index[-1] == last_index:
                        fish_dat = fish_dat.drop(equal_rows.index[:-1])
                    elif len(equal_rows) > 1 and equal_rows.index[-1] != last_index:
                        fish_dat = fish_dat.drop(equal_rows.index)
                    elif len(equal_rows) == 1:
                        fish_dat = fish_dat.drop(equal_rows.index)
                    else:
                        pass
    
                    fish_dat.drop(labels = ['transition_filter'], axis = 1, inplace = True)
                    filtered = pd.concat([filtered, fish_dat])
                except Exception as e:
                    print (f"[TTE] Warning: Error processing fish {i}: {e}")
            
            # Print summary statistics
            initial_transitions = len(self.master_state_table)
            final_transitions = len(filtered)
            removed = initial_transitions - final_transitions
            print(f"\n[TTE] Adjacency filter complete:")
            print(f"  Initial transitions: {initial_transitions:,}")
            print(f"  Final transitions: {final_transitions:,}")
            print(f"  Removed illegal movements: {removed:,} ({removed/initial_transitions*100:.1f}%)")
            
            if self.initial_state_release == False:
                self.master_state_table 

            self.master_state_table = filtered
            
        #self.master_stateTable = self.master_stateTable[self.master_stateTable.firstObs == 0]
        #self.master_state_table.to_csv(os.path.join(project.output_dir,'state_table.csv')
        
    # generate summary statistics
    def summary(self, print_summary = True):
        """Prepare the data needed for summarization."""
        self.master_state_table.dropna(subset = ['time_delta'],inplace = True)
        self.master_state_table = self.master_state_table.astype({'freq_code':'object',
                                                                  'species':'object',
                                                                  'start_state':'int32',
                                                                  'end_state':'int32',
                                                                  'presence':'int32',
                                                                  'time_stamp':'datetime64[ns]',
                                                                  'time_delta':'int32',
                                                                  'first_obs':'int32',
                                                                  'time_0':'int32',
                                                                  'time_1':'int32',
                                                                  'transition':'object'})
                                                                  
        self.master_state_table['dur'] = (
            self.master_state_table['time_1'].astype('int32') - 
            self.master_state_table['time_0'].astype('int32')
        )
        
        self.unique_fish_count = len(self.master_state_table['freq_code'].unique())
        self.count_per_state = self.master_state_table.groupby('end_state')['freq_code'].nunique()
        self.msm_state_table = pd.crosstab(self.master_state_table['start_state'], self.master_state_table['end_state'])
        self.count_table = self.master_state_table.groupby(['start_state', 'end_state'])['freq_code'].nunique().unstack().fillna(0).astype('int32')
        self.fish_trans_count = self.master_state_table.groupby(['freq_code', 'transition']).size().unstack(fill_value=0)
        
        grouped_stats = (
            self.master_state_table
            .groupby('transition')
            .agg({
                'dur': [
                    'min',
                    'median',
                    'max'
                ]
            })
        )
        self.move_summ = grouped_stats
  
        """Generate summary statistics as a dictionary."""
        min_trans_count = self.fish_trans_count.min()
        med_trans_count = self.fish_trans_count.median()
        max_trans_count = self.fish_trans_count.max()

        summary_stats = {
            "unique_fish_count": self.unique_fish_count,
            "count_per_state": self.count_per_state,
            "state_transition_table": self.msm_state_table,
            "movement_count_table": self.count_table,
            "min_transition_count": min_trans_count,
            "median_transition_count": med_trans_count,
            "max_transition_count": max_trans_count,
            "movement_duration_summary": self.move_summ
        }
        # Print stats
            
        print("-" * 110)
        print("Time To Event Data Manage Complete")
        print("-" * 110 + "\n")
        
        print("--------------------------------------- MOVEMENT SUMMARY STATISTICS -----------------------------------------\n")
        print(f"In total, there were {summary_stats['unique_fish_count']} unique fish within this competing risks model.\n")
        
        print(f"{summary_stats['unique_fish_count']} fish made the movements as enumerated in the state transition table:")
        print(summary_stats['state_transition_table'])
        print("The table should read movement from a row to a column.\n")
        
        print("The number of unique fish to make these movements are found in the following count table:")
        print(summary_stats['movement_count_table'], "\n")

        print("The number of movements a fish is expected to make is best described with min, median, and maximum statistics.\n")
        print("Minimum number of times each transition was made:")
        print(summary_stats['min_transition_count'], "\n")

        print("Median number of times each transition was made:")
        print(summary_stats['median_transition_count'], "\n")

        print("Maximum number of times each transition was made by each fish:")
        print(summary_stats['max_transition_count'], "\n")
        
        print("Movement summaries - Duration between states in seconds:")
        print(summary_stats['movement_duration_summary'], "\n")
        self.msm_state_table.to_csv(os.path.join(self.project.output_dir,'state_table.csv'))
        self.master_state_table.to_csv(os.path.join(self.project.output_dir,'tte.csv'))
        self.move_summ.to_csv(os.path.join(self.project.output_dir,'movement_summary.csv'))
        return summary_stats

# Example usage
# master_state_table = pd.DataFrame({...})
# summary = SurvivalDataSummary(master_state_table)
# summary_stats = summary.summary()
# summary.print_summary()  # Optionally print the summary