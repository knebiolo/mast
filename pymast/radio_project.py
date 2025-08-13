# -*- coding: utf-8 -*-
'''
Module contains all of the functions to create a radio telemetry project.'''

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import h5py
import datetime
import pymast.naive_bayes as naive_bayes
import pymast.parsers as  parsers
import pymast.predictors as predictors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
import shutil
import warnings
import dask.dataframe as dd
import dask.array as da
from dask_ml.cluster import KMeans
warnings.filterwarnings("ignore")

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

push = 'push'

class radio_project():
    '''
    A class to manage and organize data and parameters for a Radio Telemetry project.

    This class is designed to facilitate the handling of datasets and parameters associated with radio telemetry studies. It organizes project data, including tag, receiver, and node information, and manages the project directory structure and database.

    Attributes:
    - project_dir (str): The directory where the project data and outputs will be stored.
    - db_name (str): The name of the project database.
    - db (str): The path to the project database file.
    - tags (DataFrame or similar): Data containing information about the tags used in the project.
    - receivers (DataFrame or similar): Data containing information about the receivers used in the project.
    - nodes (DataFrame or similar, optional): Data containing information about the nodes used in the project, if applicable.
    - data_dir (str): Directory path for storing raw data.
    - training_dir (str): Directory path for storing training files.
    - output_dir (str): Directory path for storing output files.
    - figures_dir (str): Directory path for storing figures.
    - hdf5 (h5py.File): HDF5 file object for the project database.

    Methods:
    - initialize_hdf5: Initializes the HDF5 database with initial data arrays.
    '''
    
    def __init__(self, project_dir, db_name, detection_count, duration, tag_data, receiver_data, nodes_data = None):
        '''
        Initializes the radio_project class with project parameters and datasets.
        
        Sets up the project directory structure, initializes the project database, and stores the provided datasets.
        
        Parameters:
        - project_dir (str): The root directory for the project.
        - db_name (str): The name of the database file to be created or used.
        - det (DataFrame or similar): Data containing detection information.
        - duration (int or float): The duration of the project or a related parameter.
        - tag_data (DataFrame or similar): Data containing information about the tags.
        - receiver_data (DataFrame or similar): Data containing information about the receivers.
        - nodes_data (DataFrame or similar, optional): Data containing information about the nodes, if applicable.
        
        The method creates the necessary directories for the project, initializes the HDF5 database, and sets up the class attributes.
        '''
        # set model parameters
        self.project_dir = project_dir     
        self.db_name = db_name
        self.db = os.path.join(project_dir,'%s.h5'%(db_name))
        self.tags = tag_data
        self.study_tags = self.tags[self.tags.tag_type == 'study'].freq_code.values
        self.test_tags = self.tags[self.tags.tag_type == 'TEST'].freq_code.values
        self.beacon_tags = self.tags[self.tags.tag_type == 'BEACON'].freq_code.values
        self.tags.set_index('freq_code', inplace = True)
        self.receivers = receiver_data
        self.receivers.set_index('rec_id', inplace = True)
        self.nodes = nodes_data
        self.det_count = detection_count
        self.noise_window = duration
        
        # create standard BIOTAS project directory if it does not already exist
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        
        self.data_dir = os.path.join(project_dir,'Data')                                # raw data goes her
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
       
        self.training_dir = os.path.join(self.data_dir,'Training_Files')
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
       
        self.output_dir = os.path.join(self.project_dir, 'Output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.figures_dir = os.path.join(self.output_dir, 'Figures')
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
        self.figure_ws = os.path.join(project_dir,'Output','Figures')                         

        # create a project database and write initial arrays to HDF
        self.initialize_hdf5()    
        
        
    def initialize_hdf5(self):
        '''Initialize an HDF5 database for a simulation'''
        
        hdf5 = h5py.File(self.db, 'a')
        
        # Create groups for organization (optional)
        if 'project_setup' not in hdf5:
            hdf5.create_group("project_setup")
            self.tags.to_hdf(self.db, key='/project_setup/tags', mode='a')
            self.receivers.to_hdf(self.db, key='/project_setup/receivers', mode='a')
            self.nodes.to_hdf(self.db, key='/project_setup/nodes', mode='a')

        if 'raw_data' not in hdf5:
            hdf5.create_group("raw_data")  
            
        if 'trained' not in hdf5:
            hdf5.create_group("trained")
            
        if 'classified' not in hdf5:
            hdf5.create_group("classified")
            
        if 'presence' not in hdf5:
            hdf5.create_group("presence")
            
        if 'overlapping' not in hdf5:
            hdf5.create_group('overlapping')
            
        if 'recaptures' not in hdf5:
            hdf5.create_group('recaptures')
            
        hdf5.close()
            
    def telem_data_import(self,
                          rec_id,
                          rec_type,
                          file_dir,
                          db_dir,
                          scan_time = 1, 
                          channels = 1, 
                          ant_to_rec_dict = None,
                          ka_format = False):
        # list raw data files
        tFiles = os.listdir(file_dir)
        
        # for every file call the correct text parser and import
        for f in tFiles:
            print ("start importing file %s"%(f))
            # get the complete file directory
            f_dir = os.path.join(file_dir,f)
            
            if rec_type == 'srx600' :
                parsers.srx600(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'srx800':
                parsers.srx800(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'srx1200':
                parsers.srx1200(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict, ka_format = 'True')
            
            elif rec_type == 'orion':
                parsers.orion_import(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'vr2':
                parsers.vr2_import(f_dir,db_dir,rec_id)
                
            elif rec_type == 'ares':
                parsers.ares(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)

            elif rec_type == 'PIT':
                parsers.PIT(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
                
            elif rec_type == 'PIT_Multiple':
                parsers.PIT_Multiple(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)

            else:
                print ("There currently is not an import routine created for this receiver type.  Please try again")

            
            print ("File %s imported"%(f))
        
        print ("Raw Telemetry Data Import Completed")
    
    def get_fish(self, rec_id, train = True, reclass_iter = None):
        
        tags_no_idx = self.tags.reset_index(drop = False)

        if reclass_iter == None and train == True:
            dat = pd.read_hdf(self.db, 
                              key = 'raw_data',
                              where = f'rec_id = "{rec_id}"')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[(dat.tag_type != 'TEST') & (dat.tag_type != 'BEACON')]
            dat = dat[(dat.tag_type != 'beacon') & (dat.tag_type != 'test')]
            
        elif reclass_iter == None and train == False:
            dat = pd.read_hdf(self.db, 
                              key = 'raw_data',
                              where = f'rec_id = "{rec_id}"')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[(dat.tag_type == 'study') | (dat.tag_type == 'STUDY')]

        else:
            itr = reclass_iter -1 
            dat = pd.read_hdf(self.db, 
                              key = 'classified',
                              where = f'(rec_id = "{rec_id}") & (iter == {itr})')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[dat.tag_type == 'study']
    
        return dat.freq_code.unique()
    
    def train(self, freq_code, rec_id):
        '''A class object for a training dataframe and related data objects.'''
    
        # Pull raw data 
        train_dat = pd.read_hdf(self.db,
                                'raw_data',
                                where=f'(freq_code == "{freq_code}") & (rec_id == "{rec_id}")')
        
        # Data management
        train_dat['time_stamp'] = pd.to_datetime(train_dat.time_stamp)
        train_dat['epoch'] = np.round((train_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'), 6)
        train_dat.sort_values(by='epoch', inplace=True)
        
        train_dat.drop_duplicates(subset='time_stamp', keep='first', inplace=True)
        
        # Object variables
        if self.receivers.index.dtype != 'object':
            rec_id = np.int64(rec_id)
        rec_type = self.receivers.at[rec_id, 'rec_type']
    
        # Detection class
        if freq_code in self.study_tags:
            plausible = 1
        else:
            plausible = 0
    
        # Get rate
        if freq_code in self.tags.index:
            pulse_rate = self.tags.at[freq_code, 'pulse_rate']
        else:
            pulse_rate = 673.
    
        mort_rate = 8888.
        train_dat['detection'] = np.repeat(plausible, len(train_dat))
        train_dat['lag'] = train_dat.epoch.diff()
        train_dat['lag_diff'] = train_dat.lag.diff()
        
        # if freq_code in self.tags.index:
        #     print ('check')
            
        # Apply the optimized detection history function to the entire dataset at once
        detection_history, hit_ratio_arr, cons_det_arr, max_count_arr = predictors.detection_history(
            train_dat['epoch'].values,
            pulse_rate,
            self.det_count,
            train_dat['channels'].values,
            train_dat['scan_time'].values,
        )
    
        # Convert detection history arrays to concatenated strings outside Numba
        det_hist_string_arr = np.array([''.join(row.astype(str)) for row in detection_history])
        
        # Assign back to the DataFrame
        train_dat['det_hist'] = det_hist_string_arr
        train_dat['hit_ratio'] = hit_ratio_arr
        train_dat['cons_det'] = cons_det_arr
        train_dat['cons_length'] = max_count_arr
    
        train_dat.fillna(value=9999999, inplace=True)
        
        # Ensure data types are correct
        try:
            train_dat = train_dat.astype({'power': 'float32', 
                                          'time_stamp': 'datetime64[ns]',
                                          'epoch': 'float32',
                                          'freq_code': 'object',
                                          'noise_ratio': 'float32',
                                          'scan_time': 'int32',
                                          'channels': 'int32',
                                          'rec_type': 'object',
                                          'rec_id': 'object',
                                          'detection': 'int32',
                                          'lag': 'float32',
                                          'lag_diff': 'float32',
                                          'det_hist': 'object',
                                          'hit_ratio': 'float32',
                                          'cons_det': 'int32',
                                          'cons_length': 'float32'})
        except ValueError:
            print('debug - check datatypes')
            
        # Append to HDF5
        with pd.HDFStore(self.db, mode='a') as store:
            store.append(key='trained',
                         value=train_dat, 
                         format='table', 
                         index=False,
                         min_itemsize={'freq_code': 20,
                                       'rec_type': 20,
                                       'rec_id': 20,
                                       'det_hist': 20},
                         append=True, 
                         chunksize=1000000)
    
        print(f'Fish {freq_code} trained at receiver {rec_id}, plausibility: {plausible}')

    def training_summary(self,rec_type,site = None):
        # initialize some variables

        # connect to database and get data

        trained_dat = pd.read_hdf(self.db,key = 'trained')#, mode = 'r')
        trained_dat = trained_dat[(trained_dat.rec_type == rec_type)] 
        #train_dat.reset_index(inplace = True)

        # if site != None:
        #     for rec_id in site:
        #         trained_dat = trained_dat[(trained_dat.rec_id == rec_id)] 

        det_class_count = trained_dat.groupby('detection')['detection'].count().to_frame()

        print ("")
        print ("Training summary statistics report")
        print ("The algorithm collected %s detections from %s %s receivers"%(len(trained_dat),len(trained_dat.rec_id.unique()),rec_type))
        print ("----------------------------------------------------------------------------------")
        print ("")
        print ("%s detection clas statistics:"%(rec_type) )
        try:
            print ("The prior probability that a detection was true was %s"%((round(float(det_class_count.at[1,'detection'])/float(det_class_count.sum()),3))))
        except KeyError:
            print ("No known true detections found")
            pass
        try:
            print ("The prior probability that a detection was false positive was %s"%((round(float(det_class_count.at[0,'detection'])/float(det_class_count.sum()),3))))
        except KeyError:
            print ("No known true detections found")
            pass

        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("")
        trained_dat['detection'] = trained_dat.detection.astype('str')
        sta_class_count = trained_dat.groupby(['rec_id','detection'])['detection'].count().rename('det_class_count').to_frame().reset_index()
        recs = sorted(sta_class_count.rec_id.unique())
        print ("Detection Class Counts Across Stations")
        print ("             Known          Known")
        print ("             False          True")
        print ("       ______________________________")
        print ("      |              |              |")
        for i in recs:
            trues = sta_class_count[(sta_class_count.rec_id == i) & (sta_class_count.detection == '1')]
            falses = sta_class_count[(sta_class_count.rec_id == i) & (sta_class_count.detection == '0')]
            if len(trues) > 0 and len(falses) > 0:
                print ("%6s|   %8s   |   %8s   |"%(i,falses.det_class_count.values[0],trues.det_class_count.values[0]))
            elif len(trues) == 0 and len(falses) > 0:
                print ("%6s|   %8s   |   %8s   |"%(i,falses.det_class_count.values[0],0))
            else:
                try:
                    print ("%6s|   %8s   |   %8s   |"%(i,0,trues.det_clas_count.values[0]))

                except AttributeError:
                    print ("%6s|   %8s   |   %8s   |"%(i,0,0))

        print ("      |______________|______________|")
        print ("")
        print ("----------------------------------------------------------------------------------")
        print ("Compiling Figures")
        # get data by detection class for side by side histograms
        trained_dat['power']= trained_dat.power.astype(float)
        trained_dat['lag_diff'] = trained_dat.lag_diff.astype(float)
        trained_dat['cons_length'] = trained_dat.cons_length.astype(float)
        trained_dat['noise_ratio'] = trained_dat.noise_ratio.astype(float)

        trues = trained_dat[trained_dat.detection == '1']
        falses = trained_dat[trained_dat.detection == '0']
        # plot hit ratio histograms by detection class
        hitRatioBins =np.linspace(0,1.0,11)

        # plot signal power histograms by detection class
        minPower = trained_dat.power.min()//5 * 5
        maxPower = trained_dat.power.max()//5 * 5
        powerBins =np.arange(minPower,maxPower+20,10)

        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,6)

        # Consecutive Record Length ?
        conBins =np.arange(0,12,1)

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

        # make lattice plot for pubs
        figSize = (3,7)
        plt.figure()
        fig, axs = plt.subplots(5,2,tight_layout = True,figsize = figSize)
        # hit ratio
        axs[0,1].hist(trues.hit_ratio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].hist(falses.hit_ratio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].set_xlabel('Hit Ratio')
        axs[0,1].set_title('True')
        axs[0,1].set_xlabel('Hit Ratio')
        axs[0,0].set_title('False Positive')
        axs[0,0].set_title('A',loc = 'left')

        # consecutive record length
        axs[1,1].hist(trues.cons_length.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].hist(falses.cons_length.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].set_xlabel('Consecutive Hit Length')
        axs[1,1].set_xlabel('Consecutive Hit Length')
        axs[1,0].set_title('B',loc = 'left')

        # power
        axs[2,1].hist(trues.power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].hist(falses.power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].set_xlabel('Signal Power')
        axs[2,1].set_xlabel('Signal Power')
        axs[2,0].set_ylabel('Probability Density')
        axs[2,0].set_title('C',loc = 'left')

        # noise ratio
        axs[3,1].hist(trues.noise_ratio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].hist(falses.noise_ratio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].set_xlabel('Noise Ratio')
        axs[3,1].set_xlabel('Noise Ratio')
        axs[3,0].set_title('D',loc = 'left')

        # lag diff
        axs[4,1].hist(trues.lag_diff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].hist(falses.lag_diff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].set_xlabel('Lag Differences')
        axs[4,1].set_xlabel('Lag Differences')
        axs[4,0].set_title('E',loc = 'left')
        
        plt.show()
        

        plt.savefig(os.path.join(self.figure_ws,"%s_lattice_train.png"%(rec_type)),bbox_inches = 'tight', dpi = 900)
        
    def undo_training(self, rec_id, freq_code=None):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'trained' in store:
                df = store['trained']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)
                if freq_code is not None:
                    condition &= (df['freq_code'] == freq_code)

                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('trained')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('trained',
                          df,
                          format='table',
                          data_columns=True, 
                          append = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20,
                                          'det_hist':20})      

    def undo_import(self, rec_id):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'raw_data' in store:
                df = store['raw_data']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)

                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('raw_data')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('raw_data',
                          df,
                          format='table',
                          data_columns=True, 
                          append = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20})  
        
    def create_training_data(self, rec_type=None, reclass_iter=None, rec_list=None):
        """
        Function to create a training dataset for the current round of classification.
        The function supports multiple pathways for generating training data, including 
        using a receiver list (rec_list) and incorporating reclassification methods.
    
        Parameters
        ----------
        rec_type : str, optional
            The type of receiver to filter the data by. This restricts the training data 
            to a specific receiver type (e.g., 'orion', 'srx800'). If not provided and 
            `rec_list` is used, it is ignored.
        reclass_iter : int, optional
            Iteration number for reclassification. If provided, the function pulls the 
            previous classification data and incorporates known false positives and 
            assumed true positives.
        rec_list : list of str, optional
            A list of receiver IDs to filter the data by. If provided, the function 
            queries the HDF database using this list directly rather than the receiver 
            type (`rec_type`).
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the training data for the classification process, 
            incorporating any previous classifications if applicable.
    
        Notes
        -----
        - If both `rec_type` and `rec_list` are provided, the function will prioritize 
          the `rec_list` to restrict the training data.
        - Reclassification logic is based on contributions from T. Castro-Santos.
        """
        if rec_list is not None:
            # Construct the query for multiple receiver IDs using the OR operator
            rec_query = ' | '.join([f'rec_id == "{rec_id}"' for rec_id in rec_list])
            train_dat = pd.read_hdf(self.db, 'trained', where=rec_query)
        elif rec_type is not None:
            # Query based on receiver type directly
            train_dat = pd.read_hdf(self.db, 'trained', where=f'rec_type == "{rec_type}"')
        else:
            raise ValueError("Either 'rec_type' or 'rec_list' must be provided to create training data.")
    
        # Handling reclassification if this is not the first iteration
        if reclass_iter is not None:
            last_class = reclass_iter - 1
            
            # Load the classified dataset and filter by iteration
            class_dat = pd.read_hdf(self.db, 'classified', where=f'iter == {last_class}')
            
            # Further restrict classified data to the receiver list if rec_list is provided
            if rec_list is not None:
                class_query = ' | '.join([f'rec_id == "{rec_id}"' for rec_id in rec_list])
                class_dat = class_dat.query(class_query)
            
            # Selecting relevant columns for the training dataset
            columns = ['test', 'freq_code', 'power', 'noise_ratio', 'lag', 
                       'lag_diff', 'cons_length', 'cons_det', 'det_hist', 
                       'hit_ratio', 'rec_type', 'epoch']
            
            class_dat = class_dat[columns]
            class_dat.rename(columns={'test': 'detection'}, inplace=True)
            
            # Separate known falses (train_dat) and assumed trues (class_dat)
            train_dat = train_dat[train_dat['detection'] == 0]
            class_dat = class_dat[class_dat['detection'] == 1]
    
            # Append the classified data to the training data
            train_dat = pd.concat([train_dat, class_dat], ignore_index=True)
    
        return train_dat



    
    def reclassify(self, project, rec_id, threshold_ratio, likelihood_model, rec_type=None, rec_list=None):
        """
        Reclassifies fish in a project based on user-defined criteria and threshold ratios.
    
        Parameters
        ----------
        project : object
            The project object that contains methods for managing and classifying fish data.
            
        rec_id : int or str
            The unique identifier for the receiver to be reclassified.
            
        threshold_ratio : float
            The threshold ratio used for determining classification criteria.
            
        likelihood_model : list of str
            The fields to use as the likelihood model for classification.
            
        rec_type : str, optional
            The type of receiver being processed (e.g., 'srx1200', 'orion').
            
        rec_list : list of str, optional
            A list of receiver IDs to filter the data by, used for creating training data.
    
        Notes
        -----
        - The classification process involves interactive user input to determine if additional
          iterations are needed.
        - The fields used for classification are hardcoded as ['hit_ratio', 'cons_length',
          'noise_ratio', 'power', 'lag_diff'].
        """
        class_iter = None
        
        while True:
            # Get a list of fish to iterate over
            fishes = project.get_fish(rec_id=rec_id, train=False, reclass_iter=class_iter)
            
            # Generate training data for the classifier
            training_data = project.create_training_data(rec_type=rec_type, reclass_iter=class_iter, rec_list=rec_list)
    
            # Iterate over fish and classify
            for fish in fishes:
                project.classify(fish, rec_id, likelihood_model, training_data, class_iter, threshold_ratio)
            
            # Generate summary statistics
            project.classification_summary(rec_id, class_iter)
            
            # Show the figures and block execution until they are closed
            plt.show(block=True)
            
            # Ask the user if they need another iteration
            user_input = input("Do you need another classification iteration? (yes/no): ").strip().lower()
            
            if user_input in ['yes', 'y']:
                # If yes, increase class_iter and reclassify
                if class_iter is None:
                    class_iter = 2
                else:
                    class_iter += 1
            elif user_input in ['no', 'n']:
                # If no, break the loop
                print("Classification process completed.")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")


    def classify(self,
                 freq_code,
                 rec_id,
                 fields,
                 training_data,
                 reclass_iter = None,
                 threshold_ratio = None):
        
        # get rates
        try:
            pulse_rate = self.tags.at[freq_code,'pulse_rate']
        except KeyError: 
            pulse_rate = 9999.
        try:
            mort_rate = self.tags.at[freq_code,'mort_rate']
        except KeyError:
            mort_rate = 9999.
        
        # get data
        if reclass_iter == None:
            class_dat = pd.read_hdf(self.db, 
                                    'raw_data',
                                    where = f'(freq_code == "{freq_code}") & (rec_id == "{rec_id}")')
            
            columns = ['freq_code','epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        else:
            last_class = reclass_iter - 1
            class_dat = pd.read_hdf(self.db,
                                    'classified',
                                    where = f'(freq_code == "{freq_code}") & (rec_id == "{rec_id}") & (iter == {last_class} & (test == 1))')

            columns = ['freq_code','epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        class_dat = class_dat.drop_duplicates()
        
        if len(class_dat) > 0:
            # do some data management when importing training dataframe
            class_dat['time_stamp'] = pd.to_datetime(class_dat['time_stamp'])
            class_dat['epoch'] = np.round((class_dat.time_stamp - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s'), 6)

            class_dat.sort_values(by = 'time_stamp', inplace = True)
            class_dat['epoch'] = class_dat.epoch.values.astype(np.int64)
            class_dat = class_dat.drop_duplicates(subset = 'time_stamp')
            
            # calculate predictors
            class_dat['lag'] = class_dat.epoch.diff()
            class_dat['lag_diff'] = class_dat.lag.diff()
            class_dat.fillna(value = 99999999, inplace = True)
            
            # Apply the optimized detection history function to the entire dataset at once
            detection_history, hit_ratio_arr, cons_det_arr, max_count_arr = predictors.detection_history(
                class_dat['epoch'].values,
                pulse_rate,
                self.det_count,
                class_dat['channels'].values,
                class_dat['scan_time'].values
            )
        
            # Convert detection history arrays to concatenated strings outside Numba
            det_hist_string_arr = np.array([''.join(row.astype(str)) for row in detection_history])
            
            # Assign back to the DataFrame
            class_dat['det_hist'] = det_hist_string_arr
            class_dat['hit_ratio'] = hit_ratio_arr
            class_dat['cons_det'] = cons_det_arr
            class_dat['cons_length'] = max_count_arr
            
            # class_dat['series_hit'] = predictors.series_hit(class_dat.lag.values,
            #                                                 pulse_rate,
            #                                                 mort_rate,
            #                                                 'A')
            
            # categorize predictors
            hit_ratio, power, lag, con_len, noise \
                = naive_bayes.bin_predictors(class_dat.hit_ratio,
                                             class_dat.power,
                                             class_dat.lag_diff,
                                             class_dat.cons_length,
                                             class_dat.noise_ratio)
                
            binned_predictors = {'hit_ratio':hit_ratio,
                                 'power':power,
                                 'lag_diff':lag,
                                 'cons_length':con_len,
                                 'noise_ratio':noise}
    
            # calculate prior - from training data
            prior = naive_bayes.calculate_priors(training_data.detection.values)
            
            # calculate likelihoods - also from training data
            likelihoods = {True:dict(),False:dict()}
            labeled_array = training_data.detection.values
            
            # loop over assumptions and fields
            for assumption in [True, False]:
                for field in fields:
                    # get observation array 
                    observation_array = training_data[field].values
    
                    # calculate likelihood
                    likelihood = naive_bayes.calculate_likelihood(observation_array, 
                                                                  labeled_array, 
                                                                  assumption, 
                                                                  class_dat[field])
    
                    # add it to the dictionary
                    likelihoods[assumption][field] = likelihood
            
            likelihood_T = np.repeat(1., len(class_dat))  
            for pred in likelihoods[True]:
                likelihood_T = likelihood_T * likelihoods[True][pred]
    
            likelihood_F = np.repeat(1., len(class_dat))  
            for pred in likelihoods[False]:
                likelihood_F = likelihood_F * likelihoods[False][pred]
                
                
            # calculate the evidence
            observation_arrays = []
            # for each field, calculate the probability of observing each bin
            for field in fields:
                if field != 'cons_det' and field != 'series_hit': 
                    observation_arrays.append(binned_predictors[field])   
                else:
                    observation_arrays.append(class_dat[field].values)   
    
            evidence = naive_bayes.calculate_evidence(observation_arrays)
           
            # calculate posteriors
            posterior_T = naive_bayes.calculate_posterior({True:np.repeat(prior[0],evidence.shape),
                                                False:np.repeat(prior[1],evidence.shape)},
                                               evidence,
                                               likelihoods,
                                               True)
            posterior_F = naive_bayes.calculate_posterior({True:np.repeat(prior[0],evidence.shape),
                                                False:np.repeat(prior[1],evidence.shape)},
                                               evidence,
                                               likelihoods,
                                               False)
    
            # Classify
            classification = naive_bayes.classify_with_threshold(posterior_T,
                                                                 posterior_F,
                                                                 1.0)
            
            # now add these arrays to the dataframe and export to hdf5
            class_dat['likelihood_T'] = likelihood_T
            class_dat['likelihood_F'] = likelihood_F
            class_dat['posterior_T'] = posterior_T
            class_dat['posterior_F'] = posterior_F
            class_dat['test'] = classification
            if reclass_iter == None:
                reclass_iter = 1
            class_dat['iter'] = np.repeat(reclass_iter,len(class_dat))
    
            # keep it tidy cuz hdf is fragile
            class_dat = class_dat.astype({'freq_code': 'object',
                                          'epoch': 'float32',
                                          'rec_id': 'object',
                                          'time_stamp': 'datetime64[ns]',
                                          'power': 'float32', 
                                          'noise_ratio': 'float32',
                                          'scan_time': 'int32',
                                          'channels': 'int32',
                                          'rec_type': 'object',
                                          'lag': 'float32',
                                          'lag_diff': 'float32',
                                          'det_hist': 'object',
                                          'hit_ratio': 'float32',
                                          'cons_det': 'int32',
                                          'cons_length': 'float32',
                                          'likelihood_T': 'float32',
                                          'likelihood_F': 'float32',
                                          'posterior_T': 'float32',
                                          'posterior_F': 'float32',
                                          'test': 'int32',
                                          'iter': 'int32'})
            
            with pd.HDFStore(self.db, mode='a') as store:
                store.append(key = 'classified',
                             value = class_dat, 
                             format = 'table', 
                             index = False,
                             min_itemsize = {'freq_code':20,
                                             'rec_type':20,
                                             'rec_id':20,
                                             'det_hist':20},
                             append = True, 
                             data_columns = True,
                             chunksize = 1000000)
            
            # export
            #class_dat.to_csv(os.path.join(self.output_dir,'freq_code_%s_rec_%s_class_%s.csv'%(freq_code, rec_id, reclass_iter)))
            
            print ('Fish %s at receiver %s classified'%(freq_code,rec_id))
            # next step looks at results
        
        # else:
        #     print ('there are no recoreds to classify for fish %s at %s'%(freq_code, rec_id))
        
    def classification_summary(self,rec_id,reclass_iter = None): 
        '''if this is not the initial classification we need the trues from the last 
        last classification and falses from the first'''
                
        if reclass_iter == None:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified',
                                         where = f'(iter == 1) & (rec_id == "{rec_id}")')
        else:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified',
                                         where = f'(iter == {reclass_iter}) & (rec_id == "{rec_id}")')
                    
        print ("")
        print ("Classification summary statistics report %s"%(rec_id))
        print ("----------------------------------------------------------------------------------")
        det_class_count = classified_dat.groupby('test')['test'].count().to_frame()
        if len(det_class_count)>1:
            print ("")
            print ("%s detection class statistics:"%(rec_id))
            print ("The probability that a detection was classified as true was %s"%((round(float(det_class_count.at[1,'test'])/float(det_class_count.sum()),3))))
            print ("The probability that a detection was classified as false positive was %s"%((round(float(det_class_count.at[0,'test'])/float(det_class_count.sum()),3))))
            print ("")
            print ("----------------------------------------------------------------------------------")
            print ("")
            sta_class_count = classified_dat.groupby(['rec_id','test'])['test'].count().to_frame()#.reset_index(drop = False)
            recs = list(set(sta_class_count.index.levels[0]))
            print ("Detection Class Counts Across Stations")
            print ("          Classified     Classified")
            print ("             False          True")
            print ("       ______________________________")
            print ("      |              |              |")
            for i in recs:
                print ("%6s|   %8s   |   %8s   |"%(i,sta_class_count.loc[(i,0)].values[0],sta_class_count.loc[(i,1)].values[0]))
            print ("      |______________|______________|")
            print ("")
            print ("----------------------------------------------------------------------------------")
            print ("----------------------------------------------------------------------------------")

            # Plot the log likelihood ratio
            classified_dat['log_posterior_ratio'] = np.log10(classified_dat.posterior_T / classified_dat.posterior_F)
            
            # Binning and other parameters
            hit_ratio_bins = np.linspace(0, 1.0, 11)
            con_length_bins = np.arange(1, 12, 1)
            power_bins = np.arange(50, 110, 10)
            noise_bins = np.linspace(0, 1.1, 11)
            lag_bins = np.arange(-100, 110, 20)
            post_ratio_bins = np.linspace(classified_dat.log_posterior_ratio.min(), classified_dat.log_posterior_ratio.max(), 10)
            
            trues = classified_dat[classified_dat.test == 1]
            falses = classified_dat[classified_dat.test == 0]
            
            # Create a grid of subplots (3 rows x 4 columns)
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10), dpi=300)
            
            # Function to set font sizes
            def set_fontsize(ax, fontsize=6):
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fontsize)
            
            # Plot hit ratio
            axes[0, 0].hist(falses.hit_ratio.values, hit_ratio_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 0].set_xlabel('Hit Ratio')
            axes[0, 0].set_ylabel('Probability Density')
            axes[0, 0].set_title('Hit Ratio - False Positive')
            set_fontsize(axes[0, 0])
            
            axes[0, 1].hist(trues.hit_ratio.values, hit_ratio_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 1].set_xlabel('Hit Ratio')
            axes[0, 1].set_title('Hit Ratio - Valid')
            set_fontsize(axes[0, 1])
            
            # Plot consecutive record length
            axes[0, 2].hist(falses.cons_length.values, con_length_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 2].set_xlabel('Consecutive Hit Length')
            axes[0, 2].set_ylabel('Probability Density')
            axes[0, 2].set_title('Consecutive Hit Length - False Positive')
            set_fontsize(axes[0, 2])
            
            axes[0, 3].hist(trues.cons_length.values, con_length_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 3].set_xlabel('Consecutive Hit Length')
            axes[0, 3].set_title('Consecutive Hit Length - Valid')
            set_fontsize(axes[0, 3])
            
            # Plot power
            axes[1, 0].hist(falses.power.values, power_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 0].set_xlabel('Signal Power')
            axes[1, 0].set_ylabel('Probability Density')
            axes[1, 0].set_title('Signal Power - False Positive')
            set_fontsize(axes[1, 0])
            
            axes[1, 1].hist(trues.power.values, power_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 1].set_xlabel('Signal Power')
            axes[1, 1].set_title('Signal Power - Valid')
            set_fontsize(axes[1, 1])
            
            # Plot noise ratio
            axes[1, 2].hist(falses.noise_ratio.values, noise_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 2].set_xlabel('Noise Ratio')
            axes[1, 2].set_ylabel('Probability Density')
            axes[1, 2].set_title('Noise Ratio - False Positive')
            set_fontsize(axes[1, 2])
            
            axes[1, 3].hist(trues.noise_ratio.values, noise_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 3].set_xlabel('Noise Ratio')
            axes[1, 3].set_title('Noise Ratio - Valid')
            set_fontsize(axes[1, 3])
            
            # Plot lag differences
            axes[2, 0].hist(falses.lag_diff.values, lag_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 0].set_xlabel('Lag Differences')
            axes[2, 0].set_ylabel('Probability Density')
            axes[2, 0].set_title('Lag Differences - False Positive')
            set_fontsize(axes[2, 0])
            
            axes[2, 1].hist(trues.lag_diff.values, lag_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 1].set_xlabel('Lag Differences')
            axes[2, 1].set_title('Lag Differences - Valid')
            set_fontsize(axes[2, 1])

            # Plot log posterior ratio
            axes[2, 2].hist(falses.log_posterior_ratio.values, bins=20, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 2].set_xlabel('Log Posterior Ratio')
            axes[2, 2].set_ylabel('Probability Density')
            axes[2, 2].set_title('Log Posterior Ratio - False Positive')
            set_fontsize(axes[2, 2])
            
            axes[2, 3].hist(trues.log_posterior_ratio.values, bins=20, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 3].set_xlabel('Log Posterior Ratio')
            axes[2, 3].set_title('Log Posterior Ratio - Valid')
            set_fontsize(axes[2, 3])
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        else:
           print("There were insufficient data to quantify summary statistics")
           print("All remaining were classified as %s suggesting there is no more improvement in the model"%(det_class_count.index[0]))

    def undo_classification(self, rec_id, freq_code=None, class_iter=None):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'classified' in store:
                df = store['classified']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)
                if freq_code is not None:
                    condition &= (df['freq_code'] == freq_code)
                if class_iter is not None:
                    condition &= (df['class_iter'] == class_iter)
        
                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('classified')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('classified',
                          df,
                          format='table',
                          index = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20,
                                          'det_hist':20},
                          data_columns=True,
                          append = False)       

    def undo_bouts(self, rec_id):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'presence' in store:
                df = store['presence'] 
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)
        
                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                df = df.astype({'freq_code': 'object',
                                'epoch': 'float32',
                                'rec_id': 'object',
                                'class': 'object',
                                'bout_no':'int32',
                                'det_lag':'int32'})
                
                # Delete the existing table
                store.remove('presence')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('presence', 
                          df,
                          format='table', 
                          index = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_id':20,
                                          'class':20},
                          data_columns=True, 
                          append = False)   

    def make_recaptures_table(self, export=True, pit_study=False):
        '''Creates a recaptures key in the HDF5 file, iterating over receivers to manage memory.'''
        if pit_study==False:
            # Convert release dates to datetime if not already done
            self.tags['rel_date'] = pd.to_datetime(self.tags['rel_date'])
            tags_copy = self.tags.copy()
            for rec in self.receivers.index:
            #for rec in ['R14a','R14b']:
                print(f'Processing receiver {rec}...')
    
                # Read classified data for this receiver as a Dask DataFrame
                # Reading the data (assuming self.db and rec are predefined variables)
                rec_dat = dd.read_hdf(self.db, key='classified')
                
                # Filter for specific rec_id and convert to pandas DataFrame
                rec_dat = rec_dat[rec_dat['rec_id'] == rec].compute()
                
                # Convert 'timestamp' column to datetime
                rec_dat['time_stamp'] = pd.to_datetime(rec_dat['time_stamp'])
                
                # Calculate seconds since Unix epoch
                rec_dat['epoch'] = (rec_dat['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                print(f"Length of rec_dat after initial load: {len(rec_dat)}")
    
                # Merge with release dates to filter out data before release
                rec_dat = rec_dat.merge(tags_copy, left_on='freq_code', right_index=True)
                rec_dat = rec_dat[rec_dat['time_stamp'] >= rec_dat['rel_date']]
                print(f"Length of rec_dat after merging with release dates: {len(rec_dat)}")
    
                # Reset index to avoid ambiguity between index and column labels
                if 'freq_code' in rec_dat.columns and 'freq_code' in rec_dat.index.names:
                    rec_dat = rec_dat.reset_index(drop=True)
    
                # Filter by latest iteration and valid test
                idxmax_values = rec_dat.groupby(['freq_code', 'rec_id'])['iter'].idxmax()
                rec_dat = rec_dat.loc[idxmax_values]
                rec_dat = rec_dat[rec_dat['test'] == 1]
                #print(f"Columns in rec_dat after filtering by iter and test: {rec_dat.columns}")
                print(f"Length of rec_dat after filtering by iter and test: {len(rec_dat)}")

                # Check if 'presence' exists before trying to read it
                with pd.HDFStore(self.db, mode='r') as store:
                    presence_exists = 'presence' in store.keys()
                    
                if presence_exists:
                    try:
                        presence_data = dd.read_hdf(self.db, key='presence')
                        presence_data = presence_data[presence_data['rec_id'] == rec].compute()
                        presence_data = presence_data[presence_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        presence_data = presence_data[['freq_code', 'epoch', 'rec_id', 'bout_no']]
                        print(f"Length of presence_data: {len(presence_data)}")

                    except KeyError:
                        print(f"WARNING: No presence data found for receiver {rec}. Skipping presence merge.")
                else:
                    print(f"WARNING: 'presence' key not found in HDF5. Skipping presence merge.")                    
    
                # Read overlap data
                with pd.HDFStore(self.db, mode='r') as store:
                    overlap_exists = 'overlapping' in store.keys()
        
                if overlap_exists:
                    try:
                        overlap_data = dd.read_hdf(self.db, key='overlapping')
                        overlap_data = overlap_data[overlap_data['rec_id'] == rec].compute()
                        overlap_data = overlap_data[overlap_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        overlap_data = overlap_data.groupby(['freq_code', 'epoch', 'rec_id'])['overlapping'].max().reset_index()
                        print(f"Length of overlap_data: {len(overlap_data)}")

                    except KeyError:
                        print(f"WARNING: No overlap data found for receiver {rec}. Skipping overlap merge.")
                else:
                    
                    print(f"WARNING: 'overlapping' key not found in HDF5. Skipping overlap merge.")
    
                # Merge with presence data
                if presence_exists:
                    rec_dat = rec_dat.merge(presence_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                    rec_dat['bout_no'] = rec_dat['bout_no'].fillna(0).astype(int)
                else:
                    rec_dat['bout_no'] = 0
    
                # Merge with overlap data
                if overlap_exists:
                    rec_dat = rec_dat.merge(overlap_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                    rec_dat['overlapping'] = rec_dat['overlapping'].fillna(0).astype(int)
                else:
                    rec_dat['overlapping'] = 0
                
                #rec_dat = rec_dat[rec_dat['overlapping'] != 1]
    
                #print(f"Columns in rec_dat after merging with presence and overlap data: {rec_dat.columns}")
                print(f"Length of rec_dat after merging with presence and overlap data: {len(rec_dat)}")
    
                # Check for required columns
                required_columns = ['freq_code', 'rec_id', 'epoch', 'time_stamp', 'power', 'noise_ratio',
                                    'lag', 'det_hist', 'hit_ratio', 'cons_det', 'cons_length', 
                                    'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping']
                
                missing_columns = [col for col in required_columns if col not in rec_dat.columns]
                if missing_columns:
                    print(f"The following required columns are missing: {missing_columns}")
                    continue
    
                # Sort by freq code and epoch
                rec_dat = rec_dat.sort_values(by=['freq_code', 'epoch'], ascending=[True, True])
    
                # Keep only the necessary columns (including handling missing columns)
                available_columns = [col for col in required_columns if col in rec_dat.columns]
                rec_dat = rec_dat[available_columns]
    
                # Ensure correct data types
                rec_dat = rec_dat.astype({
                    'freq_code': 'object',
                    'epoch': 'float32',
                    'rec_id': 'object',
                    'time_stamp': 'datetime64[ns]',
                    'power': 'float32', 
                    'noise_ratio': 'float32',
                    'lag': 'float32',
                    'det_hist': 'object',
                    'hit_ratio': 'float32',
                    'cons_det': 'int32',
                    'cons_length': 'float32',
                    'likelihood_T': 'float32',
                    'likelihood_F': 'float32',
                    'bout_no': 'int32',
                    'overlapping': 'int32'
                })
    
                # Prompt to confirm importing the data
                #print(f"Final columns in rec_dat: {rec_dat.columns}")
                print(f"Final length of rec_dat: {len(rec_dat)}")
                import_confirmation = input("Do you want to import the data into the database? (yes/no): ").strip().lower()
                if import_confirmation != 'yes':
                    print("Data import canceled.")
                    return
    
                # Append to the HDF5 file
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(key='recaptures', value=rec_dat, format='table', 
                                 index=False, min_itemsize={'freq_code': 20, 'rec_id': 20, 'det_hist': 20},
                                 append=True, chunksize=1000000, data_columns=True)
    
                print(f'Recaps for receiver {rec} compiled.')
                
        else:
            # Loop over each receiver in self.receivers
            for rec in self.receivers.index:
                print(f'Processing receiver {rec} (PIT study)...')
        
                # Read PIT data (already parsed from your text files) from /raw_data in HDF5
                try:
                    pit_data = pd.read_hdf(self.db, key='raw_data')
                except KeyError:
                    print("No 'raw_data' key found in the HDF5 file.")
                    continue
        
                # Filter rows so that only the specified receiver is kept
                pit_data = pit_data[pit_data['rec_id'] == rec]
                print(f"Length of pit_data after filtering for {rec}: {len(pit_data)}")
        
                # Add any missing columns to align with the acoustic (non-PIT) columns
                missing_cols = [
                    'lag', 'det_hist', 'hit_ratio', 'cons_det', 'cons_length',
                    'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping'
                ]
                for col in missing_cols:
                    if col not in pit_data.columns:
                        pit_data[col] = 0
        
                # Check if 'presence' exists before trying to read it
                with pd.HDFStore(self.db, mode='r') as store:
                    presence_exists = 'presence' in store.keys()
        
                if presence_exists:
                    try:
                        presence_data = dd.read_hdf(self.db, key='presence')
                        presence_data = presence_data[presence_data['rec_id'] == rec].compute()
                        presence_data = presence_data[presence_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        presence_data = presence_data[['freq_code', 'epoch', 'rec_id', 'bout_no']]
                        
                        if not presence_data.empty:
                            pit_data = pit_data.merge(presence_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                            pit_data['bout_no'] = pit_data['bout_no'].fillna(0).astype(int)
                    except KeyError:
                        print(f"WARNING: No presence data found for receiver {rec}. Skipping presence merge.")
                else:
                    print(f"WARNING: 'presence' key not found in HDF5. Skipping presence merge.")
        
                # Check if 'overlapping' exists before trying to read it
                with pd.HDFStore(self.db, mode='r') as store:
                    overlap_exists = 'overlapping' in store.keys()
        
                if overlap_exists:
                    try:
                        overlap_data = dd.read_hdf(self.db, key='overlapping')
                        overlap_data = overlap_data[overlap_data['rec_id'] == rec].compute()
                        overlap_data = overlap_data[overlap_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        overlap_data = overlap_data.groupby(['freq_code', 'epoch', 'rec_id'])['overlapping'].max().reset_index()
        
                        if not overlap_data.empty:
                            pit_data = pit_data.merge(overlap_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                            pit_data['overlapping'] = pit_data['overlapping'].fillna(0).astype(int)
                    except KeyError:
                        print(f"WARNING: No overlap data found for receiver {rec}. Skipping overlap merge.")
                else:
                    print(f"WARNING: 'overlapping' key not found in HDF5. Skipping overlap merge.")
        
                # Sort PIT data by freq_code and epoch
                pit_data = pit_data.sort_values(['freq_code', 'epoch'])
        
                # Keep only the columns needed in `recaptures`
                required_columns = [
                    'freq_code', 'rec_id', 'epoch', 'time_stamp', 'power', 'noise_ratio', 'lag', 'det_hist',
                    'hit_ratio', 'cons_det', 'cons_length', 'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping'
                ]
                pit_data = pit_data[[c for c in required_columns if c in pit_data.columns]]
        
                # Convert each column to the correct dtype
                dtypes_map = {
                    'freq_code': 'object', 'rec_id': 'object', 'epoch': 'float32',
                    'time_stamp': 'datetime64[ns]', 'power': 'float32', 'noise_ratio': 'float32',
                    'lag': 'float32', 'det_hist': 'object', 'hit_ratio': 'float32',
                    'cons_det': 'int32', 'cons_length': 'float32', 'likelihood_T': 'float32',
                    'likelihood_F': 'float32', 'bout_no': 'int32', 'overlapping': 'int32'
                }
                for col, dt in dtypes_map.items():
                    if col in pit_data.columns:
                        pit_data[col] = pit_data[col].astype(dt)
        
                # Confirm with user before appending PIT data into 'recaptures'
                confirm = input("Import PIT data? (yes/no): ").strip().lower()
                if confirm != 'yes':
                    print("Import canceled.")
                    return
        
                # Convert 'det_hist' to string to avoid serialization issues
                if 'det_hist' in pit_data.columns:
                    pit_data['det_hist'] = pit_data['det_hist'].astype(str)        
        
                # Append PIT data to 'recaptures' in HDF5
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(
                        key='recaptures',
                        value=pit_data,
                        format='table',
                        index=False,
                        min_itemsize={'freq_code': 20, 'rec_id': 20, 'det_hist': 20},
                        append=True,
                        chunksize=1000000,
                        data_columns=True
                    )
        
                print(f"PIT recaps for receiver {rec} compiled.")


        if export:
            print("Exporting to CSV...")
            rec_data = dd.read_hdf(self.db, 'recaptures').compute()
            rec_data.to_csv(os.path.join(self.output_dir,'recaptures.csv'), index=False)
            print("Export completed.")

                
    def undo_recaptures(self):
        """
        Remove a specified key from an HDF5 file.
    
        Parameters:
        h5file (str): The path to the HDF5 file.
        key (str): The key to be removed from the HDF5 file.
        """
        with pd.HDFStore(self.db, mode='a') as store:
            if 'recaptures' in store:
                store.remove('recaptures')
                print(f"Recaptures has been removed from the HDF5 file.")
                    
    def undo_overlap(self):
        """
        Remove a specified key from an HDF5 file.
    
        Parameters:
        h5file (str): The path to the HDF5 file.
        key (str): The key to be removed from the HDF5 file.
        """
        with pd.HDFStore(self.db, mode='a') as store:
            if 'overlapping' in store:
                store.remove('overlapping')
                print(f"Overlapping has been removed from the HDF5 file.")
                
    def new_db_version(self, output_h5):
        """
        Create a new version of the working HDF5 database.

        This function creates a copy of the existing working database, allowing you to 
        backtrack or branch your analysis. If there are keys that are in error or conflict 
        with the current understanding of the system, this function helps you remove them 
        from the new version of the database.

        Parameters:
        output_h5 (str): The file name for the new HDF5 file.
        """

        # Copy the HDF5 file
        shutil.copyfile(self.db, output_h5)

        # Open the copied HDF5 file
        with h5py.File(output_h5, 'r+') as hdf:
            # List all keys in the file
            keys = list(hdf.keys())
            print("Keys in HDF5 file:", keys)

            # Ask the user to input the keys they want to modify
            selected_keys = input("Enter the keys you want to modify, separated by commas: ").split(',')

            # Clean up the input (remove whitespace)
            selected_keys = [key.strip() for key in selected_keys]

            for key in selected_keys:
                if key in hdf:
                    print(f"Processing key: '{key}'...")
                    
                    # If it's a group, recursively delete all datasets (subkeys)
                    if isinstance(hdf[key], h5py.Group):
                        print(f"Key '{key}' is a group. Deleting all subkeys...")
                        for subkey in list(hdf[key].keys()):
                            print(f"Removing subkey: '{key}/{subkey}'")
                            del hdf[key][subkey]
                        print(f"All subkeys under group '{key}' have been deleted.")
                    else:
                        # It's a dataset, clear the data in the DataFrame
                        print(f"Clearing data for dataset key: '{key}'")
                        df = pd.read_hdf(output_h5, key)
                        df.drop(df.index, inplace=True)
                        df.to_hdf(output_h5, key, mode='a', format='table', data_columns=True)
                        print(f"Data cleared for key: '{key}'")
                else:
                    print(f"Key '{key}' not found in HDF5 file.")

        # Update the project's database to the new copied database
        self.db = output_h5

            
                    
