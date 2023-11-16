# -*- coding: utf-8 -*-

'''
Module contains all of the functions to create a radio telemetry project.'''

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import h5py
import datetime
import naive_bayes
import predictors
import parsers
from matplotlib import rcParams
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'



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
        self.db = os.path.join(project_dir,'Data',db_name,'.h5')
        self.tags = tag_data
        self.tags.set_index('freq_code', drop = False, inplace = True)
        self.study_tags = self.tags[self.tag_type == 'study']
        self.receivers = receiver_data
        self.receivers.set_index('rec_id', drop = False, inplace = True)
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
        
        # create a project database and write initial arrays to HDF
        self.hdf5 = h5py.File(self.db, 'a')
        self.initialize_hdf5()    
        
    def initialize_hdf5(self):
        '''Initialize an HDF5 database for a simulation'''
        # Create groups for organization (optional)
        if 'project_setup' not in self.hdf5:
            self.hdf5.create_group("project_setup")
            self.tags.to_hdf(self.db, key='/project_setup/tags', mode='a')
            self.receivers.to_hdf(self.db, key='/project_setup/receivers', mode='a')
            self.nodes.to_hdf(self.db, key='/project_setup/nodes', mode='a')

        if 'raw_data' not in self.hdf5:
            self.hdf5.create_group("raw_data")  
            
        if 'trained' not in self.hdf5:
            self.hdf5.create_group("trained")
            
        if 'classified' not in self.hdf5:
            self.hdf5.create_group("classified")
            
        if 'presence' not in self.hdf5:
            self.hdf5.create_group("presence")
            
        if 'bouts' not in self.hdf5:
            self.hdf5.create_group('bouts')
            
        if 'overlapping' not in self.hdf5:
            self.hdf5.create_group('overlapping')
            
        if 'recaptures' not in self.hdf5:
            self.hdf5.create_group('recaptures')
            
    def telem_data_import(self,
                          rec_id,
                          rec_type,
                          file_dir,
                          db_dir,
                          scan_time = 1, 
                          channels = 1, 
                          ant_to_rec_dict = None):
        # list raw data files
        tFiles = os.listdir(file_dir)
        
        # for every file call the correct text parser and import
        for f in tFiles:
            print ("start importing file %s"%(f))
            # get the complete file directory
            f_dir = os.path.join(file_dir,f)
            
            if rec_type == 'srx600' :
                parsers.srx600(f_dir, db_dir, rec_id, self.tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'srx800':
                parsers.srx800(f_dir, db_dir, rec_id, self.tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'srx1200':
                parsers.srx1200(f_dir, db_dir, rec_id, self.tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'orion':
                parsers.orion_mport(f_dir,db_dir,rec_id, scan_time, channels, ant_to_rec_dict)
            
            elif rec_type == 'vr2':
                parsers.vr2_import(f_dir,db_dir,rec_id)
            else:
                print ("There currently is not an import routine created for this receiver type.  Please try again")
            
            print ("File %s imported"%(f))
        
        print ("Raw Telemetry Data Import Completed")
        
    def train(self, freq_code, rec_id):
        '''A class object for a training dataframe and related data objects.
    
        This class object creates a training dataframe for animal i at site j.
    
        when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).
            '''
        # pull raw data 
        train_dat = pd.read_hdf(self.db, 
                                     'raw_data', 
                                     where=[f'freq_code == {freq_code}',
                                            f'rec_id == {rec_id}'])        

        # do some data management when importing training dataframe
        train_dat['time_stamp'] = pd.to_datetime(train_dat['time_stamp'])
        
        train_dat.sort_values(by = 'Epoch', inplace = True)
        train_dat.set_index('Epoch', drop = False, inplace = True)
        train_dat = train_dat.drop_duplicates(subset = 'time_stamp')
        
        # set some object variables

        rec_type = self.receivers.at[rec_id,'rec_type']

        # for training data, we know the tag's detection class ahead of time,
        # if the tag is in the study tag list, it is a known detection class, if
        # it is a beacon tag, it is definite, if it is a study tag, it's plausible
        if freq_code in self.study_tags:
            plausible = 1
        else:
            plausible = 0
        # get rate
        if freq_code in self.tags.freq_code.values:
            pulse_rate = self.tags.at[freq_code,'pulse_rate']
        else:
            pulse_rate = 3.0
        if self.tags.at[freq_code,'mort_rate'] == np.nan or self.tags.at[freq_code,'mort_rate'] == 0:
            mort_rate = 9999.0
        else:
            mort_rate = self.tags.at[freq_code,'mort_rate']
            
        # calculate predictors
        train_dat['detection'] = np.repeat(plausible,len(train_dat))
        train_dat['lag'] = train_dat.Epoch.diff().abs()
        train_dat['lag_diff'] = train_dat.lag.diff()
        det_hist_string, det_hist, cons_det, max_count = predictors.detection_history(train_dat.Epoch.values,
                                                                           pulse_rate,
                                                                           self.det_count,
                                                                           train_dat.channels.unique.values.shape[0],
                                                                           train_dat.scan_time.values,
                                                                           train_dat.channels)
        train_dat['det_hist'] = det_hist_string
        train_dat['hit_ratio'] = det_hist
        train_dat['cons_det'] = cons_det
        train_dat['cons_length'] = max_count
        train_dat['series_hit'] = predictors.series_hit(train_dat.lag.values,
                                                  pulse_rate,
                                                  mort_rate,
                                                  'A')
        train_dat.set_index('time_stamp', inplace = True, drop = True)
        train_dat['Seconds'] = train_dat.index.hour * 3600 + train_dat.index.minute * 60 + train_dat.index.second
        train_dat.reset_index(inplace = True, drop = False)
        
        # append to hdf5
        train_dat.to_hdf(self.hdf5,
                         key = 'trained',
                         mode = 'a', 
                         format = 'table')
        
    def create_training_data(self, site, reclass_iter = None, rec_list = None):
        '''Function creates training dataset for current round of classification -
        if we only do this once, this time suck goes away'''
    
        #get training data
        '''
        Reclassification code contributed by T Castro-Santos
        '''
        rec_type = self.receivers.at[site,'rec_type']
        
        # if this is the first classification - we need data from the training array
        if reclass_iter == None:
            # if there are not a list of surfaces
            if rec_list == None:
                train_dat = pd.read_hdf(self.db, 
                                             'trained', 
                                             where = [f'rec_type == {rec_type}']) 
            # if there is a list of receivers
            else:
                train_dat = pd.DataFrame()
                where = []
                for rec_id in rec_list:
                    where.append(f'rec_id == {rec_id}')
                        
                train_dat = pd.read_hdf(self.db, 
                                             'trained', 
                                             where=where)
        
        # if this is not the first, we need known falses from training and assumed true from last classification
        else:
            last_class = reclass_iter - 1
            if rec_list == None:
                train_dat = pd.read_hdf(self.db, 'trained', where = [f'rec_type == {rec_type}']) 
                
            else:
                where = []
                for i in rec_list:
                    where.append(f'rec_id == {rec_id}')
                        
                train_dat = pd.read_hdf(self.db, 'trained', where = where)  
            
            if rec_list == None:
                train_dat = pd.read_hdf(self.db, 'trained', where = [f'rec_type == {rec_type}']) 

            else:
                where = []
                for i in rec_list:
                    where.append(f'rec_id == {rec_id}')
                        
                train_dat = pd.read_hdf(self.db, 'trained',  where = where)  
                train_dat = train_dat[train_dat.Detection == 0]

            class_dat = pd.read_hdf(self.db, 'classified/rec_%s_iter_%s'%(rec_id, last_class))
            columns = ['test', 'freq_code','power','noise_ratio','lag', 'lagDiff', 
                       'con_length_A','cons_det_A','det_hist_A','hit_ratio_A',
                       'series_hit_A','RowSeconds','rec_type','Epoch']
            
            class_dat = class_dat[columns]
            
            class_dat.rename(columns = {'con_length_A':'con_length',
                                      'cons_det_A':'cons_det',
                                      'det_hist_A':'detHist',
                                      'hitRatio_A':'hit_ratio',
                                      'series_hit_A':'series_hit',
                                      'test':'Detection',
                                      'RowSeconds':'seconds'},
                             inplace = True)
    
            train_dat = train_dat[train_dat.Detection==0]
            class_dat = class_dat[class_dat.Detection==1]

            #Next we append the classdf to the traindf
            train_dat = train_dat.append(class_dat)

        return train_dat
    
    def classify(self,
                 freq_code,
                 rec_id,
                 fields,
                 training_data,
                 reclass_iter = None):
        # get rates
        pulse_rate = self.tags.at[freq_code,'pulse_rate']
        mort_rate = self.tags.at[freq_code,'mort_rate']
        
        # get data
        if reclass_iter == None:
            class_dat = pd.read_hdf(self.db, 'raw', where = [f'freq_code == {freq_code}',
                                                             f'rec_id == {rec_id}'])
            columns = ['freq_code','Epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        else:
            class_dat = pd.read_hdf(self.db, 
                                    'classified/rec_%s_iter_%s'%(rec_id, reclass_iter - 1),
                                    where = [f'test == {1}'])

            columns = ['freq_code','Epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        # do some data management when importing training dataframe
        class_dat['time_stamp'] = pd.to_datetime(class_dat['time_stamp'])
        class_dat['RowSeconds'] = class_dat['Epoch']
        class_dat.sort_values(by = 'Epoch', inplace = True)
        class_dat.set_index('Epoch', drop = False, inplace = True)
        class_dat = class_dat.drop_duplicates(subset = 'time_stamp')
        
        # calculate predictors
        class_dat['lag'] = class_dat.Epoch.diff().abs()
        class_dat['lag_diff'] = class_dat.lag.diff()
        det_hist_string, det_hist, cons_det, max_count = predictors.detection_history(class_dat.Epoch.values,
                                                                           pulse_rate,
                                                                           self.det_count,
                                                                           class_dat.channels.unique.values.shape[0],
                                                                           class_dat.scan_time.values,
                                                                           class_dat.channels)
        class_dat['det_hist'] = det_hist_string
        class_dat['hit_ratio'] = det_hist
        class_dat['cons_det'] = cons_det
        class_dat['cons_length'] = max_count
        class_dat['series_hit'] = predictors.series_hit(class_dat.lag.values,
                                                  pulse_rate,
                                                  mort_rate,
                                                  'A')
        
        class_dat.set_index('time_stamp', inplace = True, drop = True)
        class_dat['Seconds'] = class_dat.index.hour * 3600 + class_dat.index.minute * 60 + class_dat.index.second
        class_dat.reset_index(inplace = True, drop = False)
        
        # calculate prior - from training data
        prior = naive_bayes.calculate_priors(training_data.Detection.values)
        
        # calculate likelihoods - also from training data
        likelihoods = {}
        labeled_array = training_data.Detection.values
        # loop over assumptions and fields
        for assumption in [True, False]:
            for field in fields:
                # get observation array 
                observation_array = training_data[field].values
                
                # bin if neccesary
                if field == 'lagDiff' or field == 'power' or field == 'noise_ratio':
                    observation_array = observation_array//10
                
                # calculate likelihood
                likelihood = naive_bayes.calculate_likelihood(observation_array,
                                                       labeled_array,
                                                       assumption)
                
                # add it to the dictionary
                likelihoods[assumption] = {field:likelihood}
                
        # calculate the evidence
        observation_arrays = []
        # for each field, calculate the probability of observing each bin
        for field in fields:
            if field == 'lagDiff' or field == 'power' or field == 'noise_ratio':
                observation_arrays.append(training_data[field].values // 10)   
            else:
                observation_arrays.append(training_data[field].values)   

        evidence = naive_bayes.calculate_evidnce(observation_arrays)
       
        # calculate posteriors
        posterior_T = naive_bayes.calculate_posteriors({True:np.repeat(prior[0],evidence.shape),
                                            False:np.repeat(prior[1],evidence.shape)},
                                           evidence,
                                           likelihoods,
                                           True)
        posterior_F = naive_bayes.calculate_posteriors({True:np.repeat(prior[0],evidence.shape),
                                            False:np.repeat(prior[1],evidence.shape)},
                                           evidence,
                                           likelihoods,
                                           False)

        # Classify
        classification = naive_bayes.classify_with_threshold(posterior_T, 
                                                 posterior_F,
                                                 1.0)
        
        # now add these arrays to the dataframe and export to hdf5
        class_dat['class'] = classification
        get_class_iter = lambda x: 1 if reclass_iter is None else reclass_iter
        self.hdf5(class_dat,
                  'classified/rec_%s_class_%s'%(rec_id, get_class_iter(reclass_iter)),
                  mode = 'a')
        class_dat.to_csv(os.path.join(self.output_dir,'rec_%s_class_%s.csv'%(rec_id, get_class_iter(reclass_iter))))
        
                         