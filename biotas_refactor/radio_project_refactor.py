# -*- coding: utf-8 -*-

'''
Module contains all of the functions to create a radio telemetry project.'''

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import h5py
import datetime
import biotas_refactor.naive_bayes as naive_bayes
import biotas_refactor.parsers as  parsers
import biotas_refactor.predictors as predictors
import matplotlib.pyplot as plt
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
        self.db = os.path.join(project_dir,'%s.h5'%(db_name))
        self.tags = tag_data
        self.tags.set_index('freq_code', drop = False, inplace = True)
        self.study_tags = self.tags[self.tags.tag_type == 'STUDY'].freq_code.values
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
            
        if 'bouts' not in hdf5:
            hdf5.create_group('bouts')
            
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
                parsers.orion_import(f_dir,db_dir,rec_id, self.tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type == 'vr2':
                parsers.vr2_import(f_dir,db_dir,rec_id)
            else:
                print ("There currently is not an import routine created for this receiver type.  Please try again")
            
            print ("File %s imported"%(f))
        
        print ("Raw Telemetry Data Import Completed")
    
    def get_fish(self, rec_id, reclass_iter = 0):
        
        if reclass_iter == 0:
            dat = pd.read_hdf(self.db,'raw_data')
            dat = dat[dat.rec_id == rec_id]      
        
        else:
            dat = pd.read_hdf(self.db,'classified')
            dat = dat[(dat.rec_id == rec_id) * (dat.iter ==  reclass_iter - 1)]     
            
        tags_no_idx = self.tags.reset_index(drop = True)
        dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
        #TODO - make this smart so case doesn't matter...
        dat = dat[(dat.tag_type != 'BEACON') & (dat.tag_type != 'TEST')]
    
        return dat.freq_code.unique()
    
    def train(self, freq_code, rec_id):
        '''A class object for a training dataframe and related data objects.
    
        This class object creates a training dataframe for animal i at site j.
    
        when class is initialized, we will extract information for this animal (i)
        at reciever (site) from the project database (projectDB).
            '''
        # pull raw data 
        train_dat = pd.read_hdf(self.db,'raw_data')
        train_dat = train_dat[(train_dat.freq_code == freq_code) & 
                     (train_dat.rec_id == rec_id)]
        #train_dat.drop(columns = 'index', inplace = True)
    

        # do some data management when importing training dataframe
        train_dat['time_stamp'] = pd.to_datetime(train_dat.time_stamp)
        
        #train_dat.sort_values(by = 'Epoch', inplace = True)
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
        # if self.tags.at[freq_code,'mort_rate'] == np.nan or self.tags.at[freq_code,'mort_rate'] == 0:
        #     mort_rate = 9999.0
        # else:
        #     mort_rate = self.tags.at[freq_code,'mort_rate']
        
        mort_rate = 8888.
        # calculate predictors
        train_dat['detection'] = np.repeat(plausible,len(train_dat))
        train_dat['lag'] = train_dat.Epoch.diff().abs()
        train_dat['lag_diff'] = train_dat.lag.diff()
        det_hist_string, hit_ratio, cons_det, max_count = predictors.detection_history(train_dat.Epoch.values,
                                                                           pulse_rate,
                                                                           self.det_count,
                                                                           2,
                                                                           train_dat.scan_time.values,
                                                                           train_dat.channels.values)
        train_dat['det_hist'] = det_hist_string
        train_dat['hit_ratio'] = hit_ratio
        train_dat['cons_det'] = cons_det
        train_dat['cons_length'] = max_count
        # train_dat['series_hit'] = predictors.series_hit(train_dat.lag.values,
        #                                           pulse_rate,
        #                                           mort_rate,
        #                                           'A')
        
        train_dat.fillna(value=9999999, inplace=True)      
        
        # make sure data types are correct - these next steps are critical
        train_dat = train_dat.astype({'power': 'float32', 
                                      'time_stamp': 'datetime64',
                                      'Epoch': 'float32',
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

        # append to hdf5
        with pd.HDFStore(self.db, mode='a') as store:
            store.append(key = 'trained',
                         value = train_dat, 
                         format = 'table', 
                         index = False, 
                         append = True, 
                         chunksize = 1000000)
            

        
        print ('Fish %s trained at receiver %s, plausibiity: %s'%(freq_code, rec_id, plausible))

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
                print ("%6s|   %8s   |   %8s   |"%(i,0,trues.det_clas_count.values[0]))

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

        
    def create_training_data(self, rec_type, reclass_iter = None, rec_list = None):
        '''Function creates training dataset for current round of classification -
        if we only do this once, this time suck goes away'''
    
        #get training data
        '''
        Reclassification code contributed by T Castro-Santos
        '''
        # get training data
        train_dat = pd.read_hdf(self.db,'trained')
        
        # first restrict it to the receiver type - we can't have orion's diagnose srx800's now can we?
        train_dat = train_dat[train_dat.rec_type == rec_type]
        
        # then if we are further restricting to a subset of that receiver type
        if rec_list != None:
            train_dat = train_dat[train_dat['rec_id'].isin(rec_list)] 
   
            # if this is not the first classification - we need known falses from training and assumed true from last classification
            if reclass_iter != None:
                last_class = reclass_iter - 1
                
                class_dat = pd.read_hdf(self.db, 'classified')
                class_dat = class_dat[class_dat['rec_id'].isin(rec_list)]
                class_dat = class_dat[class_dat.iter == last_class]
                
                columns = ['test', 'freq_code','power','noise_ratio','lag', 'lag_diff', 
                           'cons_length','cons_det','det_hist','hit_ratio',
                           'series_hit','rec_type','Epoch']
                
                class_dat = class_dat[columns]
                
                class_dat.rename(columns = {'test':'detection'},
                                 inplace = True)
        
                train_dat = train_dat[train_dat.detection==0]
                class_dat = class_dat[class_dat.detection==1]
    
                #Next we append the classdf to the traindf
                train_dat = train_dat.append(class_dat)
    
        return train_dat
    
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
            class_dat = pd.read_hdf(self.db, 'raw_data')
            class_dat = class_dat[(class_dat.freq_code == freq_code) & 
                                  (class_dat.rec_id == rec_id)]
            
            columns = ['freq_code','Epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        else:
            class_dat = pd.read_hdf(self.db, 'classified')
            class_dat = class_dat[(class_dat.freq_code == freq_code) &
                                  (class_dat.rec_id == rec_id) & 
                                  (class_dat.iter == (reclass_iter - 1)) &
                                  (class_dat.test == 1)]

            columns = ['freq_code','Epoch','rec_id','time_stamp','power','noise_ratio','scan_time','channels','rec_type']
            class_dat = class_dat[columns]

        if len(class_dat) > 0:
            # do some data management when importing training dataframe
            class_dat['time_stamp'] = pd.to_datetime(class_dat['time_stamp'])
            class_dat.sort_values(by = 'Epoch', inplace = True)
            #class_dat = class_dat.drop_duplicates(subset = 'time_stamp')
            
            # calculate predictors
            class_dat['lag'] = class_dat.Epoch.diff().abs()
            class_dat['lag_diff'] = class_dat.lag.diff()
            class_dat.fillna(value = 99999999, inplace = True)
            det_hist_string, det_hist, cons_det, max_count = predictors.detection_history(class_dat.Epoch.values,
                                                                               pulse_rate,
                                                                               self.det_count,
                                                                               2,
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
            
            # categorize predictors
            hit_ratio, power, lag, con_len, noise = naive_bayes.bin_predictors(class_dat.hit_ratio,
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
                                          'Epoch': 'float32',
                                          'rec_id': 'object',
                                          'time_stamp': 'datetime64',
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
                             append = True, 
                             chunksize = 1000000)
            
            # export
            class_dat.to_csv(os.path.join(self.output_dir,'freq_code_%s_rec_%s_class_%s.csv'%(freq_code, rec_id, reclass_iter)))
            
            #print ('Fish %s at receiver %s classified'%(freq_code,rec_id))
            # next step looks at results
        
        # else:
        #     print ('there are no recoreds to classify for fish %s at %s'%(freq_code, rec_id))
        
    def classification_summary(self,rec_id,reclass_iter = None): 
        '''if this is not the initial classification we need the trues from the last 
        last classification and falses from the first'''
                
        if reclass_iter == None:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified')
            classified_dat = classified_dat[classified_dat.iter == 1]
        else:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified')
            classified_dat = classified_dat[classified_dat.iter == reclass_iter]  
            
        classified_dat = classified_dat[classified_dat.rec_id == rec_id]
        
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

            # print ("Compiling Figures")

            # # plot the log likelihood ratio
            # classified_dat['log_likelihood_ratio'] = classified_dat.likelihood_T / classified_dat.likelihood_F
            # minLogRatio = classified_dat.log_likelihood_ratio.min()//1 * 1
            # maxLogRatio = classified_dat.log_likelihood_ratio.max()//1 * 1
            # ratio_bins =np.arange(minLogRatio,maxLogRatio+1,2)
            
            # # hit ratio bins
            # hit_ratio_bins =np.linspace(0,1.0,11)

            # # plot signal power histograms by detection class
            # min_power = classified_dat.power.min()//5 * 5
            # max_power = classified_dat.power.max()//5 * 5
            # power_bins =np.arange(min_power,max_power+20,10)

            # # Lag Back Differences - how steady are detection lags?
            # lag_bins =np.arange(-100,110,20)

            # # Consecutive Record Length
            # con_length_bins =np.arange(1,12,1)

            # # Noise Ratio
            # noise_bins =np.arange(0,1.1,0.1)
    
            # # plot the log of the posterior ratio
            # classified_dat['log_post_ratio'] = np.log(classified_dat.posterior_T/classified_dat.posterior_F)
            # minPostRatio = classified_dat.log_post_ratio.min()
            # maxPostRatio = classified_dat.log_post_ratio.max()
            # post_ratio_bins = np.linspace(minPostRatio,maxPostRatio,10)

            
            # trues = classified_dat[classified_dat.test == 1]
            # falses = classified_dat[classified_dat.test == 0]

            # # make lattice plot for pubs
            # figSize = (8,6)
            # plt.figure()
            # fig, axs = plt.subplots(3,4,tight_layout = True,figsize = figSize)
            # # hit ratio
            # axs[0,1].hist(trues.hit_ratio.values,
            #               hit_ratio_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[0,0].hist(falses.hit_ratio.values,
            #               hit_ratio_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[0,0].set_xlabel('Hit Ratio')
            # axs[0,1].set_title('Valid')
            # axs[0,1].set_xlabel('Hit Ratio')
            # axs[0,0].set_title('False Positive')
            # axs[0,0].set_title('A',loc = 'left')

            # # consecutive record length
            # axs[0,3].hist(trues.cons_length.values,
            #               con_length_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[0,2].hist(falses.cons_length.values,
            #               con_length_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[0,2].set_xlabel('Consecutive Hit Length')
            # axs[0,3].set_title('Valid')
            # axs[0,3].set_xlabel('Consecutive Hit Length')
            # axs[0,2].set_title('False Positive')
            # axs[0,2].set_title('B',loc = 'left')

            # # power
            # axs[1,1].hist(trues.power.values,
            #               power_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[1,0].hist(falses.power.values,
            #               power_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[1,0].set_xlabel('Signal Power')
            # axs[1,1].set_xlabel('Signal Power')
            # axs[1,0].set_ylabel('Probability Density')
            # axs[1,0].set_title('C',loc = 'left')

            # # noise ratio
            # axs[1,3].hist(trues.noise_ratio.values,
            #               noise_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[1,2].hist(falses.noise_ratio.values,
            #               noise_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[1,2].set_xlabel('Noise Ratio')
            # axs[1,3].set_xlabel('Noise Ratio')
            # axs[1,2].set_title('D',loc = 'left')

            # # lag diff
            # axs[2,1].hist(trues.lag_diff.values,
            #               lag_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[2,0].hist(falses.lag_diff.values,
            #               lag_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[2,0].set_xlabel('Lag Differences')
            # axs[2,1].set_xlabel('Lag Differences')
            # axs[2,0].set_title('E',loc = 'left')

            # # log posterior ratio
            # axs[2,3].hist(trues.log_likelihood_ratio.values,
            #               ratio_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[2,2].hist(falses.log_likelihood_ratio.values,
            #               ratio_bins,
            #               density = True,
            #               color = 'grey',
            #               edgecolor='black',
            #               linewidth=1.2)
            # axs[2,2].set_xlabel('Log Likelihood Ratio')
            # axs[2,3].set_xlabel('Log Likelihood Ratio')
            # axs[2,2].set_title('F',loc = 'left')

            # if self.figureWS != None:
            #     if self.rec_list != None:
            #        plt.savefig(os.path.join(self.figureWS,"%s_lattice_class.png"%(self.recType)),
            #                    bbox_inches = 'tight',
            #                    dpi = 900)
            #     else:
            #        plt.savefig(os.path.join(self.figureWS,"%s_%s_lattice_class.png"%(self.recType,self.site)),
            #                    bbox_inches = 'tight',
            #                    dpi = 900)
        else:
           print("There were insufficient data to quantify summary statistics")
           print("All remaining were classified as %s suggesting there is no more improvement in the model"%(det_class_count.index[0]))
                         