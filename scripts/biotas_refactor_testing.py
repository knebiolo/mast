# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas")
from biotas_refactor.radio_project_refactor import radio_project
import pandas as pd

#%% set up project
project_dir = r"C:\Users\knebiolo\Desktop\BIOTAS_Refactor"
db_name = 'refactor_experiments'
detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# create a project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

#%%  import data
rec_id = 'R001'
rec_type = 'srx1200'
training_dir = r"C:\Users\knebiolo\Desktop\BIOTAS_Refactor\Data\Training_Files"
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 2.5
channels = 2
antenna_to_rec_dict = {'Antenna 1':'R001'}

project.telem_data_import(rec_id,
                          rec_type,
                          training_dir,
                          db_dir,
                          scan_time,
                          channels,
                          antenna_to_rec_dict)

#%%  train data
# set parameters and get a list of fish to iterate over
rec_id = 'R001'
rec_type = 'srx1200'
fishes = project.get_fish(rec_id = rec_id)

# iterate over fish and train
for fish in fishes:
    project.train(fish, rec_id)

# generate summary statistics
project.training_summary(rec_type, site = [rec_id])

#%% classify data
# set parameters and get a list of fish to iterate over
rec_id = 'R001'
rec_type = 'srx1200'
class_iter = None
fishes = project.get_fish(rec_id = rec_id)
threshold_ratio = 1.0 # 1.0 = MAP Hypothesis

# then generate training data for the classifier
training_data = project.create_training_data(rec_type,class_iter,[rec_id])

# next, create your A-La Carte Likelihood function
# fields = ['cons_length','cons_det','hit_ratio','noise_ratio','series_hit','power','lag_diff']
fields = ['noise_ratio','power','lag_diff']

# iterate over fish and classify
for fish in fishes:
    project.classify(fish,rec_id,fields,training_data,class_iter,threshold_ratio)

# generate summary statistics
project.classification_summary(rec_id, class_iter)

