# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\mast")
from mast.radio_project import radio_project
import mast
import pandas as pd

#%% set up project
project_dir = r"D:\York Haven"
db_name = 'york_haven'
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
rec_id = 'R004'
rec_type = 'srx800'
training_dir = os.path.join(project_dir,'Data','Training_Files')
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 5.0         
channels = 2
antenna_to_rec_dict = {'A0':rec_id}

project.telem_data_import(rec_id,
                          rec_type,
                          training_dir,
                          db_dir,
                          scan_time,
                          channels,
                          antenna_to_rec_dict)

# project.undo_import(rec_id)

#%%  train data
# set parameters and get a list of fish to iterate over
rec_id = 'R004'
rec_type = 'srx800'
fishes = project.get_fish(rec_id = rec_id)

# iterate over fish and train
for fish in fishes:
    project.train(fish, rec_id)

# generate summary statistics
project.training_summary(rec_type, site = [rec_id])

# undo training
# project.undo_training(rec_id)

#%% classify data
#set parameters and get a list of fish to iterate over
rec_id = 'T7'
rec_type = 'orion'
class_iter = None # start with none - if we need more classifications then 2
fishes = project.get_fish(rec_id = rec_id)
threshold_ratio = 1.0 # 1.0 = MAP Hypothesis

# then generate training data for the classifier
training_data = project.create_training_data(rec_type,class_iter)#,[rec_id])

# next, create your A-La Carte Likelihood function
# fields = ['cons_length','cons_length','hit_ratio','noise_ratio','series_hit','power','lag_diff']
fields = ['hit_ratio','cons_length','noise_ratio','power','lag_diff']

# iterate over fish and classify
for fish in fishes:
    project.classify(fish,rec_id,fields,training_data,class_iter,threshold_ratio)

# generate summary statistics
project.classification_summary(rec_id, class_iter)

#project.undo_classification(rec_id, class_iter = class_iter)

#%% cross validate


#%% calculate bouts
# get nodes
#nodes = project.nodes.node
nodes = ['S2']

# for each node determine the bout threshold and enumerate presence
for node in nodes:
    # determine bout thresholds
    bout = mast.bout(project, node, 2, 21600)
    
    # Find the knot by minimizing the objective function as before, now also passing spline_der
    threshold = bout.fit_processes()
    
    bout.presence(threshold)
    
    
#%% reduce overlap
# create edges showing parent:child relationships for nodes in network
edges = [('T5','S2')]
nodes = ['T5','S2']
    
# create an overlap object and apply nested doll algorithm
doll = mast.overlap_reduction(nodes, edges, project)
doll.nested_doll()

#%% create a recaptures table
project.make_recaptures_table()

#%% create models
    
    