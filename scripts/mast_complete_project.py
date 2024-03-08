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
from mast import formatter as formatter
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
#TODO - remove these directory arguments - the project is smart
training_dir = os.path.join(project_dir,'Data','Training_Files')
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 1.         
channels = 1
antenna_to_rec_dict = {'A0':rec_id}

project.telem_data_import(rec_id,
                          rec_type,
                          training_dir,
                          db_dir,
                          scan_time,
                          channels,
                          antenna_to_rec_dict)

# undo import
# project.undo_import(rec_id)

#%%  train data
# set parameters and get a list of fish to iterate over
rec_id = 'R020'
rec_type = 'orion'
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
rec_id = 'R020'
rec_type = 'orion'
class_iter = 2 # start with none - if we need more classifications then 2
fishes = project.get_fish(rec_id = rec_id, 
                          train = False, 
                          reclass_iter = class_iter)
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

# undo classification 
# project.undo_classification(rec_id, class_iter = class_iter)

#%% cross validate


#%% calculate bouts
# get nodes
node = 'R020'

# create a bout object
bout = mast.bout(project, node, 2, 21600)
    
# Find the threshold
threshold = bout.fit_processes()
    
# calculate presences - or pass float
bout.presence(threshold)

# undo bouts
# project.undo_bouts(node)
    
#%% reduce overlap
# create edges showing parent:child relationships for nodes in network
edges = [('R010','R013'),('R010','R014'),('R010','R015'),('R010','R016'),('R010','R017'),('R010','R018'),
          ('R019','R013'),('R019','R014'),('R019','R015'),('R019','R016'),('R019','R017'),('R019','R018'),
          ('R020','R013'),('R020','R014'),('R020','R015'),('R020','R016'),('R020','R017'),('R020','R018')]

nodes = ['R010','R019','R020','R013','R014','R015','R016','R017','R018']
    
# create an overlap object and apply nested doll algorithm
doll = mast.overlap_reduction(nodes, edges, project)
doll.nested_doll()

#%% create a recaptures table
project.make_recaptures_table()

#%% create models using a Time to Event Framework
    
# what is the Node to State relationship - use Python dictionary
node_to_state = {'R001':1,'R002':1,                   # upstream
                 'R012':2,                            # forebay
                 'R013':3,'R015':3,'R016':3,'R017':3, # powerhouse
                 'R018':4,                            # sluice
                 'R003':5,                            # east channel up
                 'R007':6,                            # east channel down
                 'R008':7,                            # east channel dam
                 'R009':8,                            # NLF
                 'R010':9,'R019':19,                  # tailrace
                 'R011':10,                           # downstream
                 'R004':11,'R005':11}                 # downstream 2

# Step 1, create time to event data class 
tte = formatter.time_to_event(node_to_state,
                              project,
                              initial_state_release = True)

# Step 2, format data - with covariates
# tte.data_prep(project,
#               time_dependent_covariates = True,
#               adjacency_filter = [('R010','R013'),('R010','R014'),('R010','R015'),('R010','R016'),('R010','R017'),('R010','R018'),
#                                   ('R019','R013'),('R019','R014'),('R019','R015'),('R019','R016'),('R019','R017'),('R019','R018'),
#                                   ('R020','R013'),('R020','R014'),('R020','R015'),('R020','R016'),('R020','R017'),('R020','R018')])
# Step 3, format data - without covariates
tte.data_prep(project,
              adjacency_filter = [('R010','R013'),('R010','R014'),('R010','R015'),('R010','R016'),('R010','R017'),('R010','R018'),
                                  ('R019','R013'),('R019','R014'),('R019','R015'),('R019','R016'),('R019','R017'),('R019','R018'),
                                  ('R020','R013'),('R020','R014'),('R020','R015'),('R020','R016'),('R020','R017'),('R020','R018')])
# Step 4, generate a summary
tte.summary()
