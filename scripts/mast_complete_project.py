# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\mast\pymast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"C:\Users\knebiolo\Desktop\York Haven"
db_name = 'york_haven_2'
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
rec_id = 'R003'
rec_type = 'srx1200'
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
                          antenna_to_rec_dict,
                          True)

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

# %% classify data

# Set initial parameters
rec_id = 'R020'
rec_type = 'orion'
threshold_ratio = 1.0  # 1.0 = MAP Hypothesis
likelihood = ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff'] # a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

project.reclassify(project, rec_id, rec_type, threshold_ratio,likelihood)

# undo classification 
# project.undo_classification(rec_id, class_iter = class_iter)

#%% cross validate


#%% calculate bouts
# get nodes
node = 'R020'

# create a bout object
bout = pymast.bout(project, node, 2, 21600)
    
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
doll = pymast.overlap_reduction(nodes, edges, project)
doll.nested_doll()

# project.undo_overlap()
#%% create a recaptures table
project.make_recaptures_table()

# project.undo_recaptures()
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

#%% create a Cormack-Jolly-Seber Mark Recapture model
# what is the output directory?
output_ws = os.path.join(project_dir,'Output')
model_name = "york_haven"

# what is the Node to State relationship - use Python dictionary
receiver_to_recap = {'R001':'R01','R002':'R01',
                     'R003':'R02','R004':'R04','R005':'R04','R006':'R02',
                     'R007':'R02','R008':'R02','R009':'R02','R010':'R02',
                     'R011':'R03','R012':'R02','R013':'R02','R014':'R02',
                     'R015':'R02','R016':'R02','R017':'R02',#'R018':'R02',
                     'R019':'R03','R020':'R03',}

# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = formatter.cjs_data_prep(receiver_to_recap, project, initial_recap_release = False)
print ("Step 1 Completed, Data Class Finished")

# Step 2, Create input file for MARK
cjs.input_file(model_name,output_ws)
cjs.inp.to_csv(os.path.join(output_ws,model_name + '.csv'), index = False)

print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

