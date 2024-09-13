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
project_dir = r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\PYMAST\Nuyakuk"
db_name = 'nuyakuk_kpn'
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

# create a new version of an existing project 
# project.new_db_version(os.path.join(project_dir,'Nuyakuk_v2.h5'))
# ,presence,classified,trained

# #%%  import data
# rec_id = 'R02'
# rec_type = 'srx1200'
# #TODO - remove these directory arguments - the project is smart
# training_dir = os.path.join(project_dir,'Data','Training_Files')
# db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
# scan_time = 1.         
# channels = 1
# antenna_to_rec_dict = {'Antenna 1':rec_id}

# project.telem_data_import(rec_id,
#                           rec_type,
#                           training_dir,
#                           db_dir,
#                           scan_time,
#                           channels,
#                           antenna_to_rec_dict,
#                           True)

# # undo import
# # project.undo_import(rec_id)

# #%%  train data
# # set parameters and get a list of fish to iterate over
# rec_id = 'R02'
# rec_type = 'srx1200'
# fishes = project.get_fish(rec_id = rec_id)

# # iterate over fish and train
# for fish in fishes:
#     project.train(fish, rec_id)

# # generate summary statistics
# project.training_summary(rec_type, site = [rec_id])

# # undo training
# # project.undo_training(rec_id)

# # %% classify data

# # Set initial parameters
# rec_id = 'R03'
# rec_type = 'srx1200'
# threshold_ratio = 1  # 1.0 = MAP Hypothesis
# likelihood = ['hit_ratio','cons_length', 'noise_ratio', 'power', 'lag_diff'] # a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

# project.reclassify(project, rec_id, rec_type, threshold_ratio, likelihood)

# # undo classification 
# # project.undo_classification(rec_id)

#%% cross validate


#%% calculate bouts
# get nodes
node = 'R14b'

# create a bout object
bout = pymast.bout(project, node, 2, 21600)
    
# Find the threshold
threshold = bout.fit_processes()
    
# calculate presences - or pass float
bout.presence(threshold)
# bout.presence(3600)

# undo bouts
# project.undo_bouts(node)
    
#%% reduce overlap
# create edges showing parent:child relationships for nodes in network
edges = []
for i in project.receivers.index:
    for j in ['R14a','R14b']:
        if i != j:
            edges.append((i, j))
        
print (edges)

#nodes = project.receivers.index
nodes = project.receivers.index

# create an overlap object and apply one of the nested doll algorithms
doll = pymast.overlap_reduction(nodes, edges, project)
# doll.nested_doll() 
doll.unsupervised_removal()

#project.undo_overlap()

#%% create a recaptures table
#project.undo_recaptures()
project.make_recaptures_table(export = False) 

#%% create Time to Event Model
    
# what is the Node to State relationship - use Python dictionary
node_to_state = {'R01':1,'R02':2,'R03':3,'R04':4,'R10':10,'R11':11,'R12':12,
                 'R13':13,'R14':14,'R14a':14,'R14b':14,'R15':15,'R16':16
                }                   

# Step 1, create time to event data class 
tte = formatter.time_to_event(node_to_state,
                              project,
                              initial_state_release = True)

# Step 2, format data - without covariates
tte.data_prep(project, 
              # adjacency_filter = [(4, 1),(4, 2),(4, 3),(4, 8), (4, 9), (4, 12), (4, 13), (4, 15),(4, 16),
              #                     (5, 1),(5, 2),(5, 9),(5, 12),(5, 13),(5, 16),
              #                     (6, 2),(6, 10),
              #                     (7, 1),(7, 8),
              #                     (8,10),(8,11),(8,12),
              #                     (9, 1),(9, 2),(9, 3),(9, 8),(9, 12),(9, 13),(9,16),
              #                     (10, 1),(10, 2),(10, 8),(10, 9),(10, 12),(10, 13),(10, 16),(10, 17),
              #                     (11, 1),(11, 2),(11, 8),(11, 9),(11, 12),(11, 13),(11, 15),(11, 16),
              #                     (19, 1),(19, 2),(19, 3),(19, 13),(19, 14),
              #                     (20, 1),(20, 2),(20, 3),(20, 13),(20, 15),(20, 16),(20, 17)]
              )
# Step 3, generate a summary
stats = tte.summary()
tte.master_state_table.to_csv(os.path.join(project_dir,'Output','nuyakuk_tte.csv'))




#%% create a Cormack-Jolly-Seber Mark Recapture model
# what is the output directory?
output_ws = os.path.join(project_dir,'Output')
model_name = "york_haven"

# what is the Node to State relationship - use Python dictionary
receiver_to_recap = {'R001':'R01','R002':'R01',
                      'R003':'R02','R006':'R02',
                      'R007':'R02','R008':'R02','R009':'R02','R010':'R02',
                      'R011':'R03','R012':'R02','R013':'R02','R014':'R02',
                      'R015':'R02','R016':'R02','R017':'R02',#'R018':'R02',
                      'R019':'R03','R020':'R03',
                      'R004':'R04','R005':'R04',}

# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = formatter.cjs_data_prep(receiver_to_recap, project, initial_recap_release = False)
print ("Step 1 Completed, Data Class Finished")

# Step 2, Create input file for MARK
cjs.input_file(model_name,output_ws)
cjs.inp.to_csv(os.path.join(output_ws,model_name + '.csv'), index = False)

print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

