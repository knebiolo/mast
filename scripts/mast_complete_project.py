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
project_dir = r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\PYMAST Round 2"
db_name = 'nuyakuk_kpn_v2'
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
# rec_id = 'R11'
# rec_type = 'ares'
# #TODO - remove these directory arguments - the project is smart
# training_dir = os.path.join(project_dir,'Data',rec_id)
# db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
# scan_time = 10.5 #10.5        
# channels = 2. #2
# antenna_to_rec_dict = {'1':rec_id}

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
# rec_id = 'R11'
# rec_type = 'ares'
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
# rec_id = 'R11'
# rec_type = 'ares'
# threshold_ratio = 1  # 1.0 = MAP Hypothesis
# likelihood = ['hit_ratio','cons_length', 'noise_ratio', 'power', 'lag_diff'] # a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

# project.reclassify(project=project, rec_id=rec_id, threshold_ratio=threshold_ratio, 
#                     likelihood_model=likelihood, rec_type=rec_type, rec_list=None)

# # undo classification 
# # project.undo_classification(rec_id)

# #%% cross validate


# #%% calculate bouts
# # get nodes
# node = 'R16'

# # create a bout object
# bout = pymast.bout(project, node, 2, 21600)
    
# # Find the threshold
# threshold = bout.fit_processes()
    
# # calculate presences - or pass float
# bout.presence(threshold)
# # bout.presence(3600)

# # undo bouts
# # project.undo_bouts(node)
    
# #%% reduce overlap
# # create edges showing parent:child relationships for nodes in network

# # parents
# for i in project.receivers.index:
#     # children
#     for j in project.receivers.index:
#         if i != j:
#             edges = []
#             edges.append((i, j))
#             nodes = [i,j]
            
#             # create an overlap object and apply one of the nested doll algorithms
#             doll = pymast.overlap_reduction(nodes, edges, project)
            
#             # doll.nested_doll() 
#             doll.unsupervised_removal()

# #nodes = project.receivers.index
# nodes = project.receivers.index

# #project.undo_overlap()

# #%% create a recaptures table
# #project.undo_recaptures()
# project.make_recaptures_table(export = False) 

#%% create Time to Event Model
    
# what is the Node to State relationship - use Python dictionary
node_to_state = {'R01':1,'R02':1,'R10':2,'R12':2,'R03':3,'R04':3}
# node_to_state = {'R01':1,'R02':2,'R03':3,'R04':4,'R10':10,'R11':11,'R12':12,
#                  'R13':13,'R14':14,'R14a':14,'R14b':14,'R15':15,'R16':16
#                 }                   

# Step 1, create time to event data class 
tte = formatter.time_to_event(node_to_state,
                              project,
                              initial_state_release = False,
                              cap_loc = None,
                              rel_loc = 'Kleinschmidt',
                              species = 'Sockeye',
                              rel_date = '01-01-2023')

# Step 2, format data - without covariates
tte.data_prep(project, 
                adjacency_filter = [(3, 1),(3, 2),
                                    (2, 1)]
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

