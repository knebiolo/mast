"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
sys.path.append(r"C:\Users\EMuhlestein\OneDrive - Kleinschmidt Associates, Inc\Software\MAST\mast")
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"C:\Users\EMuhlestein\Documents\Thompson\TEST"
db_name = 'Thompson_Data_04292025'
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

#%%  import data
rec_id = 'R2024'
rec_type = 'PIT_Multiple'
#TODO - remove these directory arguments - the project is smart
training_dir = os.path.join(project_dir,'Data', 'Training_Files')
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 1 #10.5        
channels = 1. #2
antenna_to_rec_dict = {'1':rec_id}

project.telem_data_import(rec_id,
                          rec_type,
                          training_dir,
                          db_dir,
                          scan_time,
                          channels,
                          antenna_to_rec_dict,
                          True)

# # undo import
# # project.undo_import(rec_id)

# #%%  train data
# # set parameters and get a list of fish to iterate over
# rec_id = 'T15'
# rec_type = 'orion'
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
# rec_id = 'T15'
# rec_type = 'orion'
# threshold_ratio = 1  # 1.0 = MAP Hypothesis
# likelihood = ['hit_ratio','cons_length','noise_ratio', 'power', 'lag_diff'] # a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

# project.reclassify(project=project, rec_id=rec_id, threshold_ratio=threshold_ratio, 
#                     likelihood_model=likelihood, rec_type=rec_type, rec_list=None)

# # undo classification 
# # project.undo_classification(rec_id)

# #%% cross validate


# #%% calculate bouts
# # get nodes
# node = 'T15'

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

# # unsupervised
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

# # nested doll
# edges =[('T5','T7'),('T6','T7')]
# nodes = ['T7','T5','T6']
# nested = pymast.overlap_reduction(nodes, edges, project)
# nested.nested_doll() 

# #nodes = project.receivers.index
# nodes = project.receivers.index

# #project.undo_overlap()

# #%% create a recaptures table
# #project.undo_recaptures()
project.make_recaptures_table(export = True) 

#%% create Time to Event Model
    
# what is the Node to State relationship - use Python dictionary
# node_to_state = {'T5':1,'T6':1,'T7':2,'T15':3,'T3':4}

# # #Submersible vs ladder
# node_to_state = {'R1696':1,'R1698':1,'R1699':1,'R1695':1, 'R0004': 2, 'R0005':2,'R0001':2,
#                   'R0002':2,'R0003':2
#                 }   

# # #Entering/exiting ladder
# node_to_state = {'R1696':1, 'R1698': 1, 'R0004': 2, 'R0005':2
#                 } 

# #Full arrays
node_to_state = {'R1696':1,'R1698':2,'R1699':2,'R1695':4, 'R0004': 5, 'R0005':5,'R0001':6,
                  'R0002':7,'R0003':8
                }

# #Submersible split vs ladder
# node_to_state = {'R1696':1,'R1698':2,'R1699':3,'R1695':4, 'R0004': 5, 'R0005':5,'R0001':5,
#                   'R0002':5,'R0003':5
#                 }                 

# Step 1, create time to event data class 
tte = formatter.time_to_event(node_to_state,
                              project,
                              initial_state_release = False,
                              last_presence_time0 = False,
                              cap_loc = None,
                              rel_loc = None,
                              species = None,
                              rel_date = None,
                              recap_date = None)

# Step 2, format data - without covariates
tte.data_prep(project)#, 
              #  adjacency_filter = [(3, 1),(3, 2),
              #                      (2, 1)]
              #)
# Step 3, generate a summary
stats = tte.summary()
tte.master_state_table.to_csv(os.path.join(project_dir,'Output','cabot_tailrace.csv'))







#Print off dataframes of the Moement Summary, State Table, Tailrace Table
# ensure Spyder prints every column
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

out = os.path.join(project_dir, "Output")
df_movement_summary = pd.read_csv(os.path.join(out, "movement_summary.csv"))
df_state_table      = pd.read_csv(os.path.join(out, "state table.csv"))
df_tailrace         = pd.read_csv(os.path.join(out, "cabot_tailrace.csv"))

# --- print to console ---
print("=== Movement Summary ===")
print(df_movement_summary, "\n")  

print("=== State Table ===")
print(df_state_table, "\n")

print("=== Tailrace Table ===")
# head(14) plus full‑column display:
print(df_tailrace.head(14), "\n")




# #%% create a Cormack-Jolly-Seber Mark Recapture model
# # what is the output directory?
# output_ws = os.path.join(project_dir,'Output')
# model_name = "nuyakuk"

# # what is the Node to State relationship - use Python dictionary
# receiver_to_recap = {'R01':'R01','R02':'R01',
#                      'R10':'R02','R12':'R02',
#                      'R13':'R03','R14':'R03','R14a':'R03','R14b':'R03',
#                      'R15':'R04','R16':'R04',
#                      'R03':'R05','R04':'R05'}

# # Step 1, create time to event data class - we only need to feed it the directory and file name of input data
# cjs = formatter.cjs_data_prep(receiver_to_recap, project, species = 'Sockeye', initial_recap_release = False)
# print ("Step 1 Completed, Data Class Finished")

# # Step 2, Create input file for MARK
# cjs.input_file(model_name,output_ws)
# cjs.inp.to_csv(os.path.join(output_ws,model_name + '.csv'), index = False)

# print ("Step 2 Completed, MARK Input file created")
# print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

