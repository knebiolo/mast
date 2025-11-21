"""
Created on Thu Nov 16 09:42:22 2023

@author: KNebiolo
"""

# import modules
import os
import sys
# Ensure the repository root (one level up from scripts/) is on sys.path so imports like
# `import pymast` work regardless of the absolute path or username. This is safer than a
# hard-coded path which can break on different machines or user accounts.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('DEBUG: sys.executable=', sys.executable)
print('DEBUG: repo_root=', repo_root)
print('DEBUG: sys.path[0]=', sys.path[0])
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"C:\Users\Kevin.Nebiolo\Desktop\Scotland KPN"
db_name = 'Scotland'

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
# Leave interactive by default - set project.non_interactive = True only for automated runs
# project.non_interactive = True

#create a new version of an existing project 
#project.new_db_version(os.path.join(project_dir,'Nuyakuk_v2.h5'))
#,presence,classified,trained

#%%  import data
# rec_id = 'R15'
# rec_type = 'srx800'
#  #TODO - remove these directory arguments - the project is smart
# training_dir = os.path.join(project_dir,'Data','Training_Files')
# db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
# scan_time = 1 #10.5        
# channels = 1. #2
# antenna_to_rec_dict = {'A0':rec_id}

# project.telem_data_import(rec_id,
#                            rec_type,
#                            training_dir,
#                            db_dir,
#                            scan_time,
#                            channels,
#                            antenna_to_rec_dict,
#                            True)

# # undo import
# # project.undo_import(rec_id)

#%%  train data
# # set parameters and get a list of fish to iterate over
# rec_id = 'R15'
# rec_type = 'srx800'
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
# rec_id = 'R15'
# rec_type = 'srx800'
# threshold_ratio = 1  # 1.0 = MAP Hypothesis
# likelihood = ['hit_ratio','cons_length','noise_ratio', 'lag_diff'] 
# # a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

# project.reclassify(project=project, rec_id=rec_id, threshold_ratio=threshold_ratio, 
#                     likelihood_model=likelihood, rec_type=rec_type, rec_list=None)

# # undo classification 
# # project.undo_classification(rec_id)

#%% cross validate


#%% calculate bouts

receivers = list(project.receivers.index)

for rec_id in receivers:
    print(f"Processing bouts for {rec_id}")
    
    # Recommended parameters:
    # - eps_multiplier=5: longer bouts for better overlap detection (5x pulse rate)
    # - lag_window=2: tighter temporal window (2 sec) to reduce false positives
    bout = pymast.bout(project, rec_id, eps_multiplier=5, lag_window=2)
    
    # Fit mixture model to find threshold
    threshold = bout.fit_processes()
    
    # Calculate presences
    bout.presence(threshold)
    
    print(f"  ✓ Completed {rec_id}: threshold={threshold:.1f} seconds")

print(f"\nBouts complete for {len(receivers)} receivers")

# project.undo_bouts()
    
#%% reduce overlap
# Choose overlap reduction method, we have an unsupervised method or the nested doll

# unsupervised - initialize one overlap object for all receiver pairs (much faster)
nodes = list(project.receivers.index)
edges = [(i, j) for i in nodes for j in nodes if i != j]

# create single overlap object (loads data once)
print('Initializing overlap reduction for all nodes...')
import time
start_t = time.time()

doll = pymast.overlap_removal.overlap_reduction(nodes, edges, project)
print('Running unsupervised removal with statistical testing (t-test + Cohen\'s d)')
# Statistical approach: uses t-test + effect size for cleaner movement trajectories
# p_value_threshold=0.05: require statistical significance
# effect_size_threshold=0.2: small effect size (more sensitive to differences)
# min_detections=2: allow smaller bouts (need at least 2 for t-test)
# bout_expansion=30: ±30 second buffer to catch near-overlaps
doll.unsupervised_removal(method='posterior', 
                         p_value_threshold=0.05, 
                         effect_size_threshold=0.2, 
                         min_detections=1, 
                         bout_expansion=30)
print('Overlap reduction took', time.time() - start_t, 'seconds')

# nested doll
# create edges showing parent:child relationships for nodes in network
edges =[('R04','R15'),('R04','R14'),('R04','R13'),('R04','R12'),('R04','R10'),('R04','R08'),('R04','R09'),('R04','R05'),('R04','R03'),
        ('R03','R15'),('R03','R14'),('R03','R13'),('R03','R12'),('R03','R10'),('R03','R08'),('R03','R09'),('R03','R05'),('R03','R04'),
        ('R08','R10'),
        ('R09','R10')]
nodes = ['R03','R04','R05','R08','R09','R10','R12','R13','R14','R15']
nested = pymast.overlap_reduction(nodes, edges, project)
nested.nested_doll()  

# project.undo_overlap()

#%% create a recaptures table
# project.undo_recaptures()
# project.repack_database()

#project.make_recaptures_table(export = True) 

#%% create Time to Event Model
    
# what is the Node to State relationship - use Python dictionary
# node_to_state = {'T5':1,'T6':1,'T7':2,'T15':3,'T3':4}
upstream_states = {'R15':1,'R14':1,    # occum
                 'R13':2,'R12':2,    # downstream gate
                 'R10':3,            # tailrace
                 'R11':4,            # fish lift entrance
                 'R08':5,'R09':5,    # spillway
                 'R06':6,'R07':6,    # surface bypasss
                 'R05':7,            # submerged bypass
                 'R04':8,            # fish lift exit
                 'R03':9,            # forebay
                 'R01':10,'R02':10}  # windham

downstream_states = {'R15':1,'R14':1,# occum
                 'R13':2,'R12':2,    # downstream gate
                 'R10':2,            # tailrace
                 'R11':2,            # fish lift entrance
                 'R08':3,'R09':3,    # spillway
                 'R06':6,'R07':6,    # surface bypasss
                 'R05':7,            # submerged bypass
                 'R04':9,            # fish lift exit
                 'R03':9,            # forebay
                 'R01':10,'R02':10}  # windham
                                   

# Step 1, create time to event data class 
tte = formatter.time_to_event(upstream_states,
                              project,
                              initial_state_release = True,
                              last_presence_time0 = False,
                              cap_loc = None,
                              rel_loc = 'windham',
                              species = None,
                              rel_date = None,
                              recap_date = None)

# upstream_adjacency_filter = [(9, 1),(9, 2),(9, 3),(9, 5),(9, 8),(9, 9),(9, 6),
#                               (8, 1),(8, 2),(8, 3),(8, 5),(8, 9),(8, 8),(8, 6),
#                                (6, 1),(6, 6),(6, 5),
#                                 (1, 8),(1, 9),(2, 9),(2, 8),(3, 8),(3, 9),(0, 8),(0, 9),(5, 9)]

# downstream_adjacency_filter = [(1, 6),(1, 7),(1, 8),(1, 9),
#                                (9, 1),(8, 1),(9, 2),(9, 3),
#                                (2, 9),(2, 7),(2, 6),(2, 10),(2, 3),
#                                (3, 2),(3, 9)]

# Step 2, format data - without covariates
tte.data_prep(project)#, adjacency_filter = downstream_adjacency_filter)
# Step 3, generate a summary
stats = tte.summary()
#tte.master_state_table.to_csv(os.path.join(project_dir,'Output','thompson_falls.csv'))



# #Print off dataframes of the Moement Summary, State Table, Tailrace Table
# # ensure Spyder prints every column
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

# out = os.path.join(project_dir, "Output")
# df_movement_summary = pd.read_csv(os.path.join(out, "movement_summary.csv"))
# df_state_table      = pd.read_csv(os.path.join(out, "state_table.csv"))
# df_recaptures       = pd.read_csv(os.path.join(out, "recaptures.csv"))

# # --- print to console ---
# print("=== Movement Summary ===")
# print(df_movement_summary, "\n")  

# print("=== State Table ===")
# print(df_state_table, "\n")

# print("== recaptures ==")
# print(df_recaptures.head(5), "\n")

# # #%% create a Cormack-Jolly-Seber Mark Recapture model
# # # what is the output directory?
# # output_ws = os.path.join(project_dir,'Output')
# # model_name = "nuyakuk"

# # # what is the Node to State relationship - use Python dictionary
# # receiver_to_recap = {'R01':'R01','R02':'R01',
# #                      'R10':'R02','R12':'R02',
# #                      'R13':'R03','R14':'R03','R14a':'R03','R14b':'R03',
# #                      'R15':'R04','R16':'R04',
# #                      'R03':'R05','R04':'R05'}

# # # Step 1, create time to event data class - we only need to feed it the directory and file name of input data
# # cjs = formatter.cjs_data_prep(receiver_to_recap, project, species = 'Sockeye', initial_recap_release = False)
# # print ("Step 1 Completed, Data Class Finished")

# # # Step 2, Create input file for MARK
# # cjs.input_file(model_name,output_ws)
# # cjs.inp.to_csv(os.path.join(output_ws,model_name + '.csv'), index = False)

# # print ("Step 2 Completed, MARK Input file created")
# # print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

