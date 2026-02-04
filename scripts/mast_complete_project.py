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
from pymast.radio_project import radio_project
from pymast import formatter as formatter
import pymast
import pandas as pd
import matplotlib.pyplot as plt

#%% set up project
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_12_04"  # UPDATE THIS
db_name = 'thompson_2025_qc_1'

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
rec_id = 'R1695'
rec_type = 'PIT'
 #TODO - remove these directory arguments - the project is smart
training_dir = os.path.join(project_dir,'Data','Training_Files')
db_dir = os.path.join(project_dir,'%s.h5'%(db_name))
scan_time = 1 #10.5        
channels = 1. #2
antenna_to_rec_dict = {1:'R0001',
                        2:'R0002',
                        3:'R0003',
                        4:'R0004',
                        5:'R0005'}

project.telem_data_import(rec_id,
                           rec_type,
                           training_dir,
                           db_dir,
                           scan_time,
                           channels,
                           None,
                           True)

# undo import
# project.undo_import(rec_id)

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


#%% Calculate bouts using DBSCAN (receiver-based)
"""
DBSCAN-based bout detection with physics-based epsilon.
- Runs automatically during initialization (no fit_processes needed)
- epsilon = pulse_rate * eps_multiplier (default 5x)
- Recommended: eps_multiplier=5, lag_window=2
"""

# Undo existing bouts if rerunning
# project.undo_bouts()

print("\n" + "="*80)
print("BOUT DETECTION (DBSCAN)")
print("="*80)

receivers = list(project.receivers.index)
print(f"Processing {len(receivers)} receivers with DBSCAN clustering...")

successful_receivers = []
failed_receivers = []

for rec_id in receivers:
    print(f"\n[{rec_id}] Starting bout detection...")
    
    # Create bout object - DBSCAN runs automatically during __init__
    # eps_multiplier=5: epsilon = 5x pulse rate (~40-50 sec for 8-10 sec tags)
    # lag_window=2: kept for compatibility (not used in DBSCAN)
    bout = pymast.bout(project, rec_id, eps_multiplier=60, lag_window=2)
    
    # Write results to database
    bout.presence()
    bout.visualize_bout_lengths()  # Creates plots and statistics
    
    successful_receivers.append(rec_id)
    print(f"[{rec_id}] ✓ Complete")
        
# To undo all bouts: project.undo_bouts()
# To undo specific receiver: project.undo_bouts(rec_id='R03')
    
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
doll.unsupervised_removal(method='power', 
                         p_value_threshold=0.05,     # p_value_threshold=0.05: require statistical significanc
                         effect_size_threshold=0.1,  # effect_size_threshold=0.2: small effect size (more sensitive to differences)
                         min_detections=5,           # min_detections=2: allow smaller bouts (need at least 2 for t-test)
                         bout_expansion=60)          # bout_expansion=30: ±30 second buffer to catch near-overlaps
doll.visualize_overlaps()  # Creates comprehensive visualization
print('Overlap reduction took', time.time() - start_t, 'seconds')

# nested doll
# create edges showing parent:child relationships for nodes in network
# edges =[('R04','R15'),('R04','R14'),('R04','R13'),('R04','R12'),('R04','R10'),('R04','R08'),('R04','R09'),('R04','R05'),('R04','R03'),
#         ('R03','R15'),('R03','R14'),('R03','R13'),('R03','R12'),('R03','R10'),('R03','R08'),('R03','R09'),('R03','R05'),('R03','R04'),
#         ('R08','R10'),
#         ('R09','R10')]
# nodes = ['R03','R04','R05','R08','R09','R10','R12','R13','R14','R15']
# nested = pymast.overlap_reduction(nodes, edges, project)
# nested.nested_doll()  

# project.undo_overlap()

#%% create a recaptures table
# project.undo_recaptures()
# project.repack_database()

#project.make_recaptures_table(export = True, pit_study = True) 

#%% create Time to Event Model
    
# what is the Node to State relationship - use Python dictionary
# node_to_state = {'T5':1,'T6':1,'T7':2,'T15':3,'T3':4}
states = {'R1696':1,
          'R1699-1':2,
          'R1699-2':3,
          'R1698':4,
          'R1699-3':5,
          'R1695':5,
          'R0004':6,
          'R0005':6,
          'R0001':7,
          'R0002':7,
          'R0003':8} 

# states = {'R1699-3':5,
#           'R1695':5,
#           'R0004':6,
#           'R0005':6,
#           'R0001':7,
#           'R0002':7,
#           'R0003':8}
                                   
# Step 1: Create time to event data class 
tte = formatter.time_to_event(states,
                              project,
                              initial_state_release = True,
                              last_presence_time0 = False,
                              hit_ratio_filter = False,
                              cap_loc = None,
                              rel_loc = None,
                              species = "BULL",
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
tte.data_prep(project)
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

