"""
Overlap Reduction Script
Identifies and marks overlapping detections between receivers using statistical methods.

@author: KNebiolo
"""

# import modules
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('DEBUG: sys.executable=', sys.executable)
print('DEBUG: repo_root=', repo_root)
print('DEBUG: sys.path[0]=', sys.path[0])
from pymast.radio_project import radio_project
import pymast
import pandas as pd
import time

# Set up project
project_dir = r"C:\Users\Kevin.Nebiolo\Desktop\Scotland KPN"
db_name = 'Scotland_repacked'

detection_count = 5
duration = 1
tag_data = pd.read_csv(os.path.join(project_dir,'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir,'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir,'tblNodes.csv'))

# Create/load project
project = radio_project(project_dir,
                        db_name,
                        detection_count,
                        duration,
                        tag_data,
                        receiver_data,
                        nodes_data)

#%% Reduce overlap
# Choose overlap reduction method, we have an unsupervised method or the nested doll

# Unsupervised - initialize one overlap object for all receiver pairs (much faster)
nodes = list(project.receivers.index)
edges = [(i, j) for i in nodes for j in nodes if i != j]

# Create single overlap object (loads data once)
print('Initializing overlap reduction for all nodes...')
start_t = time.time()

doll = pymast.overlap_removal.overlap_reduction(nodes, edges, project)
print('Running unsupervised removal with statistical testing (t-test + Cohen\'s d)')
# Statistical approach: uses t-test + effect size for cleaner movement trajectories
# p_value_threshold=0.05: require statistical significance
# effect_size_threshold=0.2: small effect size (more sensitive to differences)
# min_detections=1: allow smaller bouts (need at least 1 for comparison)
# bout_expansion=30: ±30 second buffer to catch near-overlaps
doll.unsupervised_removal(method='posterior', 
                         p_value_threshold=0.05, 
                         effect_size_threshold=0.2, 
                         min_detections=1, 
                         bout_expansion=30)
print('Overlap reduction took', time.time() - start_t, 'seconds')

# Alternative: Nested doll method
# Create edges showing parent:child relationships for nodes in network
# edges =[('R04','R15'),('R04','R14'),('R04','R13'),('R04','R12'),('R04','R10'),('R04','R08'),('R04','R09'),('R04','R05'),('R04','R03'),
#         ('R03','R15'),('R03','R14'),('R03','R13'),('R03','R12'),('R03','R10'),('R03','R08'),('R03','R09'),('R03','R05'),('R03','R04'),
#         ('R08','R10'),
#         ('R09','R10')]
# nodes = ['R03','R04','R05','R08','R09','R10','R12','R13','R14','R15']
# nested = pymast.overlap_removal.overlap_reduction(nodes, edges, project)
# nested.nested_doll()  

# Undo overlap if needed
# project.undo_overlap()
