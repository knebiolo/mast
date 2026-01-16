"""
Classification Script
Classifies detections as true or false positive using Naive Bayes classifier.

@author: KNebiolo
"""

# import modules
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from pymast.radio_project import radio_project
import pandas as pd

# Set up project
project_dir = r"C:\path\to\your\project"  # UPDATE THIS
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

#%% Classify data
# Set initial parameters
rec_id = 'R15'
rec_type = 'srx800'
threshold_ratio = 1  # 1.0 = MAP Hypothesis
likelihood = ['hit_ratio','cons_length','noise_ratio', 'lag_diff'] 
# a-la carte likelihood, standard fields: ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']

project.reclassify(project=project, rec_id=rec_id, threshold_ratio=threshold_ratio, 
                    likelihood_model=likelihood, rec_type=rec_type, rec_list=None)

# Undo classification if needed
# project.undo_classification(rec_id)
