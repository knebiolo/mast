"""
MAST Quick Start Example
========================

This script demonstrates a complete MAST workflow from project creation 
through classification and overlap removal.

Before running:
1. Create input files (tblMasterTag.csv, tblMasterReceiver.csv, tblNodes.csv)
2. Place raw receiver data in Data/Training_Files/
3. Update the project_dir variable below
"""

import os
import pandas as pd
from pymast.radio_project import radio_project
import pymast

# =============================================================================
# CONFIGURATION - Update these paths for your project
# =============================================================================

# Set your project directory (no spaces recommended)
project_dir = r"C:\path\to\your\project"  # UPDATE THIS
db_name = 'my_telemetry_project'

# Training parameters
detection_count = 5  # Number of detections required in detection history
duration = 1  # Duration window for noise ratio calculation (seconds)

# =============================================================================
# STEP 1: Initialize Project
# =============================================================================

print("=" * 80)
print("STEP 1: Loading project data and initializing database")
print("=" * 80)

# Load your input data files
tag_data = pd.read_csv(os.path.join(project_dir, 'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir, 'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir, 'tblNodes.csv'))

# Create the project
project = radio_project(
    project_dir,
    db_name,
    detection_count,
    duration,
    tag_data,
    receiver_data,
    nodes_data
)

print(f"✓ Project initialized at: {project.db}")
print(f"✓ Study tags: {len(project.study_tags)}")
print(f"✓ Receivers: {len(project.receivers)}")

# =============================================================================
# STEP 2: Import Raw Data
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: Importing raw telemetry data")
print("=" * 80)

# Configure for your receiver
rec_id = 'R01'  # UPDATE THIS
rec_type = 'srx800'  # Options: 'srx600', 'srx800', 'srx1200', 'orion', 'ares', 'VR2'
scan_time = 1.0  # Channel scan time (seconds)
channels = 1  # Number of channels
antenna_to_rec_dict = {'A0': rec_id}  # Map antennas to receivers

# Import the data
project.telem_data_import(
    rec_id,
    rec_type,
    project.training_dir,
    project.db,
    scan_time,
    channels,
    antenna_to_rec_dict,
    ka_format=False
)

print(f"✓ Data imported for receiver {rec_id}")

# =============================================================================
# STEP 3: Train the Classifier
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: Training Naive Bayes classifier")
print("=" * 80)

# Get list of fish detected at this receiver
fishes = project.get_fish(rec_id=rec_id)
print(f"Found {len(fishes)} fish at receiver {rec_id}")

# Train on each fish
for i, fish in enumerate(fishes, 1):
    print(f"Training {i}/{len(fishes)}: {fish}")
    project.train(fish, rec_id)

# Generate and display training summary
project.training_summary(rec_type, site=[rec_id])

print(f"✓ Training complete for receiver {rec_id}")

# =============================================================================
# STEP 4: Classify Detections
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: Classifying detections")
print("=" * 80)

# Classification parameters
threshold_ratio = 1.0  # 1.0 = Maximum A Posteriori (MAP) hypothesis
likelihood_model = ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff']

# Classify the data
project.reclassify(
    project=project,
    rec_id=rec_id,
    threshold_ratio=threshold_ratio,
    likelihood_model=likelihood_model,
    rec_type=rec_type,
    rec_list=None
)

print(f"✓ Classification complete for receiver {rec_id}")

# =============================================================================
# STEP 5: Calculate Bouts (Optional)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: Calculating bouts and presences")
print("=" * 80)

# Create a bout object
node = rec_id  # Can be a node with multiple receivers
bout = pymast.bout(project, node, lag_window=2, time_limit=21600)

# Fit the bout model
threshold = bout.fit_processes()

# Calculate presences using the threshold
bout.presence(threshold)

print(f"✓ Bouts calculated for node {node}")

# =============================================================================
# STEP 6: Remove Overlap (if applicable)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: Removing overlapping detections")
print("=" * 80)

# Define parent:child relationships between receivers
# Example: edges = [('R01', 'R02'), ('R01', 'R03')]
edges = []  # UPDATE THIS based on your receiver layout
nodes = [rec_id]  # UPDATE THIS with your nodes

if edges:
    overlap_remover = pymast.overlap_reduction(nodes, edges, project)
    overlap_remover.nested_doll()
    print(f"✓ Overlap removed between receivers")
else:
    print("⊘ No overlap edges defined, skipping")

# =============================================================================
# STEP 7: Create Recaptures Table
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: Creating final recaptures table")
print("=" * 80)

# Compile all classified data into final recaptures table
project.make_recaptures_table(export=True)

print(f"✓ Recaptures table created and exported to: {project.output_dir}")

# =============================================================================
# DONE!
# =============================================================================

print("\n" + "=" * 80)
print("WORKFLOW COMPLETE!")
print("=" * 80)
print(f"\nYour processed data is available at:")
print(f"  Database: {project.db}")
print(f"  CSV Export: {os.path.join(project.output_dir, 'recaptures.csv')}")
print(f"\nNext steps:")
print("  - Review fish histories with pymast.fish_history")
print("  - Format data for statistical analysis (CJS, Competing Risks)")
print("  - Generate visualizations")
