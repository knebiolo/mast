"""
MAST Example Script - Complete Telemetry Project Workflow

This script demonstrates a complete MAST workflow from project setup through
statistical output generation. Modify the parameters in the CONFIGURATION section
to match your project.

Author: MAST Development Team
Date: 2025-10-06
"""

# Import modules
import os
import sys
import pandas as pd
import pymast
from pymast.radio_project import radio_project

# =============================================================================
# CONFIGURATION - Edit these parameters for your project
# =============================================================================

# Project directories - UPDATE THESE
PROJECT_DIR = r"path/to/your/project"  # Root project directory
DB_NAME = 'my_study'  # Database name (no spaces recommended)

# Training parameters
DETECTION_COUNT = 5  # Number of detections to look ahead/behind for detection history
DURATION = 1  # Time window for noise ratio calculation (minutes)

# Receiver information for import - UPDATE FOR EACH RECEIVER
REC_ID = 'R01'  # Receiver ID from tblMasterReceiver.csv
REC_TYPE = 'srx800'  # Options: 'srx600', 'srx800', 'srx1200', 'orion', 'ares', 'VR2'
SCAN_TIME = 1.0  # Channel scan time (seconds) - use 1 if not applicable
CHANNELS = 1  # Number of channels - use 1 if not applicable

# Classification parameters
THRESHOLD_RATIO = 1.0  # 1.0 = Maximum A Posteriori, >1.0 = more strict, <1.0 = less strict
LIKELIHOOD_MODEL = ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff']  
# Available: 'hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff'

# Bout detection parameters
NODE = 'R01'  # Node ID for bout detection
LAG_WINDOW = 2  # Lag window for bout detection (seconds)
TIME_LIMIT = 21600  # Maximum time between detections to consider (seconds)

# Overlap reduction - define parent:child relationships
OVERLAP_EDGES = [
    ('R04', 'R15'), ('R04', 'R14'), ('R04', 'R13'),
    # Add more edges as needed: (parent_receiver, child_receiver)
]
OVERLAP_NODES = ['R04', 'R15', 'R14', 'R13']  # All nodes involved in overlap

# =============================================================================
# PART 1: PROJECT SETUP
# =============================================================================

def setup_project():
    """Initialize a new MAST project."""
    print("=" * 80)
    print("MAST PROJECT SETUP")
    print("=" * 80)
    
    # Load metadata files
    tag_file = os.path.join(PROJECT_DIR, 'tblMasterTag.csv')
    receiver_file = os.path.join(PROJECT_DIR, 'tblMasterReceiver.csv')
    nodes_file = os.path.join(PROJECT_DIR, 'tblNodes.csv')
    
    # Check if files exist
    for file_path, file_name in [(tag_file, 'tblMasterTag.csv'), 
                                   (receiver_file, 'tblMasterReceiver.csv'),
                                   (nodes_file, 'tblNodes.csv')]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_name}\n"
                                   f"Please ensure it exists at: {file_path}")
    
    tag_data = pd.read_csv(tag_file)
    receiver_data = pd.read_csv(receiver_file)
    nodes_data = pd.read_csv(nodes_file)
    
    # Create project
    project = radio_project(
        project_dir=PROJECT_DIR,
        db_name=DB_NAME,
        detection_count=DETECTION_COUNT,
        duration=DURATION,
        tag_data=tag_data,
        receiver_data=receiver_data,
        nodes_data=nodes_data
    )
    
    print(f"✓ Project '{DB_NAME}' created successfully")
    print(f"✓ Database location: {project.db}")
    print(f"✓ Loaded {len(tag_data)} tags")
    print(f"✓ Loaded {len(receiver_data)} receivers")
    print(f"✓ Loaded {len(nodes_data)} nodes")
    
    return project


# =============================================================================
# PART 2: DATA IMPORT
# =============================================================================

def import_receiver_data(project):
    """Import raw telemetry data from receiver files."""
    print("\n" + "=" * 80)
    print("DATA IMPORT")
    print("=" * 80)
    
    training_dir = os.path.join(PROJECT_DIR, 'Data', 'Training_Files')
    
    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"Training directory not found: {training_dir}\n"
                               "Please create it and add receiver data files.")
    
    # Map antennas to receivers (adjust for your setup)
    antenna_to_rec_dict = {'A0': REC_ID}
    
    print(f"Importing data for receiver: {REC_ID}")
    print(f"Receiver type: {REC_TYPE}")
    
    project.telem_data_import(
        rec_id=REC_ID,
        rec_type=REC_TYPE,
        file_dir=training_dir,
        db_dir=project.db,
        scan_time=SCAN_TIME,
        channels=CHANNELS,
        ant_to_rec_dict=antenna_to_rec_dict,
        ka_format=True
    )
    
    print(f"✓ Data imported for receiver {REC_ID}")


# =============================================================================
# PART 3: TRAINING DATA CREATION
# =============================================================================

def train_receiver(project):
    """Create training data for Naive Bayes classifier."""
    print("\n" + "=" * 80)
    print("TRAINING DATA CREATION")
    print("=" * 80)
    
    # Get list of fish detected at this receiver
    fishes = project.get_fish(rec_id=REC_ID)
    
    print(f"Found {len(fishes)} fish at receiver {REC_ID}")
    
    # Train each fish
    for i, fish in enumerate(fishes, 1):
        print(f"Training fish {i}/{len(fishes)}: {fish}")
        project.train(fish, REC_ID)
    
    # Generate summary statistics
    print("\nGenerating training summary...")
    project.training_summary(REC_TYPE, site=[REC_ID])
    
    print(f"✓ Training complete for receiver {REC_ID}")


# =============================================================================
# PART 4: CLASSIFICATION
# =============================================================================

def classify_receiver(project):
    """Classify detections as true or false positive."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION")
    print("=" * 80)
    
    print(f"Classifying receiver: {REC_ID}")
    print(f"Threshold ratio: {THRESHOLD_RATIO}")
    print(f"Likelihood model: {', '.join(LIKELIHOOD_MODEL)}")
    
    project.reclassify(
        project=project,
        rec_id=REC_ID,
        threshold_ratio=THRESHOLD_RATIO,
        likelihood_model=LIKELIHOOD_MODEL,
        rec_type=REC_TYPE,
        rec_list=None
    )
    
    print(f"✓ Classification complete for receiver {REC_ID}")


# =============================================================================
# PART 5: BOUT DETECTION (OPTIONAL)
# =============================================================================

def detect_bouts(project):
    """Identify presence/absence bouts at a node."""
    print("\n" + "=" * 80)
    print("BOUT DETECTION")
    print("=" * 80)
    
    print(f"Detecting bouts at node: {NODE}")
    
    # Create bout object
    bout = pymast.bout(project, NODE, LAG_WINDOW, TIME_LIMIT)
    
    # Fit processes to find optimal threshold
    print("Fitting bout processes...")
    threshold = bout.fit_processes()
    
    # Calculate presences using fitted threshold
    print(f"Calculating presences with threshold: {threshold}")
    bout.presence(threshold)
    
    print(f"✓ Bout detection complete for node {NODE}")


# =============================================================================
# PART 6: OVERLAP REMOVAL (OPTIONAL)
# =============================================================================

def remove_overlap(project):
    """Remove overlapping detections between receivers."""
    print("\n" + "=" * 80)
    print("OVERLAP REMOVAL")
    print("=" * 80)
    
    print(f"Removing overlap for {len(OVERLAP_NODES)} nodes")
    print(f"Processing {len(OVERLAP_EDGES)} parent-child relationships")
    
    # Create overlap reduction object
    nested = pymast.overlap_reduction(OVERLAP_NODES, OVERLAP_EDGES, project)
    
    # Run nested doll algorithm
    nested.nested_doll()
    
    print("✓ Overlap removal complete")


# =============================================================================
# PART 7: CREATE RECAPTURES TABLE
# =============================================================================

def create_recaptures(project):
    """Compile final recaptures table for statistical analysis."""
    print("\n" + "=" * 80)
    print("CREATING RECAPTURES TABLE")
    print("=" * 80)
    
    project.make_recaptures_table(export=True)
    
    output_file = os.path.join(project.output_dir, 'recaptures.csv')
    print(f"✓ Recaptures table created")
    print(f"✓ Exported to: {output_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete MAST workflow."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MAST TELEMETRY ANALYSIS WORKFLOW" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Step 1: Setup project
        project = setup_project()
        
        # Step 2: Import data (run once per receiver)
        # Uncomment when ready:
        # import_receiver_data(project)
        
        # Step 3: Train data (run once per receiver after import)
        # Uncomment when ready:
        # train_receiver(project)
        
        # Step 4: Classify data (can be run multiple times with different parameters)
        # Uncomment when ready:
        # classify_receiver(project)
        
        # Step 5: Bout detection (optional, run once per node)
        # Uncomment when ready:
        # detect_bouts(project)
        
        # Step 6: Overlap removal (optional, run once for entire project)
        # Uncomment when ready:
        # remove_overlap(project)
        
        # Step 7: Create final recaptures table
        # Uncomment when ready:
        # create_recaptures(project)
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Uncomment the functions you want to run")
        print("2. Update REC_ID and other parameters for each receiver")
        print("3. Run the script again for each receiver in your project")
        print("4. Use fish_history to visualize results")
        print("5. Export data for statistical analysis")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
