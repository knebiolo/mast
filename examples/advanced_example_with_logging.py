"""
Advanced MAST Example - With Validation, Logging, and Error Handling
=====================================================================

This script demonstrates best practices for MAST including:
- Input data validation
- Logging configuration
- Proper error handling
- Batch processing
- Progress tracking
"""

import os
import sys
import logging
import pandas as pd
from pymast.radio_project import radio_project
from pymast import setup_logging, validate_tag_data, validate_receiver_data, validate_nodes_data, ValidationError
import pymast

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project settings
PROJECT_DIR = r"C:\path\to\your\project"  # UPDATE THIS
DB_NAME = 'telemetry_project'

# Training parameters
DETECTION_COUNT = 5
DURATION = 1.0

# Classification parameters
THRESHOLD_RATIO = 1.0
LIKELIHOOD_MODEL = ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff']

# Receivers to process (update with your receiver IDs)
RECEIVERS = [
    {'rec_id': 'R01', 'rec_type': 'srx800', 'scan_time': 1.0, 'channels': 1},
    {'rec_id': 'R02', 'rec_type': 'srx800', 'scan_time': 1.0, 'channels': 1},
    # Add more receivers as needed
]

# =============================================================================
# SETUP LOGGING
# =============================================================================

# Create log file in Output directory
log_file = os.path.join(PROJECT_DIR, 'Output', 'mast_analysis.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Set up logging
logger = setup_logging(level=logging.INFO, log_file=log_file)
logger.info("=" * 80)
logger.info("MAST Analysis Started")
logger.info("=" * 80)

# =============================================================================
# STEP 1: LOAD AND VALIDATE INPUT DATA
# =============================================================================

logger.info("Step 1: Loading and validating input data")

try:
    # Load input files
    tag_file = os.path.join(PROJECT_DIR, 'tblMasterTag.csv')
    receiver_file = os.path.join(PROJECT_DIR, 'tblMasterReceiver.csv')
    nodes_file = os.path.join(PROJECT_DIR, 'tblNodes.csv')
    
    logger.info(f"Loading tag data from: {tag_file}")
    tag_data = pd.read_csv(tag_file)
    logger.info(f"  Found {len(tag_data)} tags")
    
    logger.info(f"Loading receiver data from: {receiver_file}")
    receiver_data = pd.read_csv(receiver_file)
    logger.info(f"  Found {len(receiver_data)} receivers")
    
    logger.info(f"Loading nodes data from: {nodes_file}")
    nodes_data = pd.read_csv(nodes_file)
    logger.info(f"  Found {len(nodes_data)} nodes")
    
    # Validate input data
    logger.info("Validating input data...")
    validate_tag_data(tag_data)
    validate_receiver_data(receiver_data)
    validate_nodes_data(nodes_data)
    logger.info("✓ All input data validated successfully")
    
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
    logger.error("Please ensure all input CSV files are in the project directory")
    sys.exit(1)
    
except ValidationError as e:
    logger.error(f"Input data validation failed: {e}")
    logger.error("Please check your input files and try again")
    sys.exit(1)
    
except Exception as e:
    logger.error(f"Unexpected error loading input data: {e}")
    sys.exit(1)

# =============================================================================
# STEP 2: INITIALIZE PROJECT
# =============================================================================

logger.info("\nStep 2: Initializing project")

try:
    project = radio_project(
        PROJECT_DIR,
        DB_NAME,
        DETECTION_COUNT,
        DURATION,
        tag_data,
        receiver_data,
        nodes_data
    )
    logger.info(f"✓ Project initialized: {project.db}")
    logger.info(f"  Study tags: {len(project.study_tags)}")
    logger.info(f"  Test tags: {len(project.test_tags)}")
    logger.info(f"  Beacon tags: {len(project.beacon_tags)}")
    
except Exception as e:
    logger.error(f"Failed to initialize project: {e}")
    sys.exit(1)

# =============================================================================
# STEP 3: PROCESS RECEIVERS (BATCH MODE)
# =============================================================================

logger.info("\nStep 3: Processing receivers in batch mode")
logger.info(f"Processing {len(RECEIVERS)} receivers")

successful_receivers = []
failed_receivers = []

for i, rec_config in enumerate(RECEIVERS, 1):
    rec_id = rec_config['rec_id']
    rec_type = rec_config['rec_type']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing receiver {i}/{len(RECEIVERS)}: {rec_id}")
    logger.info(f"{'='*60}")
    
    try:
        # IMPORT
        logger.info(f"  Importing data for {rec_id}...")
        ant_dict = {'A0': rec_id}  # Modify if multiple antennas
        
        project.telem_data_import(
            rec_id,
            rec_type,
            project.training_dir,
            project.db,
            rec_config['scan_time'],
            rec_config['channels'],
            ant_dict,
            ka_format=False
        )
        logger.info(f"  ✓ Import complete")
        
        # TRAIN
        logger.info(f"  Training classifier for {rec_id}...")
        fishes = project.get_fish(rec_id=rec_id)
        logger.info(f"    Found {len(fishes)} fish")
        
        for j, fish in enumerate(fishes, 1):
            logger.info(f"    Training {j}/{len(fishes)}: {fish}")
            project.train(fish, rec_id)
        
        project.training_summary(rec_type, site=[rec_id])
        logger.info(f"  ✓ Training complete")
        
        # CLASSIFY
        logger.info(f"  Classifying detections for {rec_id}...")
        project.reclassify(
            project=project,
            rec_id=rec_id,
            threshold_ratio=THRESHOLD_RATIO,
            likelihood_model=LIKELIHOOD_MODEL,
            rec_type=rec_type,
            rec_list=None
        )
        logger.info(f"  ✓ Classification complete")
        
        successful_receivers.append(rec_id)
        logger.info(f"✓ {rec_id} processed successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to process {rec_id}: {e}")
        failed_receivers.append({'rec_id': rec_id, 'error': str(e)})
        continue

# =============================================================================
# STEP 4: CALCULATE BOUTS
# =============================================================================

logger.info("\n" + "="*60)
logger.info("Step 4: Calculating bouts")
logger.info("="*60)

for rec_id in successful_receivers:
    try:
        logger.info(f"  Calculating bouts for {rec_id}...")
        bout = pymast.bout(project, rec_id, lag_window=2, time_limit=21600)
        threshold = bout.fit_processes()
        bout.presence(threshold)
        logger.info(f"  ✓ Bouts calculated for {rec_id}")
    except Exception as e:
        logger.warning(f"  Could not calculate bouts for {rec_id}: {e}")
        continue

# =============================================================================
# STEP 5: REMOVE OVERLAP (IF APPLICABLE)
# =============================================================================

logger.info("\n" + "="*60)
logger.info("Step 5: Removing overlap")
logger.info("="*60)

# Define your overlap relationships here
# Example: edges = [('R01', 'R02'), ('R01', 'R03')]
edges = []  # UPDATE THIS based on your receiver layout

if edges:
    try:
        nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
        logger.info(f"  Removing overlap for nodes: {', '.join(nodes)}")
        
        overlap_remover = pymast.overlap_reduction(nodes, edges, project)
        overlap_remover.nested_doll()
        
        logger.info(f"  ✓ Overlap removed")
    except Exception as e:
        logger.error(f"  Failed to remove overlap: {e}")
else:
    logger.info("  No overlap edges defined, skipping")

# =============================================================================
# STEP 6: CREATE FINAL RECAPTURES TABLE
# =============================================================================

logger.info("\n" + "="*60)
logger.info("Step 6: Creating final recaptures table")
logger.info("="*60)

try:
    project.make_recaptures_table(export=True)
    output_file = os.path.join(project.output_dir, 'recaptures.csv')
    logger.info(f"  ✓ Recaptures table created")
    logger.info(f"  ✓ Exported to: {output_file}")
except Exception as e:
    logger.error(f"  Failed to create recaptures table: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

logger.info("\n" + "="*80)
logger.info("ANALYSIS COMPLETE")
logger.info("="*80)

logger.info(f"\nSuccessfully processed: {len(successful_receivers)}/{len(RECEIVERS)} receivers")
if successful_receivers:
    logger.info(f"  {', '.join(successful_receivers)}")

if failed_receivers:
    logger.warning(f"\nFailed to process: {len(failed_receivers)} receivers")
    for failure in failed_receivers:
        logger.warning(f"  {failure['rec_id']}: {failure['error']}")

logger.info(f"\nOutput files:")
logger.info(f"  Database: {project.db}")
logger.info(f"  Figures: {project.figures_dir}")
logger.info(f"  Log file: {log_file}")

logger.info("\nNext steps:")
logger.info("  1. Review fish histories (pymast.fish_history)")
logger.info("  2. Check training summary figures")
logger.info("  3. Format data for statistical analysis")
logger.info("  4. Generate final reports")

logger.info("\n" + "="*80)
logger.info("Log saved to: " + log_file)
logger.info("="*80)
