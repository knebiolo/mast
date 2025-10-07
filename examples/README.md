# MAST Examples

This directory contains example scripts demonstrating MAST functionality.

## Quick Start Example

**File:** `quick_start_example.py`

A complete workflow from project initialization through final recaptures table. This is the best starting point for new users.

**What it demonstrates:**
- Project initialization
- Data import
- Classifier training
- Detection classification
- Bout calculation
- Overlap removal
- Recaptures table generation

**To use:**
1. Update `project_dir` to your project location
2. Ensure your input CSV files are ready
3. Update receiver parameters (`rec_id`, `rec_type`)
4. Run the script

## Additional Examples

### Data Import

Example showing how to import data from different receiver types:

```python
# Lotek SRX series
project.telem_data_import('R01', 'srx800', training_dir, db_dir, 1.0, 1, {'A0': 'R01'})

# Sigma Eight Orion
project.telem_data_import('R02', 'orion', training_dir, db_dir, 10.5, 2, {'A0': 'R02', 'A1': 'R02'})

# ATS ARES
project.telem_data_import('R03', 'ares', training_dir, db_dir, 1.0, 1, {'A0': 'R03'})
```

### Classification Tuning

Example of iterating classification with different parameters:

```python
# Conservative (fewer false positives)
project.reclassify(project, 'R01', 1.2, ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'], 'srx800')

# Standard
project.reclassify(project, 'R01', 1.0, ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'], 'srx800')

# Liberal (more detections)
project.reclassify(project, 'R01', 0.8, ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'], 'srx800')
```

### Batch Processing

Process multiple receivers:

```python
receivers = ['R01', 'R02', 'R03', 'R04']
rec_type = 'srx800'

for rec_id in receivers:
    print(f"Processing {rec_id}...")
    
    # Import
    project.telem_data_import(rec_id, rec_type, training_dir, db_dir, 1.0, 1, {'A0': rec_id})
    
    # Train
    fishes = project.get_fish(rec_id=rec_id)
    for fish in fishes:
        project.train(fish, rec_id)
    
    # Classify
    project.reclassify(project, rec_id, 1.0, ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'], rec_type)
    
    print(f"✓ {rec_id} complete")
```

### Complex Overlap Networks

Example of defining overlap in a complex telemetry network:

```python
# Define hierarchical relationships
# Format: (parent_with_larger_zone, child_with_smaller_zone)
edges = [
    # Main river overlaps all tributaries
    ('R_main', 'R_trib1'),
    ('R_main', 'R_trib2'),
    ('R_main', 'R_trib3'),
    
    # Tributary overlaps fishway entrance
    ('R_trib1', 'R_fishway_entrance'),
    
    # Fishway entrance overlaps internal fishway receiver
    ('R_fishway_entrance', 'R_fishway_internal'),
]

nodes = ['R_main', 'R_trib1', 'R_trib2', 'R_trib3', 'R_fishway_entrance', 'R_fishway_internal']

# Apply nested doll
overlap = pymast.overlap_reduction(nodes, edges, project)
overlap.nested_doll()
```

### Fish History Visualization

Visualize and check specific fish:

```python
history = pymast.fish_history(project.db, filtered=True, overlapping=False)

# Get all fish in project
fish_list = history.recaptures.freq_code.unique()

# Plot first 10
for fish in fish_list[:10]:
    print(f"Plotting {fish}...")
    history.fish_plot(fish)
```

### Statistical Formatting

#### CJS for Upstream Migration

```python
# Receivers in upstream order
receiver_to_recap = {
    'R_release': 'R00',      # Release location
    'R_tailrace': 'R01',     # Below dam
    'R_fishway': 'R02',      # Fishway entrance
    'R_forebay': 'R03',      # Above dam
    'R_upstream': 'R04',     # Far upstream
}

cjs = pymast.formatter.cjs_data_prep(receiver_to_recap, project)
cjs.input_file('upstream_migration', project.output_dir)
```

#### CJS for Downstream Migration

```python
# Start from first recapture (fish already upstream)
receiver_to_recap = {
    'R_upstream': 'R00',     # First detection (becomes "release")
    'R_forebay': 'R01',      # Approaching dam
    'R_turbine': 'R02',      # Turbine passage
    'R_spillway': 'R03',     # Spillway passage
    'R_downstream': 'R04',   # Below dam
}

cjs = pymast.formatter.cjs_data_prep(
    receiver_to_recap, 
    project,
    initial_recap_release=True  # Use first recap as release
)
cjs.input_file('downstream_migration', project.output_dir)
```

## Sample Data

Sample datasets (if available) are in `data/`:
- `lotek fwfs.xlsx`: Lotek fixed-width format specifications
- `srx1200_fwf.txt`: Example SRX1200 raw data file

## Configuration

See `config_template.yaml` in the root directory for a template configuration file that can define all project parameters in one place.

## Need Help?

- Check the tutorial: `docs/TUTORIAL.md`
- Review API documentation: `docs/API_REFERENCE.md`
- Look at the main README: `README.md`
- Open an issue on GitHub

## Contributing Examples

Have a useful example? Please contribute!
1. Create a new .py file in this directory
2. Add clear comments explaining what it does
3. Update this README with a description
4. Submit a pull request
