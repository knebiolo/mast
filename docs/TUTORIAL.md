# MAST Tutorial: Complete Workflow

This tutorial walks you through a complete MAST analysis from start to finish.

## Prerequisites

- MAST installed (`pip install pymast`)
- Input CSV files prepared (tblMasterTag.csv, tblMasterReceiver.csv, tblNodes.csv)
- Raw receiver data files in Data/Training_Files/

## 1. Setup and Import

```python
import os
import pandas as pd
from pymast.radio_project import radio_project
import pymast

# Set your project directory
project_dir = r"C:\path\to\your\project"
db_name = 'tutorial_project'

# Load input data
tag_data = pd.read_csv(os.path.join(project_dir, 'tblMasterTag.csv'))
receiver_data = pd.read_csv(os.path.join(project_dir, 'tblMasterReceiver.csv'))
nodes_data = pd.read_csv(os.path.join(project_dir, 'tblNodes.csv'))

# Create project
project = radio_project(
    project_dir,
    db_name,
    detection_count=5,  # Detection history window
    duration=1,         # Noise ratio window (seconds)
    tag_data,
    receiver_data,
    nodes_data
)
```

## 2. Import Raw Data

Import data for each receiver in your project:

```python
# Configure receiver parameters
rec_id = 'R01'
rec_type = 'srx800'  # Receiver type
scan_time = 1.0      # Channel scan time
channels = 1         # Number of channels
antenna_to_rec_dict = {'A0': rec_id}

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
```

**Tip:** Repeat this cell for each receiver, updating `rec_id` and `rec_type`.

## 3. Train the Classifier

Training data comes from beacon tags (known true detections) and noise (known false detections):

```python
# Get all fish detected at this receiver
fishes = project.get_fish(rec_id='R01')

# Train on each fish
for fish in fishes:
    project.train(fish, 'R01')

# View training summary
project.training_summary('srx800', site=['R01'])
```

This generates histograms showing predictor distributions for true vs false detections.

## 4. Classify Detections

Use the trained Naive Bayes classifier to identify true detections:

```python
# Set classification parameters
threshold_ratio = 1.0  # MAP hypothesis (adjust 0.8-1.2 for sensitivity)
likelihood_model = ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff']

# Classify
project.reclassify(
    project=project,
    rec_id='R01',
    threshold_ratio=threshold_ratio,
    likelihood_model=likelihood_model,
    rec_type='srx800',
    rec_list=None
)
```

**Understanding threshold_ratio:**
- 1.0 = Standard (Maximum A Posteriori)
- >1.0 = More conservative (fewer false positives, may lose true detections)
- <1.0 = More liberal (captures more true detections, may include false positives)

## 5. Calculate Bouts (Optional but Recommended)

Bouts identify discrete visits to a receiver location:

```python
# Create bout object
node = 'R01'
bout_obj = pymast.bout(
    project,
    node,
    lag_window=2,      # Binning window for lags
    time_limit=21600   # Max lag time (6 hours)
)

# Fit the bout model (interactive - follow prompts)
threshold = bout_obj.fit_processes()

# Calculate presences
bout_obj.presence(threshold)
```

The bout model fits a broken-stick regression to lag frequencies. The knots identify:
- Process 1: Continuous presence
- Process 2: Edge behavior (milling)
- Process 3: True departures

## 6. Remove Overlap

If receivers have overlapping detection zones, remove redundant detections:

```python
# Define parent:child relationships
# Parent = larger detection zone, Child = smaller/nested zone
edges = [
    ('R01', 'R02'),  # R01 overlaps R02
    ('R01', 'R03'),  # R01 overlaps R03
]
nodes = ['R01', 'R02', 'R03']

# Apply nested doll algorithm
overlap = pymast.overlap_reduction(nodes, edges, project)
overlap.nested_doll()
```

**Alternative: Unsupervised method**
```python
# For pairwise overlap without defined hierarchy
for i in project.receivers.index:
    for j in project.receivers.index:
        if i != j:
            edges = [(i, j)]
            nodes = [i, j]
            doll = pymast.overlap_reduction(nodes, edges, project)
            doll.unsupervised_removal()
```

## 7. Create Recaptures Table

Compile all processed data into final recaptures table:

```python
project.make_recaptures_table(export=True)
```

This creates:
- HDF5 table at `/recaptures` in the database
- CSV export at `Output/recaptures.csv`

## 8. Visualize Fish Histories

Check data quality by viewing individual fish movements:

```python
# Create fish history object
history = pymast.fish_history(
    project.db,
    filtered=True,      # Use classified data
    overlapping=False   # Exclude overlapping detections
)

# Plot a specific fish
history.fish_plot('164.123 45')  # Use your freq_code
```

This creates a 3D plot showing:
- X, Y: Spatial coordinates (from nodes table)
- Z: Time since first detection

Look for:
- Logical movement patterns
- Remaining false positives (impossible movements)
- Overlap issues (rapid back-and-forth)

## 9. Format for Statistical Analysis

### Cormack-Jolly-Seber (CJS)

```python
# Define receiver to recapture occasion mapping
receiver_to_recap = {
    'R00': 'R00',  # Release
    'R01': 'R01',  # First recapture occasion
    'R02': 'R02',  # Second recapture occasion
    'R03': 'R03',  # Third recapture occasion
}

# Create CJS data
cjs = pymast.formatter.cjs_data_prep(
    receiver_to_recap,
    project,
    initial_recap_release=False  # True if starting from first recap
)

# Export input file for MARK
cjs.input_file('my_model', project.output_dir)
```

### Competing Risks (Time-to-Event)

```python
# Define state mapping
node_to_state = {
    'R01': 1,  # Staging
    'R02': 2,  # Upstream passage
    'R03': 3,  # Downstream passage
}

# Create competing risks data
# (Implementation in formatter module)
```

## Troubleshooting

### If classifications look wrong:
1. Check training summary histograms - are true/false well-separated?
2. Adjust `threshold_ratio` (try 0.9 or 1.1)
3. Try different predictor combinations in `likelihood_model`
4. Add more beacon tag data for training

### If bout fitting fails:
1. Try different `time_limit` values
2. Check if fish are actually present long enough for bouts
3. May need manual threshold instead of fitted

### If overlap removal doesn't work:
1. Verify parent:child relationships are correct
2. Check that bouts were calculated first
3. Try unsupervised method for complex overlaps

## Next Steps

- Cross-validate training data (`project.cross_validate()`)
- Iterate classification with different parameters
- Generate project reports and figures
- Export data for external analysis (R, MARK, etc.)

## Best Practices

1. **Always train with both beacon tags and noise detections**
2. **Visualize fish histories before statistical analysis**
3. **Document your threshold_ratio and likelihood_model choices**
4. **Keep original data - use `undo_*` functions to iterate**
5. **Export recaptures table when satisfied with results**

## Getting Help

- Check API reference: `docs/API_REFERENCE.md`
- Review example scripts: `examples/`
- Open an issue on GitHub
- Read the published methodology papers

---

**Congratulations!** You've completed a full MAST workflow. Your data is now ready for movement analysis and survival modeling.
