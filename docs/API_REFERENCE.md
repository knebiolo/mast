# MAST API Reference

## Core Classes

### `radio_project`

Main project class for managing telemetry data analysis.

#### Initialization

```python
project = radio_project(
    project_dir,      # str: Project directory path
    db_name,          # str: Database name (without .h5 extension)
    detection_count,  # int: Number of detections for detection history
    duration,         # float: Duration window for noise ratio (seconds)
    tag_data,         # DataFrame: Master tag table
    receiver_data,    # DataFrame: Master receiver table
    nodes_data        # DataFrame: Network nodes table (optional)
)
```

#### Key Methods

##### Data Import
```python
project.telem_data_import(
    rec_id,              # str: Receiver ID
    rec_type,            # str: Receiver type ('srx600', 'srx800', 'srx1200', 'orion', 'ares', 'VR2')
    file_dir,            # str: Directory containing raw data files
    db_dir,              # str: Database directory
    scan_time=1,         # float: Channel scan time (seconds)
    channels=1,          # int: Number of channels
    ant_to_rec_dict=None,# dict: Antenna to receiver mapping
    ka_format=False      # bool: Use Kleinschmidt format
)
```

##### Training
```python
project.train(
    freq_code,  # str: Frequency code to train
    rec_id      # str: Receiver ID
)

project.training_summary(
    rec_type,   # str: Receiver type
    site=None   # list: List of receiver IDs (optional)
)
```

##### Classification
```python
project.reclassify(
    project,           # radio_project: The project instance
    rec_id,            # str: Receiver ID
    threshold_ratio,   # float: Classification threshold (1.0 = MAP)
    likelihood_model,  # list: Predictors to use
    rec_type,          # str: Receiver type
    rec_list=None      # list: Specific receivers (optional)
)
```

##### Data Management
```python
project.make_recaptures_table(export=False)  # Compile final recaptures
project.undo_import(rec_id)                  # Remove imported data
project.undo_training(rec_id, freq_code=None)# Remove training data
project.undo_classification(rec_id)          # Remove classifications
project.undo_bouts(node)                     # Remove bout calculations
project.undo_overlap()                       # Remove overlap reductions
project.undo_recaptures()                    # Remove recaptures table
```

---

### `bout`

Calculate bouts and presences at receivers.

```python
bout_obj = pymast.bout(
    radio_project,  # radio_project: Project instance
    node,           # str: Node ID
    lag_window,     # int: Lag binning window (seconds)
    time_limit      # int: Maximum lag time to consider (seconds)
)

# Fit the bout model
threshold = bout_obj.fit_processes()

# Calculate presences
bout_obj.presence(threshold)  # float or fitted threshold
```

---

### `overlap_reduction`

Remove overlapping detections between receivers.

```python
overlap = pymast.overlap_reduction(
    nodes,   # list: Node IDs
    edges,   # list: Tuples of (parent, child) relationships
    project  # radio_project: Project instance
)

# Apply nested doll algorithm
overlap.nested_doll()

# Or use unsupervised method
overlap.unsupervised_removal()
```

---

### `fish_history`

Visualize fish movement through the telemetry network.

```python
history = pymast.fish_history(
    projectDB,         # str: Path to project database
    filtered=True,     # bool: Use classified data
    overlapping=False, # bool: Include overlapping detections
    rec_list=None,     # list: Specific receivers (optional)
    filter_date=None   # datetime: Filter by date (optional)
)

# Plot a specific fish
history.fish_plot(freq_code)  # str: Frequency code
```

---

## Data Formatters

### Cormack-Jolly-Seber (CJS)

```python
cjs = pymast.formatter.cjs_data_prep(
    receiver_to_recap,        # dict: Receiver to recapture occasion mapping
    project,                  # radio_project: Project instance
    input_type='query',       # str: Input type
    species=None,             # str: Filter by species
    rel_loc=None,             # str: Filter by release location
    cap_loc=None,             # str: Filter by capture location
    initial_recap_release=False  # bool: Use first recap as release
)

# Generate input file
cjs.input_file(model_name, output_dir)
```

### Live Recapture Dead Recovery (LRDR)

```python
lrdr = pymast.formatter.lrdr_data_prep(
    receiver_to_recap,        # dict: Receiver to recapture mapping
    mobile_to_recap,          # dict: Mobile tracking to recapture mapping
    dbDir,                    # str: Database directory
    input_type='query',       # str: Input type
    rel_loc=None,             # str: Release location filter
    cap_loc=None,             # str: Capture location filter
    initial_recap_release=False,  # bool: Use first recap as release
    time_limit=None           # float: Time limit filter
)
```

---

## Input Data Requirements

### Master Tag Table (tblMasterTag.csv)

| Field       | Type     | Required | Description                          |
|-------------|----------|----------|--------------------------------------|
| freq_code   | str      | Yes      | Unique frequency-code combination    |
| pit_id      | int      | No       | PIT tag ID if applicable             |
| pulse_rate  | float    | Yes      | Seconds between pulses               |
| mort_rate   | float    | No       | Mortality pulse rate                 |
| cap_loc     | str      | Yes      | Capture location                     |
| rel_loc     | str      | Yes      | Release location                     |
| tag_type    | str      | Yes      | 'study', 'BEACON', or 'TEST'         |
| length      | int      | No       | Fish length (mm)                     |
| sex         | str      | No       | 'M' or 'F'                           |
| rel_date    | datetime | Yes      | Release date and time                |

### Master Receiver Table (tblMasterReceiver.csv)

| Field     | Type | Required | Description                          |
|-----------|------|----------|--------------------------------------|
| name      | str  | No       | Common name of location              |
| rec_type  | str  | Yes      | Receiver type                        |
| rec_id    | str  | Yes      | Unique receiver ID                   |
| node      | str  | Yes      | Associated network node              |

### Network Nodes Table (tblNodes.csv)

| Field  | Type | Required | Description                     |
|--------|------|----------|---------------------------------|
| node   | str  | Yes      | Unique node ID                  |
| reach  | str  | No       | Common name of reach            |
| X      | int  | Yes      | X coordinate (arbitrary)        |
| Y      | int  | Yes      | Y coordinate (arbitrary)        |

---

## Predictors for Classification

Available predictors for the Naive Bayes classifier:

- `hit_ratio`: Ratio of detected to expected pulses
- `cons_length`: Maximum consecutive detection length
- `noise_ratio`: Ratio of miscoded to total detections
- `power`: Signal power
- `lag_diff`: Second-order difference in detection lags

**Recommended starting model:**
```python
likelihood_model = ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff']
```

---

## Supported Receiver Types

- `srx600`: Lotek SRX600
- `srx800`: Lotek SRX800
- `srx1200`: Lotek SRX1200
- `orion`: Sigma Eight Orion
- `ares`: Advanced Telemetry Systems ARES
- `VR2`: Vemco VR2 (acoustic)

---

## Common Workflows

### Basic Workflow
1. Initialize project
2. Import raw data
3. Train classifier
4. Classify detections
5. Create recaptures table

### Advanced Workflow
1-5. (Same as basic)
6. Calculate bouts
7. Remove overlap
8. Visualize fish histories
9. Format for statistical analysis

---

For more examples, see the `examples/` directory.
