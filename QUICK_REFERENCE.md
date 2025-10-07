# MAST Quick Reference Card

## Installation
```bash
pip install git+https://github.com/knebiolo/mast.git
```

## Basic Workflow

### 1. Initialize Project
```python
import os, pandas as pd
from pymast.radio_project import radio_project
import pymast

project = radio_project(
    project_dir, db_name, 
    detection_count=5, duration=1,
    tag_data, receiver_data, nodes_data
)
```

### 2. Import Data
```python
project.telem_data_import(
    'R01', 'srx800', 
    project.training_dir, project.db,
    scan_time=1.0, channels=1, 
    ant_to_rec_dict={'A0': 'R01'}
)
```

### 3. Train Classifier
```python
fishes = project.get_fish(rec_id='R01')
for fish in fishes:
    project.train(fish, 'R01')
project.training_summary('srx800', site=['R01'])
```

### 4. Classify
```python
project.reclassify(
    project, 'R01', 1.0,
    ['hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'],
    'srx800'
)
```

### 5. Calculate Bouts
```python
bout = pymast.bout(project, 'R01', 2, 21600)
threshold = bout.fit_processes()
bout.presence(threshold)
```

### 6. Remove Overlap
```python
edges = [('R01', 'R02')]
overlap = pymast.overlap_reduction(['R01','R02'], edges, project)
overlap.nested_doll()
```

### 7. Create Recaptures
```python
project.make_recaptures_table(export=True)
```

## Common Commands

### Undo Operations
```python
project.undo_import('R01')
project.undo_training('R01')
project.undo_classification('R01')
project.undo_bouts('R01')
project.undo_overlap()
project.undo_recaptures()
```

### Fish History
```python
history = pymast.fish_history(project.db)
history.fish_plot('164.123 45')
```

### Statistical Formatting
```python
# CJS
cjs = pymast.formatter.cjs_data_prep(receiver_to_recap, project)
cjs.input_file('model_name', project.output_dir)
```

## Receiver Types
- `srx600`, `srx800`, `srx1200` (Lotek)
- `orion` (Sigma Eight)
- `ares` (ATS)
- `VR2` (Vemco acoustic)

## Input Files
- `tblMasterTag.csv` - freq_code, pulse_rate, tag_type, rel_date...
- `tblMasterReceiver.csv` - rec_id, rec_type, node
- `tblNodes.csv` - node, X, Y

## Predictors
- `hit_ratio` - Detection consistency
- `cons_length` - Consecutive hits
- `noise_ratio` - Miscoded detections
- `power` - Signal strength
- `lag_diff` - Timing regularity

## Threshold Ratio
- `1.0` = Standard (MAP)
- `>1.0` = Conservative
- `<1.0` = Liberal

## Troubleshooting
- Import error → Check Python >=3.9, `pip install pymast`
- Weird classifications → Check training histograms, adjust threshold
- Bout fitting fails → Try different time_limit or manual threshold
- Memory errors → Process one receiver at a time

## Quick Links
- Tutorial: `docs/TUTORIAL.md`
- API Reference: `docs/API_REFERENCE.md`
- Installation: `docs/INSTALLATION.md`
- FAQ: `docs/FAQ.md`
- Examples: `examples/`

## Getting Help
1. Check FAQ
2. Search GitHub issues
3. Open new issue with error details
4. Include: Python version, MAST version, minimal example

---
**MAST v1.0** | MIT License | github.com/knebiolo/mast
