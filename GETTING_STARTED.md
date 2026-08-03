# Getting Started with PyMAST

Welcome to PyMAST (Movement Analysis Software for Telemetry)! This guide will help you set up and run your first radio telemetry analysis.

## 📋 Prerequisites

- Python 3.9 or higher
- Basic understanding of radio telemetry concepts
- Receiver data files (Lotek, Orion, or VR2 format)
- Tag deployment metadata
- Receiver location metadata

## ⚡ Quick Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install pymast
```

### Option 2: Install from Source
```bash
# Clone or download this repository
cd mast
pip install -e .
```

## 🎯 Your First Analysis (5 Steps)

### Step 1: Organize Your Data

Create a project directory with this structure:
```
my_telemetry_project/
├── raw_data/              # Receiver detection files
│   ├── receiver_001.csv
│   ├── receiver_002.csv
│   └── ...
├── tblMasterTag.csv       # Tag metadata
├── tblMasterReceiver.csv  # Receiver metadata
├── tblNodes.csv           # Spatial nodes (optional)
└── tblLines.csv           # Node connections (optional)
```

### Step 2: Prepare Metadata Files

**tblMasterTag.csv** - One row per tag:
```csv
freq_code,pulse_rate,tag_type,rel_date,cap_loc,rel_loc
166.380 7,5.0,Study,2024-05-15,Tailrace,Tailrace
166.380 12,5.0,Study,2024-05-15,Tailrace,Tailrace
166.380 19,5.0,Study,2024-05-16,Tailrace,Tailrace
```

**tblMasterReceiver.csv** - One row per receiver:
```csv
rec_id,rec_type,node,latitude,longitude,first_deploy,last_deploy
REC001,srx1200,N01,42.3601,-71.0589,2024-05-01,2024-09-01
REC002,srx1200,N02,42.3612,-71.0601,2024-05-01,2024-09-01
REC003,ares,N03,42.3623,-71.0615,2024-05-01,2024-09-01
```

### Step 3: Initialize Your Project

```python
import os
import pandas as pd
from pymast.radio_project import radio_project

# Set up paths
project_dir = r"C:\path\to\my_telemetry_project"
db_name = "my_study.h5"

# Load metadata
tags = pd.read_csv(os.path.join(project_dir, 'tblMasterTag.csv'))
receivers = pd.read_csv(os.path.join(project_dir, 'tblMasterReceiver.csv'))

# Create project
proj = radio_project(
    project_dir=project_dir,
    db_name=db_name,
    detection_count=5,
    duration=1,
    tag_data=tags,
    receiver_data=receivers
)
```

### Step 4: Import Receiver Data

```python
import glob

# Import all files from a receiver directory
proj.telem_data_import(
    rec_id='REC001',
    rec_type='srx1200',  # Options: srx600, srx800, srx1200, orion, ares, vr2, pit
    file_dir=os.path.join(project_dir, 'raw_data'),
    db_dir=proj.db,
    scan_time=2.5,
    channels=1
)
```

### Step 5: Process and Analyze

```python
from pymast.overlap_removal import bout, overlap_reduction

# Generate recaptures (link detections to locations)
proj.make_recaptures_table()

# Detect bouts (continuous presence periods)
for rec_id in receivers['rec_id']:
    bout_obj = bout(
        radio_project=proj,
        rec_id=rec_id,
        eps_multiplier=5  # 5x pulse rate for temporal clustering
    )
    bout_obj.presence()

# Resolve overlapping detections
overlap_obj = overlap_reduction(
    nodes=list(receivers['rec_id']),
    edges=[],  # Add receiver connections if known
    radio_project=proj
)
overlap_obj.unsupervised_removal(method='posterior')

# Visualize results
overlap_obj.visualize_overlaps()
```

## 📊 Export for Statistical Analysis

```python
from pymast.formatter import time_to_event

# Create time-to-event format for survival analysis
tte = time_to_event(
    receiver_to_state={'REC001': 1},
    project=proj
)

# Build transition records and export
tte.data_prep(proj)
tte.master_state_table.to_csv(os.path.join(project_dir, 'time_to_event.csv'), index=False)
```

## 🔍 Quality Control

```python
from pymast.fish_history import fish_history

# Visualize individual fish tracks
fh = fish_history(
    projectDB=proj.db,
    filtered=True,
    overlapping=False
)

# View specific fish
fh.change_fish('166.380 7')
fh.plot_3d_trajectory()
```

## 📚 Next Steps

- **Classification**: Train Naive Bayes classifier to remove false positives
- **Advanced Filtering**: Configure adjacency filter parameters
- **Statistical Models**: Export CJS or LRDR formats
- **Custom Analysis**: Access HDF5 database directly with pandas

## 🆘 Getting Help

- **Documentation**: See `/docs` folder for detailed guides
- **Architecture**: Read `ARCHITECTURE.md` for system overview
- **API Reference**: Use `help(pymast.module_name)` in Python
- **Examples**: Check `/examples` folder for complete workflows
- **Issues**: Report bugs on GitHub issues page

## 💡 Pro Tips

1. **Start Small**: Test workflow on one receiver before batch processing
2. **Check Metadata**: Verify freq_codes match exactly between tags and detections
3. **Visualize Early**: Use overlap visualizations to understand your data
4. **Document Settings**: Keep notes on eps_multiplier and filter parameters
5. **Backup Database**: HDF5 files are your processed data - back them up!

## 🎓 Common Workflows

### Workflow 1: Basic Detection Processing
Import → Recaptures → Bouts → Export

### Workflow 2: With Classification
Import → Train Classifier → Test Detections → Recaptures → Bouts → Export

### Workflow 3: With Overlap Resolution
Import → Recaptures → Bouts → Overlap Removal → Adjacency Filter → Export

### Workflow 4: Complete QA/QC
Import → Classify → Recaptures → Bouts → Overlap Removal → Fish History QC → Adjacency Filter → Export

---

**Ready to dive deeper?** Check out:
- `ARCHITECTURE.md` - System design and database structure
- `docs/TUTORIAL.md` - Step-by-step complete analysis
- `docs/API_REFERENCE.md` - Detailed function documentation
- `CHANGELOG.md` - Recent updates and features
