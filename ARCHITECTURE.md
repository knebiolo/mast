# PyMAST Architecture Documentation

**Multi-state Analysis of Stream Telemetry (MAST)**  
Python package for processing and analyzing radio telemetry data from fish tracking studies.

---

## Overview

PyMAST is a complete pipeline for converting raw radio telemetry detections into clean movement data suitable for statistical analysis. It handles data import, classification, bout detection, overlap removal, and formatting for survival/movement models.

---

## Core Workflow

```
┌─────────────────┐
│  Raw Data       │  CSV/Excel files from receivers
│  (Lotek, etc)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Import      │  parsers.py → HDF5 database
│  (parsers)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Train       │  naive_bayes.py → train classifier
│  (naive_bayes)  │  Uses known-location fish
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Classify    │  naive_bayes.py → identify true detections
│  (naive_bayes)  │  Separates fish from noise
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Bouts       │  overlap_removal.bout → DBSCAN clustering
│  (overlap)      │  Groups detections into presence periods
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Overlaps    │  overlap_removal.overlap_reduction
│  (overlap)      │  Resolves multi-receiver conflicts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Recaptures  │  radio_project.make_recaptures_table
│  (project)      │  Aggregates detections per bout
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. Format      │  formatter.time_to_event
│  (formatter)    │  Prepares for statistical models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  8. Analysis    │  Export to R/MARK/etc
│  (external)     │  CJS, TTE, LRDR models
└─────────────────┘
```

---

## Module Descriptions

### 1. **radio_project.py**
**Purpose**: Central project management class  
**Key Class**: `radio_project`

**What it does**:
- Creates and manages HDF5 database (`project.h5`)
- Stores tag metadata (fish IDs, species, release info)
- Stores receiver metadata (locations, node relationships)
- Manages project directory structure
- Coordinates data flow between modules

**Key Methods**:
- `__init__(project_dir, db_name)` - Initialize project
- `import_data(receiver_list)` - Import raw CSV data
- `make_recaptures_table()` - Aggregate classified bouts into recaptures

**HDF5 Tables**:
- `/raw_data` - Imported detections
- `/classified` - After Naive Bayes classification
- `/presence` - After bout detection (DBSCAN)
- `/overlapping` - After overlap removal
- `/recaptures` - Final aggregated data

---

### 2. **parsers.py**
**Purpose**: Import raw data from receiver files  
**Key Functions**: Format-specific parsers

**What it does**:
- Reads CSV/Excel files from Lotek, Orion, SRX receivers
- Standardizes column names and formats
- Writes to `/raw_data` table in HDF5
- Handles different file formats automatically

**Supported Formats**:
- Lotek .DTA files
- Lotek .CSV exports
- SRX text files
- Custom fixed-width formats

---

### 3. **naive_bayes.py**
**Purpose**: Classify true detections vs noise  
**Key Classes**: `train`, `test`

**What it does**:
- **Training**: Learns statistical patterns from known-location fish
  - Uses presence/absence periods
  - Calculates lag distributions, hit ratios, consecutive detection patterns
  - Builds Naive Bayes model
  
- **Testing**: Classifies all detections as true fish (test=1) or noise (test=0)
  - Outputs `posterior_T` (probability it's a true fish, 0-1)
  - Outputs `posterior_F` (probability it's noise, 0-1)
  - Threshold typically 0.5, but configurable

**Key Tables**:
- Input: `/raw_data`
- Output: `/classified` (adds test, posterior_T, posterior_F columns)

---

### 4. **overlap_removal.py**
**Purpose**: Handle temporal/spatial overlap conflicts  
**Key Classes**: `bout`, `overlap_reduction`

#### **Class: `bout`**
**What it does**:
- **DBSCAN temporal clustering**: Groups detections into bouts (presence periods)
- Configurable via `eps_multiplier` parameter:
  - `eps_multiplier=5`: ~40-50 sec window (default)
  - `eps_multiplier=10`: ~80-100 sec window (longer bouts)
  
**Algorithm**:
```python
eps = pulse_rate * eps_multiplier  # e.g., 8 sec * 5 = 40 sec window
DBSCAN clusters detections within eps seconds into same bout
```

**Output**: `/presence` table with bout_no assigned

**Methods**:
- `__init__(project, rec_id, eps_multiplier=5)` - Initialize for receiver
- `presence()` - Write bout results to database
- `visualize_bout_lengths()` - Diagnostic plots

#### **Class: `overlap_reduction`**
**What it does**:
- Identifies when fish detected at multiple receivers simultaneously
- Resolves conflicts using power/posterior comparison
- Marks ambiguous cases where we can't decide

**Algorithm** (method='power'):
```
For each temporal overlap:
  1. Power comparison (normalized 0-1 per receiver)
     - If |power_diff| > 0.2 → keep stronger, mark weaker as overlapping
  
  2. Posterior comparison (if power ambiguous)
     - If |posterior_diff| > 0.1 → keep higher confidence
  
  3. Keep both (if both ambiguous)
     - Mark both as ambiguous_overlap=1
```

**Output Flags**:
- `overlapping=1`: Detection removed (weaker signal or lower confidence)
- `ambiguous_overlap=1`: Can't decide, keep but flag as uncertain

**Spatial Filter** (NEW):
- After power/posterior resolution
- Finds bouts at different receivers with temporal overlap
- Keeps longer bout (more detections), marks shorter as overlapping
- Handles antenna bleed (back lobes detecting fish on wrong side)

**Methods**:
- `unsupervised(method='power', min_detections=1, bout_expansion=30)`
- `visualize_overlaps()` - Diagnostic plots

---

### 5. **formatter.py**
**Purpose**: Format data for statistical models  
**Key Classes**: `cjs_data_prep`, `lrdr_data_prep`, `time_to_event`

#### **Class: `time_to_event`**
**Most commonly used for competing risks / multi-state models**

**What it does**:
- Reads `/recaptures` table
- Filters out overlapping and ambiguous detections
- Filters out small bouts (< 3 detections)
- Maps receivers to states (locations in the model)
- Creates state transition records
- Applies adjacency filter (removes impossible movements)

**Adjacency Filter**:
- List of illegal (from_state, to_state) tuples
- Example: `[(9,1), (9,2)]` = can't go from tailrace to forebay
- Iteratively removes impossible transitions
- Preserves timing by carrying forward start times

**Output**: `master_state_table` with columns:
- `freq_code`: Fish ID
- `start_state`, `end_state`: Movement transition
- `time_0`, `time_1`: Entry/exit times (epoch seconds)
- `time_delta`: Duration in state
- `transition`: Tuple (start, end) for filtering

**Methods**:
- `__init__(receiver_to_state, project, ...)` - Load and filter data
- `data_prep(project, adjacency_filter=None)` - Build state transitions
- `summary()` - Print movement statistics

---

### 6. **predictors.py**
**Purpose**: Feature engineering for classification  
**Key Function**: `predictor_class()`

**What it does**:
- Calculates features from detection sequences:
  - `lag`: Time since last detection (seconds)
  - `cons_det`: Consecutive detections
  - `cons_length`: Duration of consecutive period
  - `hit_ratio`: Detections per expected (based on pulse rate)
  - `det_hist`: Rolling presence/absence pattern

**Used by**: `naive_bayes.py` for training/testing

---

### 7. **validation.py**
**Purpose**: Input data validation  
**Key Functions**: `validate_tag_data()`, `validate_receiver_data()`

**What it does**:
- Checks CSV files have required columns
- Validates data types and ranges
- Raises `ValidationError` with helpful messages

---

### 8. **fish_history.py**
**Purpose**: Individual fish trajectory analysis (deprecated)  
**Status**: Legacy code, rarely used

---

## Data Flow Details

### HDF5 Database Structure

```
project.h5
├── /raw_data           # Imported detections (100K-10M+ rows)
│   ├── freq_code       # Fish ID (e.g., "12345.67")
│   ├── rec_id          # Receiver ID (e.g., "R03")
│   ├── epoch           # Unix timestamp (float)
│   ├── time_stamp      # Datetime
│   ├── power           # Signal strength (dB)
│   └── ...
│
├── /classified         # After Naive Bayes
│   ├── (all raw_data columns)
│   ├── test            # 1=true fish, 0=noise
│   ├── posterior_T     # P(true fish | data)
│   └── posterior_F     # P(noise | data)
│
├── /presence           # After bout detection
│   ├── freq_code
│   ├── rec_id
│   ├── epoch
│   ├── bout_no         # Bout ID (0, 1, 2, ...)
│   ├── power
│   └── class           # "study"
│
├── /overlapping        # After overlap removal
│   ├── freq_code
│   ├── rec_id
│   ├── epoch
│   ├── time_stamp
│   ├── overlapping     # 1=removed, 0=keep
│   ├── ambiguous_overlap  # 1=uncertain
│   ├── power           # (optional)
│   └── posterior_T     # (optional)
│
└── /recaptures         # Aggregated by bout
    ├── freq_code
    ├── rec_id
    ├── epoch           # Min epoch in bout
    ├── time_stamp
    ├── lag             # Median lag
    ├── bout_no
    ├── overlapping     # Max (0 or 1)
    ├── ambiguous_overlap  # Max (0 or 1)
    ├── hit_ratio       # Median
    └── ...
```

---

## Configuration Parameters

### Bout Detection
- `eps_multiplier` (default 5): DBSCAN window multiplier
  - **Lower** (3-5): Stricter, shorter bouts
  - **Higher** (10-20): Looser, longer bouts (use if fish movements slow)

### Overlap Removal
- `method` (default 'power'): Power-based vs posterior-based resolution
- `min_detections` (default 1): Minimum bout size to consider
- `bout_expansion` (default 30): Expand bout windows by ±N seconds

### TTE Formatting
- `min_bout_size` (default 3): Filter out bouts with < N detections
- `adjacency_filter`: List of illegal transitions
- `temporal_overlap_threshold` (default 0.5): Bout conflict threshold (50% overlap)

---

## Common Pitfalls

### 1. **Memory Issues**
- Large datasets (>1M detections per receiver)
- **Solution**: Uses Dask for lazy loading, caching with `.copy()` to prevent contamination

### 2. **Antenna Bleed**
- Directional antennas detect fish from "wrong" direction
- **Solution**: Bout spatial filter + adjacency filter

### 3. **Index Mismatches**
- DataFrames losing indices between operations
- **Solution**: Cache data once, use `.copy()` per operation

### 4. **Overlapping=1 but Still in Data**
- TTE should filter overlapping=0 AND ambiguous_overlap=0
- **Check**: Recaptures table includes both columns

---

## Performance

**Typical dataset**:
- 50 fish, 15 receivers, 6 months
- ~5-10M raw detections
- Processing time: 10-30 minutes (depending on hardware)

**Bottlenecks**:
- Naive Bayes training (looping over training fish)
- Overlap removal (O(n²) bout comparisons per fish)

**Optimizations**:
- HDF5 WHERE clauses for selective reading
- Caching recap data per node
- Parallel processing in overlap removal (via ProcessPoolExecutor)

---

## Output for Statistical Models

### Time-to-Event (Multi-state)
```python
tte.master_state_table.to_csv('tte.csv')
```
Columns: freq_code, start_state, end_state, time_0, time_1, time_delta

Import to R:
```r
library(msm)
data <- read.csv("tte.csv")
model <- msm(state ~ time, subject=freq_code, data=data, ...)
```

### CJS (Cormack-Jolly-Seber)
```python
cjs.input_file('model_name', output_dir)
```
Generates `.inp` file for Program MARK

---

## Troubleshooting

### "No overlaps found" but expect many
- Check if `/presence` table exists and has `bout_no`
- Verify `bout_expansion` parameter (try 60-120 seconds)
- Check receiver IDs match between presence and classified tables

### "Impossible movements" in TTE output
- Enable adjacency filter with illegal transitions
- Check bout spatial filter is running (should see output)
- Verify receiver-to-state mapping makes biological sense

### Poor classification (too many noise detections kept)
- Retrain with more training fish
- Adjust posterior threshold (default 0.5, try 0.6-0.7)
- Check training fish actually present/absent as labeled

---

## Version History

- **v1.0.0**: Production release
  - Naive Bayes classification
  - DBSCAN bout detection
  - Power/posterior overlap resolution
  - Ambiguous overlap tracking
  - Bout spatial filter
  - Automated recaptures building
  - Visualization tools

---

## Contact & Support

For questions about implementation or study design, contact Kleinschmidt Associates.
