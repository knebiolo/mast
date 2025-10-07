# Updated Overlap Removal Strategy - Handle All Tag Types

## Key Insight: Different Data Sources Need Different Approaches

### Radio Telemetry (Lotek, Sigma Eight, ATS, Sigma Six)
- **Data source**: `/classified` table in HDF5
- **Has columns**: `posterior_T`, `posterior_F`, `test`, `iter`
- **Classification**: Naive Bayes for false positive removal
- **Overlap strategy**: Use `posterior_T` (BEST - multi-factor, normalized)
- **Fallback**: Signal power comparison or nested_doll

### Acoustic (Vemco VR2/VR2W, VR2Tx) 
- **Data source**: `/recaptures` table (no classification step)
- **Has columns**: `power` (maybe), detection counts
- **Classification**: None (Vemco has built-in detection algorithm)
- **Overlap strategy**: 
  - If power available: normalized power comparison (with caution)
  - Otherwise: nested_doll only
- **Caveat**: Different receiver models may have different power scales

### PIT Tags (Biomark, Oregon RFID)
- **Data source**: `/recaptures` or `/raw_data` table
- **Has columns**: timestamp, antenna, typically NO power
- **Classification**: None (binary detection)
- **Overlap strategy**: nested_doll ONLY
- **Note**: Usually single antenna per location, less overlap issue

---

## Revised Implementation

### 1. Detect Tag/Data Type Automatically

```python
def __init__(self, nodes, edges, radio_project):
    """Initialize and detect data type."""
    self.db = radio_project.db
    self.project = radio_project
    self.nodes = nodes
    self.edges = edges
    self.G = nx.DiGraph()
    self.G.add_edges_from(edges)
    
    # Detect what kind of data we're working with
    self.data_type = self._detect_data_type()
    logger.info(f"Detected data type: {self.data_type}")
    
    # Load appropriate data
    self.node_pres_dict = {}
    self.node_recap_dict = {}
    
    for node in nodes:
        if self.data_type == 'radio_classified':
            # Radio telemetry with classification
            pres_data = pd.read_hdf(self.db, 'presence', where=f"rec_id == {node}")
            recap_data = pd.read_hdf(self.db, 'classified', where=f"rec_id == {node}")
            recap_data = recap_data[recap_data['iter'] == recap_data['iter'].max()]
            recap_data = recap_data[recap_data['test'] == 1]
            
        elif self.data_type == 'radio_recaptures':
            # Radio telemetry without classification (shouldn't happen, but handle it)
            pres_data = pd.read_hdf(self.db, 'presence', where=f"rec_id == {node}")
            recap_data = pd.read_hdf(self.db, 'recaptures', where=f"rec_id == {node}")
            
        elif self.data_type == 'acoustic':
            # Acoustic telemetry
            # No presence table, use recaptures directly
            recap_data = pd.read_hdf(self.db, 'recaptures', where=f"rec_id == {node}")
            # Create pseudo-presence from recaptures for nested_doll
            pres_data = self._create_pseudo_presence(recap_data)
            
        elif self.data_type == 'pit':
            # PIT tags
            recap_data = pd.read_hdf(self.db, 'recaptures', where=f"rec_id == {node}")
            pres_data = self._create_pseudo_presence(recap_data)
        
        # Process and store
        if self.data_type in ['radio_classified', 'radio_recaptures']:
            # Group presence by bouts
            summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
                min_epoch=('epoch', 'min'),
                max_epoch=('epoch', 'max'),
                median_power=('power', 'median')
            ).reset_index()
        else:
            # Use the pseudo-presence
            summarized_data = pres_data
        
        self.node_pres_dict[node] = summarized_data
        self.node_recap_dict[node] = recap_data
        logger.info(f"Loaded data for node {node}")


def _detect_data_type(self):
    """
    Detect what kind of telemetry data we're working with.
    
    Returns
    -------
    str : One of 'radio_classified', 'radio_recaptures', 'acoustic', 'pit'
    """
    with pd.HDFStore(self.db, 'r') as store:
        keys = store.keys()
        
        # Check for classified data (radio with Naive Bayes)
        if '/classified' in keys:
            # Verify it has posterior columns
            sample = pd.read_hdf(self.db, 'classified', stop=1)
            if 'posterior_T' in sample.columns:
                return 'radio_classified'
            else:
                logger.warning("classified table exists but no posterior_T column")
                return 'radio_recaptures'
        
        # Check for recaptures (could be radio, acoustic, or PIT)
        if '/recaptures' in keys:
            sample = pd.read_hdf(self.db, 'recaptures', stop=10)
            
            # Check for columns that indicate data type
            has_posterior = 'posterior_T' in sample.columns
            has_likelihood = 'likelihood_T' in sample.columns
            has_power = 'power' in sample.columns
            has_test = 'test' in sample.columns
            
            if has_posterior or has_likelihood:
                return 'radio_classified'
            elif has_test:  # Radio but not classified yet
                return 'radio_recaptures'
            elif has_power and sample['power'].notna().any():
                # Has power values - likely acoustic
                return 'acoustic'
            else:
                # No power, no classification - likely PIT
                return 'pit'
        
        # Check for raw_data only (early stage)
        if '/raw_data' in keys:
            logger.warning("Only raw_data found - run classification or make recaptures first")
            return 'radio_recaptures'
        
        raise ValueError("No suitable data tables found in HDF5 database")


def _create_pseudo_presence(self, recap_data, time_window=300):
    """
    Create pseudo-presence bouts from recaptures for acoustic/PIT data.
    
    Parameters
    ----------
    recap_data : pd.DataFrame
        Recapture data with at least freq_code, epoch, rec_id
    time_window : int, default=300
        Seconds to group detections into pseudo-bouts
    
    Returns
    -------
    pd.DataFrame
        Summarized data with min_epoch, max_epoch per bout
    """
    if recap_data.empty:
        return pd.DataFrame(columns=['freq_code', 'bout_no', 'rec_id', 
                                     'min_epoch', 'max_epoch'])
    
    # Sort by time
    recap_data = recap_data.sort_values(['freq_code', 'epoch'])
    
    # Calculate time gaps
    recap_data['time_gap'] = recap_data.groupby('freq_code')['epoch'].diff()
    
    # New bout when gap > time_window
    recap_data['new_bout'] = (recap_data['time_gap'] > time_window) | recap_data['time_gap'].isna()
    recap_data['bout_no'] = recap_data.groupby('freq_code')['new_bout'].cumsum()
    
    # Summarize by bout
    summarized = recap_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
        min_epoch=('epoch', 'min'),
        max_epoch=('epoch', 'max')
    ).reset_index()
    
    # Add median_power if available
    if 'power' in recap_data.columns:
        power_summary = recap_data.groupby(['freq_code', 'bout_no', 'rec_id'])['power'].median()
        summarized['median_power'] = summarized.merge(
            power_summary, on=['freq_code', 'bout_no', 'rec_id'], how='left'
        )['power']
    
    return summarized
```

### 2. Update unsupervised_removal to Handle All Types

```python
def unsupervised_removal(self, confidence_threshold=0.1, method='auto'):
    """
    Identify overlapping detections using best available method for data type.
    
    Parameters
    ----------
    confidence_threshold : float, default=0.1
        Minimum difference in confidence metric to classify as overlap.
        For posteriors: difference in mean posterior_T
        For power: difference in normalized power
    method : str, default='auto'
        Method to use:
        - 'auto': Automatically select based on data type (recommended)
        - 'posterior': Use posterior_T (requires radio telemetry with classification)
        - 'power': Use normalized power comparison
        - 'nested_doll': Simple interval overlap (most conservative)
    
    Returns
    -------
    None
        Writes results to HDF5 'overlapping' table
    
    Raises
    ------
    ValueError
        If method='posterior' but posteriors not available
    
    Notes
    -----
    **Automatic Method Selection:**
    - Radio telemetry (classified): Uses posterior_T (best - multi-factor)
    - Radio telemetry (not classified): Uses nested_doll (warns user to classify)
    - Acoustic telemetry: Uses power if available, else nested_doll
    - PIT tags: Uses nested_doll only (no power data)
    
    **Radio Telemetry (Lotek, Sigma Eight, ATS):**
    - Best method: posterior_T from Naive Bayes classifier
    - Accounts for: hit_ratio, noise, power, lag, consecutive detections
    - Handles different receiver types/gains/calibrations
    
    **Acoustic Telemetry (Vemco VR2):**
    - Power comparison if available (use with caution - different models have different scales)
    - Recommend nested_doll for mixed Vemco models
    
    **PIT Tags (Biomark, Oregon RFID):**
    - Only nested_doll available (binary detections, no power)
    
    Examples
    --------
    >>> # Radio telemetry - automatic (will use posteriors)
    >>> overlap_obj.unsupervised_removal()
    >>> 
    >>> # Force nested_doll (conservative)
    >>> overlap_obj.unsupervised_removal(method='nested_doll')
    >>> 
    >>> # Acoustic with power comparison
    >>> overlap_obj.unsupervised_removal(method='power', confidence_threshold=0.15)
    
    See Also
    --------
    nested_doll : Simple interval-based detection (works for all data types)
    """
    logger.info("Starting unsupervised overlap removal")
    logger.info(f"  Data type: {self.data_type}")
    logger.info(f"  Method: {method}")
    
    # Determine which method to use
    if method == 'auto':
        if self.data_type == 'radio_classified':
            method = 'posterior'
            logger.info("  Auto-selected method: posterior_T (recommended for radio)")
        elif self.data_type == 'radio_recaptures':
            logger.warning("  Radio data found but not classified - using nested_doll")
            logger.warning("  Recommendation: Run classification first for better results")
            method = 'nested_doll'
        elif self.data_type == 'acoustic':
            # Check if power is available
            sample_node = self.nodes[0]
            if 'power' in self.node_recap_dict[sample_node].columns:
                method = 'power'
                logger.info("  Auto-selected method: power comparison (acoustic)")
                logger.warning("  Note: Power comparison may be unreliable across different Vemco models")
            else:
                method = 'nested_doll'
                logger.info("  Auto-selected method: nested_doll (no power available)")
        elif self.data_type == 'pit':
            method = 'nested_doll'
            logger.info("  Auto-selected method: nested_doll (PIT tags have no power data)")
    
    # Validate method is possible
    if method == 'posterior':
        sample_node = self.nodes[0]
        if 'posterior_T' not in self.node_recap_dict[sample_node].columns:
            raise ValueError(
                f"Method 'posterior' requires posterior_T column. "
                f"Data type is '{self.data_type}'. "
                f"Run classification first or use method='auto' or method='nested_doll'"
            )
    
    # Route to appropriate implementation
    if method == 'nested_doll':
        logger.info("Using nested_doll algorithm (interval-based)")
        return self.nested_doll()
    elif method == 'posterior':
        logger.info("Using posterior-based algorithm (confidence-weighted)")
        return self._unsupervised_removal_posterior(confidence_threshold)
    elif method == 'power':
        logger.info("Using power-based algorithm")
        logger.warning("⚠ Power comparison assumes similar receiver types/calibrations")
        return self._unsupervised_removal_power(confidence_threshold)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'posterior', 'power', or 'nested_doll'")


def _unsupervised_removal_posterior(self, confidence_threshold=0.1):
    """
    Posterior-based overlap removal (radio telemetry only).
    
    This is the RECOMMENDED method for radio telemetry as it uses the
    multi-factor Naive Bayes posterior probabilities which account for
    different receiver types, gains, and calibrations.
    """
    # (Implementation from PROPOSED_UNSUPERVISED_REMOVAL.py)
    # Uses posterior_T comparison
    pass  # See full implementation in proposal


def _unsupervised_removal_power(self, confidence_threshold=0.15):
    """
    Power-based overlap removal (acoustic telemetry).
    
    WARNING: Use with caution. Different Vemco models may have different
    power scales. Best for single receiver model deployments.
    """
    # Similar to posterior method but uses normalized power
    # Higher default threshold (0.15) because power is less reliable
    pass


def _warn_about_data_type(self):
    """Provide guidance based on data type."""
    if self.data_type == 'radio_recaptures':
        logger.warning("=" * 80)
        logger.warning("RECOMMENDATION: Run Naive Bayes classification first")
        logger.warning("This will provide posterior_T for better overlap detection")
        logger.warning("Currently using nested_doll as fallback (more conservative)")
        logger.warning("=" * 80)
    
    elif self.data_type == 'acoustic':
        logger.info("=" * 80)
        logger.info("ACOUSTIC DATA DETECTED")
        logger.info("Power comparison will be used if available")
        logger.info("Note: Assumes similar receiver models and calibrations")
        logger.info("For mixed Vemco models, consider using nested_doll only")
        logger.info("=" * 80)
    
    elif self.data_type == 'pit':
        logger.info("=" * 80)
        logger.info("PIT TAG DATA DETECTED")
        logger.info("Only nested_doll method available (no power/classification)")
        logger.info("This is normal for PIT tags - bouts define presence")
        logger.info("=" * 80)
```

---

## Updated Method Selection Logic

```python
# Decision tree for method selection

IF data_type == 'radio_classified':
    → Use posterior_T (BEST - handles all radio receiver types)
    
ELIF data_type == 'radio_recaptures':
    → Warn user to run classification
    → Fall back to nested_doll
    
ELIF data_type == 'acoustic':
    IF power column exists:
        → Use power comparison (with warning about mixed models)
    ELSE:
        → Use nested_doll only
        
ELIF data_type == 'pit':
    → Use nested_doll only (no other option)
```

---

## User Interface Examples

### Example 1: Radio Telemetry (Ideal Case)
```python
# After classification
project.classify(...)

# Overlap removal automatically uses posteriors
overlap_obj = overlap_reduction(nodes, edges, project)
overlap_obj.unsupervised_removal()  # Uses posterior_T automatically
# Output: "Using posterior-based algorithm (confidence-weighted)"
```

### Example 2: Radio Telemetry (No Classification Yet)
```python
# No classification done
overlap_obj = overlap_reduction(nodes, edges, project)
overlap_obj.unsupervised_removal()
# Output: "WARNING: Radio data found but not classified - using nested_doll"
# Output: "RECOMMENDATION: Run classification first for better results"
```

### Example 3: Acoustic Telemetry
```python
# Acoustic data (Vemco)
overlap_obj = overlap_reduction(nodes, edges, project)
overlap_obj.unsupervised_removal()
# Output: "ACOUSTIC DATA DETECTED"
# Output: "Power comparison will be used if available"
# Output: "⚠ Power comparison assumes similar receiver types/calibrations"
```

### Example 4: PIT Tags
```python
# PIT tag data
overlap_obj = overlap_reduction(nodes, edges, project)
overlap_obj.unsupervised_removal()
# Output: "PIT TAG DATA DETECTED"
# Output: "Only nested_doll method available (no power/classification)"
```

### Example 5: Force Method
```python
# Always use conservative method regardless of data type
overlap_obj.unsupervised_removal(method='nested_doll')

# Force power comparison (acoustic)
overlap_obj.unsupervised_removal(method='power', confidence_threshold=0.2)
```

---

## Summary of Changes

### What This Fixes:
✅ **Correctly identifies data type** (radio/acoustic/PIT)
✅ **Only uses posteriors for radio telemetry** where they exist
✅ **Provides appropriate fallbacks** for each data type
✅ **Clear warnings and guidance** for users
✅ **Works with all tag types** your software supports

### Method Priority by Data Type:

**Radio Telemetry:**
1. 🥇 posterior_T (if classified) - BEST
2. 🥈 nested_doll (if not classified) - CONSERVATIVE

**Acoustic Telemetry:**
1. 🥇 nested_doll - SAFEST (recommended for mixed models)
2. 🥈 Power comparison - OK for single model deployments

**PIT Tags:**
1. 🥇 nested_doll - ONLY OPTION

---

**Ready to implement this revised approach?** This should handle all the telemetry types in your project correctly! 🎯
