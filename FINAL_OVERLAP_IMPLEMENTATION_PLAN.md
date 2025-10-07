# Final Overlap Removal Implementation Plan
## Focus: Radio Telemetry with Posterior Probabilities

**Philosophy**: Perfect the radio telemetry use case (95% of projects), provide simple fallback for others.

---

## Implementation Strategy

### Radio Telemetry (PRIMARY - Make it Perfect)

**Data Flow:**
```
Raw Data → Train Classifier → Classify → Make Recaptures → Overlap Removal
                                          ↓
                                    posterior_T, posterior_F
                                          ↓
                                    Use for overlap detection!
```

**Key Insight**: 
- `recaptures` table has `posterior_T` and `posterior_F` columns
- These are receiver-specific (learned during training)
- Perfect for comparing confidence across different receivers/manufacturers
- Already accounts for: power, hit_ratio, noise, lag, cons_length

**Algorithm**:
```python
For each parent→child receiver pair:
    For each fish:
        For each overlapping bout:
            parent_confidence = mean(parent.posterior_T)
            child_confidence = mean(child.posterior_T)
            
            if parent_confidence > child_confidence + threshold:
                # Fish near parent → remove child detections
                child.overlapping = 1
            elif child_confidence > parent_confidence + threshold:
                # Fish near child → remove parent detections  
                parent.overlapping = 1
            else:
                # Similar confidence → keep both (conservative)
                pass
```

### Acoustic/PIT (SECONDARY - Simple Fallback)

**Recommendation in docs**: "For acoustic or PIT tags, use `nested_doll()` method (conservative interval-based detection)"

**Code handling**: 
- Check if `posterior_T` exists
- If not: error with clear message pointing to `nested_doll()`
- Don't try to auto-detect or handle - keep it simple

---

## Clean Implementation

### 1. Update `__init__()` - Minimal Changes

```python
def __init__(self, nodes, edges, radio_project):
    """
    Initialize overlap reduction for radio telemetry data.
    
    Parameters
    ----------
    nodes : list
        Receiver IDs in the network
    edges : list of tuples
        Parent→child relationships, e.g., [('R01', 'R02'), ('R02', 'R03')]
    radio_project : radio_project object
        Project containing classified telemetry data
    
    Notes
    -----
    Requires that classification has been run (data in /classified or /recaptures).
    For acoustic or PIT tag data, use nested_doll() method instead of 
    unsupervised_removal().
    """
    logger.info("Initializing overlap_reduction")
    self.db = radio_project.db
    self.project = radio_project
    self.nodes = nodes
    self.edges = edges
    self.G = nx.DiGraph()
    self.G.add_edges_from(edges)
    
    self.node_pres_dict = {}
    self.node_recap_dict = {}
    
    logger.info(f"  Loading data for {len(nodes)} nodes")
    
    for node in tqdm(nodes, desc="Loading nodes", unit="node"):
        # Load presence data (from bout detection)
        pres_data = pd.read_hdf(
            self.db, 'presence',
            columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'],
            where=f"rec_id == '{node}'"
        )
        
        # Load classified data or recaptures
        try:
            # Try classified first (most common after classification)
            recap_data = pd.read_hdf(
                self.db, 'classified',
                columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 
                         'iter', 'test', 'posterior_T', 'posterior_F'],
                where=f"rec_id == '{node}'"
            )
            recap_data = recap_data[recap_data['iter'] == recap_data['iter'].max()]
            recap_data = recap_data[recap_data['test'] == 1]
            
        except (KeyError, ValueError):
            # Fall back to recaptures table
            logger.debug(f"  {node}: No classified data, trying recaptures")
            recap_data = pd.read_hdf(
                self.db, 'recaptures',
                where=f"rec_id == '{node}'"
            )
        
        # Summarize presence by bout
        summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
            min_epoch=('epoch', 'min'),
            max_epoch=('epoch', 'max'),
            median_power=('power', 'median')
        ).reset_index()
        
        self.node_pres_dict[node] = summarized_data
        self.node_recap_dict[node] = recap_data
        
        logger.debug(f"  {node}: {len(pres_data)} presence records, {len(recap_data)} detections")
    
    logger.info(f"✓ Data loaded for {len(nodes)} nodes")
```

### 2. Implement Clean `unsupervised_removal()`

```python
def unsupervised_removal(self, confidence_threshold=0.1):
    """
    Remove overlapping detections using Naive Bayes posterior probabilities.
    
    For radio telemetry data, this method uses the posterior_T (probability of 
    true detection) from the Naive Bayes classifier to determine which receiver
    has higher confidence for overlapping detections. The receiver with lower
    confidence has its detections marked as overlapping (redundant).
    
    Parameters
    ----------
    confidence_threshold : float, default=0.1
        Minimum difference in mean posterior_T required to classify as overlap.
        - Larger values (e.g., 0.2): More conservative, keeps more detections
        - Smaller values (e.g., 0.05): Less conservative, marks more overlaps
        Recommended range: 0.05 - 0.20
    
    Returns
    -------
    None
        Results are written to HDF5 database at /overlapping key
    
    Raises
    ------
    ValueError
        If posterior_T column not found (classification not run)
    
    Notes
    -----
    **Requirements:**
    - Must run classification first (train + classify methods)
    - Data must have posterior_T and posterior_F columns
    - For acoustic/PIT data, use nested_doll() instead
    
    **Algorithm:**
    1. For each parent→child edge in network
    2. For each fish detected at both receivers
    3. For each overlapping bout (temporal overlap)
    4. Compare mean(parent.posterior_T) vs mean(child.posterior_T)
    5. Mark lower-confidence receiver as overlapping
    
    **Why posterior_T is better than power:**
    - Accounts for different manufacturers (Lotek, Sigma Eight, ATS)
    - Accounts for different gain settings and calibrations
    - Multi-factor decision (power + hit_ratio + noise + lag + cons_length)
    - Scale-invariant (always 0-1, comparable across receivers)
    - Already validated by classification process
    
    Examples
    --------
    >>> # Standard usage (after classification)
    >>> project = pymast.radio_project(...)
    >>> project.train(...)
    >>> project.classify(...)
    >>> 
    >>> # Define network
    >>> nodes = ['R01', 'R02', 'R03']
    >>> edges = [('R01', 'R02'), ('R02', 'R03')]
    >>> 
    >>> # Remove overlaps
    >>> overlap_obj = overlap_reduction(nodes, edges, project)
    >>> overlap_obj.unsupervised_removal(confidence_threshold=0.1)
    >>> 
    >>> # More conservative (keeps more detections)
    >>> overlap_obj.unsupervised_removal(confidence_threshold=0.2)
    
    See Also
    --------
    nested_doll : Simple interval-based overlap detection (more conservative)
    """
    import logging
    from tqdm import tqdm
    import gc
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting unsupervised overlap removal")
    logger.info(f"  Method: Posterior probability comparison")
    logger.info(f"  Confidence threshold: {confidence_threshold}")
    logger.info(f"  Processing {len(self.edges)} parent→child edges")
    
    # Validate that posterior_T exists
    sample_node = self.nodes[0]
    sample_data = self.node_recap_dict[sample_node]
    
    if 'posterior_T' not in sample_data.columns:
        raise ValueError(
            "posterior_T column not found in data. This method requires classified "
            "radio telemetry data.\n\n"
            "Solutions:\n"
            "1. Run classification first: project.train(...) then project.classify(...)\n"
            "2. For acoustic or PIT data, use nested_doll() method instead:\n"
            "   overlap_obj.nested_doll()\n\n"
            f"Available columns: {list(sample_data.columns)}"
        )
    
    overlaps_processed = 0
    detections_marked = 0
    decisions = {'remove_parent': 0, 'remove_child': 0, 'keep_both': 0}
    
    # Process each parent→child edge
    for edge_idx, (parent, child) in enumerate(tqdm(self.edges, desc="Processing edges", unit="edge")):
        logger.debug(f"Edge {edge_idx+1}/{len(self.edges)}: {parent} → {child}")
        
        parent_bouts = self.node_pres_dict[parent]
        parent_dat = self.node_recap_dict[parent].copy()
        child_dat = self.node_recap_dict[child].copy()
        
        if parent_bouts.empty or parent_dat.empty or child_dat.empty:
            logger.debug(f"  Skipping {parent}→{child}: empty data")
            continue
        
        # Initialize overlapping columns
        if 'overlapping' not in parent_dat.columns:
            parent_dat['overlapping'] = np.float32(0)
        if 'overlapping' not in child_dat.columns:
            child_dat['overlapping'] = np.float32(0)
        
        # Get unique fish in parent bouts
        fishes = parent_bouts['freq_code'].unique()
        logger.debug(f"  Processing {len(fishes)} fish")
        
        for fish_id in fishes:
            # Filter for this fish
            parent_fish_bouts = parent_bouts[parent_bouts['freq_code'] == fish_id]
            parent_fish_dat = parent_dat[parent_dat['freq_code'] == fish_id]
            child_fish_dat = child_dat[child_dat['freq_code'] == fish_id]
            
            if parent_fish_dat.empty or child_fish_dat.empty:
                continue
            
            # Process each bout
            for _, bout_row in parent_fish_bouts.iterrows():
                min_epoch = bout_row['min_epoch']
                max_epoch = bout_row['max_epoch']
                
                # Get detections in this bout window
                parent_mask = (
                    (parent_fish_dat['epoch'] >= min_epoch) & 
                    (parent_fish_dat['epoch'] <= max_epoch)
                )
                child_mask = (
                    (child_fish_dat['epoch'] >= min_epoch) & 
                    (child_fish_dat['epoch'] <= max_epoch)
                )
                
                parent_in_bout = parent_fish_dat[parent_mask]
                child_in_bout = child_fish_dat[child_mask]
                
                # Check for overlap
                if len(parent_in_bout) == 0 or len(child_in_bout) == 0:
                    continue  # No temporal overlap
                
                overlaps_processed += 1
                
                # Calculate confidence (mean posterior_T)
                parent_confidence = parent_in_bout['posterior_T'].mean()
                child_confidence = child_in_bout['posterior_T'].mean()
                confidence_diff = parent_confidence - child_confidence
                
                # Make classification decision
                if confidence_diff > confidence_threshold:
                    # Parent has significantly higher confidence
                    # → Fish is near parent → Keep parent, remove child
                    parent_dat.loc[parent_mask, 'overlapping'] = np.float32(0)
                    child_dat.loc[child_mask, 'overlapping'] = np.float32(1)
                    decision = 'remove_child'
                    decisions['remove_child'] += 1
                    detections_marked += len(child_in_bout)
                    
                elif confidence_diff < -confidence_threshold:
                    # Child has significantly higher confidence
                    # → Fish is near child → Keep child, remove parent
                    parent_dat.loc[parent_mask, 'overlapping'] = np.float32(1)
                    child_dat.loc[child_mask, 'overlapping'] = np.float32(0)
                    decision = 'remove_parent'
                    decisions['remove_parent'] += 1
                    detections_marked += len(parent_in_bout)
                    
                else:
                    # Confidence difference too small → Ambiguous → Keep both
                    decision = 'keep_both'
                    decisions['keep_both'] += 1
                
                logger.debug(
                    f"    {fish_id} [{min_epoch:.0f}-{max_epoch:.0f}]: "
                    f"parent_conf={parent_confidence:.3f}, "
                    f"child_conf={child_confidence:.3f}, "
                    f"diff={confidence_diff:.3f} → {decision}"
                )
        
        # Write results for both parent and child
        logger.debug(f"  Writing results for {parent}")
        self.write_results_to_hdf5(parent_dat)
        
        logger.debug(f"  Writing results for {child}")
        self.write_results_to_hdf5(child_dat)
        
        # Memory management
        del parent_bouts, parent_dat, child_dat
        gc.collect()
    
    # Summary
    logger.info("✓ Unsupervised overlap removal complete")
    logger.info(f"  Overlapping bouts processed: {overlaps_processed}")
    logger.info(f"  Detections marked as overlapping: {detections_marked}")
    logger.info(f"  Decision breakdown:")
    logger.info(f"    - Removed parent detections: {decisions['remove_parent']} bouts")
    logger.info(f"    - Removed child detections: {decisions['remove_child']} bouts")
    logger.info(f"    - Kept both (ambiguous): {decisions['keep_both']} bouts")
    
    if decisions['keep_both'] > overlaps_processed * 0.5:
        logger.warning(
            f"  ⚠ Over 50% of overlaps kept both receivers (ambiguous)"
            f"\n  Consider increasing confidence_threshold for more conservative results"
            f"\n  Or decreasing threshold if you want more aggressive overlap removal"
        )
```

### 3. Update `nested_doll()` with Logging Only

```python
def nested_doll(self):
    """
    Identify and mark overlapping detections using simple interval comparison.
    
    This method uses a conservative approach: any detection at a parent receiver
    that temporally overlaps with a bout at a child receiver is marked as 
    overlapping (redundant).
    
    Returns
    -------
    None
        Results written to HDF5 database at /overlapping key
    
    Notes
    -----
    **When to use:**
    - Acoustic telemetry (Vemco) or PIT tag data (no posteriors available)
    - Radio telemetry before classification
    - Want most conservative overlap detection
    - Quick first-pass analysis
    
    **Advantages:**
    - Simple and deterministic
    - Fast (vectorized operations)
    - No statistical assumptions
    - Works for any telemetry data type
    
    **Disadvantages:**
    - Does not consider signal strength or confidence
    - May mark more overlaps than necessary (conservative)
    - Binary decision (overlap or not)
    
    **For radio telemetry, unsupervised_removal() is recommended** as it uses
    posterior probabilities to make more informed decisions.
    
    Examples
    --------
    >>> # Use for acoustic/PIT data
    >>> overlap_obj = overlap_reduction(nodes, edges, project)
    >>> overlap_obj.nested_doll()
    >>> 
    >>> # Or for quick conservative radio analysis
    >>> overlap_obj.nested_doll()  # Before classification
    
    See Also
    --------
    unsupervised_removal : Confidence-based method for radio telemetry (less conservative)
    """
    logger.info("Starting nested_doll overlap detection")
    logger.info("  Method: Interval-based (conservative)")
    
    overlaps_found = False
    overlap_count = 0
    
    for node in tqdm(self.node_recap_dict, desc="Processing nodes", unit="node"):
        fishes = self.node_recap_dict[node].freq_code.unique()
        logger.debug(f"  {node}: {len(fishes)} fish")
        
        for fish_id in fishes:
            children = list(self.G.successors(node))
            fish_dat = self.node_recap_dict[node][
                self.node_recap_dict[node].freq_code == fish_id
            ].copy()
            fish_dat['overlapping'] = np.float32(0)
            
            if len(children) > 0:
                for child in children:
                    child_dat = self.node_pres_dict[child][
                        self.node_pres_dict[child].freq_code == fish_id
                    ]
                    
                    if len(child_dat) > 0:
                        min_epochs = child_dat.min_epoch.values
                        max_epochs = child_dat.max_epoch.values
                        fish_epochs = fish_dat.epoch.values
                        
                        # Vectorized overlap check
                        overlaps = np.any(
                            (min_epochs[:, None] <= fish_epochs) & 
                            (max_epochs[:, None] > fish_epochs),
                            axis=0
                        )
                        
                        overlap_indices = np.where(overlaps)[0]
                        if overlap_indices.size > 0:
                            overlaps_found = True
                            overlap_count += overlap_indices.size
                            fish_dat.loc[overlaps, 'overlapping'] = np.float32(1)
                            logger.debug(f"    {fish_id}: {len(overlap_indices)} overlaps with {child}")
            
            # Write results
            fish_dat = fish_dat[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping']]
            self.write_results_to_hdf5(fish_dat)
    
    if overlaps_found:
        logger.info(f"✓ Nested doll complete")
        logger.info(f"  Total overlaps found: {overlap_count}")
    else:
        logger.info("✓ Nested doll complete - no overlaps found")
```

### 4. Keep `write_results_to_hdf5()` As-Is (Maybe Add Logging)

```python
def write_results_to_hdf5(self, df):
    """
    Write overlapping detection results to HDF5 database.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'overlapping' column
    """
    try:
        df = df.astype({
            'freq_code': 'object',
            'epoch': 'int32',
            'rec_id': 'object',
            'overlapping': 'int32',
        })
        
        with pd.HDFStore(self.project.db, mode='a') as store:
            store.append(
                key='overlapping',
                value=df[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping']],
                format='table',
                data_columns=True,
                min_itemsize={'freq_code': 20, 'rec_id': 20}
            )
        logger.debug(f"    Wrote {len(df)} detections to /overlapping")
        
    except Exception as e:
        logger.error(f"Error writing to HDF5: {e}")
        raise
```

---

## Files to Modify

### Primary File: `pymast/overlap_removal.py`

**Keep untouched** (lines 1-450):
- ✅ `bout` class entirely
- ✅ All piecewise exponential decay code

**Refactor** (lines 690-832):
- 🔧 `overlap_reduction.__init__()` - add logging
- 🔧 `unsupervised_removal()` - complete rewrite with posteriors
- 🔧 `nested_doll()` - add logging only
- 🔧 `write_results_to_hdf5()` - add logging only

**Delete** (lines 450-690, 832-1254):
- 🗑️ All commented-out code (~700 lines)

**Final size**: ~750 lines (from 1254)

---

## Documentation Updates

### Module Docstring

```python
"""
Overlap removal for radio telemetry data.

This module provides two approaches for identifying and removing redundant 
detections when multiple receivers detect the same tagged fish:

1. **unsupervised_removal** (Recommended for Radio Telemetry)
   Uses Naive Bayes posterior probabilities to determine which receiver has
   higher confidence. Accounts for different receiver types, gains, and 
   calibrations. Less conservative than nested_doll.

2. **nested_doll** (Conservative/Universal)
   Simple interval-based overlap detection. More conservative (marks more
   overlaps). Use for acoustic/PIT data, or quick radio analysis.

Classes
-------
bout
    Detects temporal bouts of presence using piecewise exponential decay
overlap_reduction
    Manages receiver network and implements overlap removal algorithms

Examples
--------
Radio Telemetry Workflow::

    import pymast
    from pymast.overlap_removal import bout, overlap_reduction
    
    # 1. Setup project and classify
    project = pymast.radio_project(...)
    project.train(...)
    project.classify(...)
    
    # 2. Detect bouts
    bout_obj = bout(project, 'R01', lag_window=5, time_limit=300)
    bout_obj.fit_processes()
    bout_obj.presence(threshold=120)
    
    # 3. Remove overlaps (posterior-based, recommended)
    nodes = ['R01', 'R02', 'R03']
    edges = [('R01', 'R02'), ('R02', 'R03')]
    overlap_obj = overlap_reduction(nodes, edges, project)
    overlap_obj.unsupervised_removal(confidence_threshold=0.1)

Acoustic/PIT Workflow::

    # Use conservative method (no posteriors available)
    overlap_obj = overlap_reduction(nodes, edges, project)
    overlap_obj.nested_doll()

Notes
-----
**Radio Telemetry (Lotek, Sigma Eight, ATS):**
- Requires classification first (train + classify)
- Use unsupervised_removal() with posterior probabilities
- Handles mixed receiver types and calibrations

**Acoustic (Vemco) / PIT (Biomark):**
- Use nested_doll() method (more conservative)
- No classification or posterior probabilities available

See Also
--------
pymast.radio_project : Main project class for telemetry analysis
pymast.naive_bayes : Classification algorithms
"""
```

### README Addition

```markdown
## Overlap Removal

When multiple receivers detect the same fish, MAST can identify and remove redundant detections.

### Radio Telemetry (Recommended Method)

```python
# After classification
project.classify(...)

# Define receiver network (parent → child relationships)
nodes = ['R01', 'R02', 'R03']
edges = [('R01', 'R02'), ('R02', 'R03')]  # R01 upstream of R02, R02 upstream of R03

# Remove overlaps using posterior probabilities
overlap_obj = overlap_reduction(nodes, edges, project)
overlap_obj.unsupervised_removal(confidence_threshold=0.1)

# Results in project.db at /overlapping key
overlaps = pd.read_hdf(project.db, 'overlapping')
```

**How it works**: Compares Naive Bayes posterior probabilities between receivers. The receiver with higher confidence "wins". Accounts for different manufacturers, gains, and calibrations.

### Acoustic/PIT Tags (Conservative Method)

```python
# Use simple interval-based detection
overlap_obj.nested_doll()
```

**Note**: Acoustic and PIT data don't have classification, so we use the more conservative interval-based method.
```

---

## Testing Plan

### Test 1: Basic Posterior Comparison
```python
def test_posterior_comparison():
    """Test that higher posterior wins."""
    # Create mock data
    parent_data = pd.DataFrame({
        'freq_code': ['164.123'] * 5,
        'epoch': [100, 101, 102, 103, 104],
        'posterior_T': [0.85, 0.88, 0.90, 0.87, 0.89],  # High confidence
        'rec_id': 'R01'
    })
    
    child_data = pd.DataFrame({
        'freq_code': ['164.123'] * 5,
        'epoch': [100, 101, 102, 103, 104],
        'posterior_T': [0.35, 0.38, 0.40, 0.37, 0.39],  # Low confidence
        'rec_id': 'R02'
    })
    
    # Run algorithm
    result = process_overlap(parent_data, child_data, threshold=0.1)
    
    # Parent should be kept (overlapping=0), child removed (overlapping=1)
    assert (result['parent']['overlapping'] == 0).all()
    assert (result['child']['overlapping'] == 1).all()
```

### Test 2: No Posterior Available
```python
def test_missing_posterior():
    """Test error handling when posterior_T missing."""
    data_no_posterior = pd.DataFrame({
        'freq_code': ['164.123'],
        'epoch': [100],
        'power': [75.0],
        'rec_id': 'R01'
    })
    
    with pytest.raises(ValueError, match="posterior_T column not found"):
        overlap_obj.unsupervised_removal()
```

---

## Final Checklist

- [ ] Add logging import to top of file
- [ ] Add tqdm import  
- [ ] Replace `unsupervised_removal()` with posterior-based version
- [ ] Add logging to `__init__()`
- [ ] Add logging to `nested_doll()`
- [ ] Add logging to `write_results_to_hdf5()`
- [ ] Delete all commented code (lines 450-690, 832-1254)
- [ ] Update module docstring
- [ ] Add comprehensive method docstrings
- [ ] Test with real radio telemetry project
- [ ] Update README with examples
- [ ] Update CHANGELOG.md

---

**Ready to implement?** This is clean, focused, and solves your actual use case (radio telemetry) perfectly! 🎯
