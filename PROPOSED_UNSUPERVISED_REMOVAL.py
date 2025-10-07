# Proposed refactored unsupervised_removal method
# This shows the structure - we'll implement this properly in the actual file

def unsupervised_removal(self, confidence_threshold=0.1, use_posteriors=True, 
                        fallback_to_power=False):
    """
    Identifies and removes overlapping detections using classifier confidence.
    
    Uses posterior_T probabilities from Naive Bayes classifier to determine which
    receiver has higher confidence for overlapping detections. The receiver with
    lower confidence has its detections marked as overlapping (redundant).
    
    Parameters
    ----------
    confidence_threshold : float, default=0.1
        Minimum difference in mean posterior_T required to classify overlap.
        Larger values make the method more conservative.
    use_posteriors : bool, default=True
        Use posterior_T from Naive Bayes (recommended for mixed receiver types).
        If False, falls back to normalized power comparison.
    fallback_to_power : bool, default=False
        If posterior_T not available, fall back to power comparison instead of error.
    
    Returns
    -------
    None
        Updates 'overlapping' column in parent and child data, writes to HDF5.
    
    Raises
    ------
    ValueError
        If use_posteriors=True but posterior_T column not found and fallback disabled.
    
    Notes
    -----
    Algorithm:
    1. For each parent→child edge in network graph
    2. For each fish detected at both receivers
    3. For each overlapping bout (temporal overlap)
    4. Compare mean(parent.posterior_T) vs mean(child.posterior_T)
    5. If difference > threshold: mark lower confidence receiver as overlapping
    6. If difference <= threshold: keep both (conservative)
    
    Advantages over power-based methods:
    - Handles heterogeneous receiver types (different manufacturers, gains)
    - Multi-factor decision (incorporates hit_ratio, noise, lag, etc.)
    - Already validated by classification process
    - Scale-invariant (probabilities always 0-1)
    
    Examples
    --------
    >>> overlap_obj = overlap_reduction(nodes, edges, project)
    >>> # Standard usage - requires prior classification
    >>> overlap_obj.unsupervised_removal(confidence_threshold=0.1)
    >>> 
    >>> # More conservative (larger threshold)
    >>> overlap_obj.unsupervised_removal(confidence_threshold=0.2)
    >>> 
    >>> # Use power if posteriors not available (not recommended)
    >>> overlap_obj.unsupervised_removal(use_posteriors=False)
    
    See Also
    --------
    nested_doll : Simpler interval-based overlap detection
    """
    import logging
    from tqdm import tqdm
    logger = logging.getLogger(__name__)
    
    logger.info("Starting unsupervised overlap removal")
    logger.info(f"  Using {'posterior probabilities' if use_posteriors else 'signal power'}")
    logger.info(f"  Confidence threshold: {confidence_threshold}")
    logger.info(f"  Processing {len(self.edges)} parent→child edges")
    
    # Validate that posterior_T exists if requested
    if use_posteriors:
        sample_node = self.nodes[0]
        sample_data = self.node_recap_dict[sample_node]
        if 'posterior_T' not in sample_data.columns:
            if fallback_to_power:
                logger.warning("posterior_T not found, falling back to power comparison")
                use_posteriors = False
            else:
                raise ValueError(
                    "posterior_T column not found in classified data. "
                    "Run classification first, or set use_posteriors=False, "
                    "or enable fallback_to_power=True"
                )
    
    overlaps_processed = 0
    total_detections_marked = 0
    
    # Process each parent→child edge with progress bar
    for i, (parent, child) in enumerate(tqdm(self.edges, desc="Processing edges", unit="edge")):
        logger.debug(f"Processing edge {i+1}/{len(self.edges)}: {parent} → {child}")
        
        parent_bouts = self.node_pres_dict[parent]
        parent_dat = self.node_recap_dict[parent]
        child_dat = self.node_recap_dict[child]
        
        if parent_bouts.empty or parent_dat.empty or child_dat.empty:
            logger.debug(f"  Skipping {parent}→{child}: empty data")
            continue
        
        # Iterate over each unique fish in parent data
        fishes = parent_bouts['freq_code'].unique()
        logger.debug(f"  Processing {len(fishes)} fish")
        
        for fish_id in fishes:
            # Filter data for this fish
            parent_fish_bouts = parent_bouts[parent_bouts['freq_code'] == fish_id]
            parent_fish_dat = parent_dat[parent_dat['freq_code'] == fish_id].copy()
            child_fish_dat = child_dat[child_dat['freq_code'] == fish_id].copy()
            
            if parent_fish_dat.empty or child_fish_dat.empty:
                continue
            
            # Initialize overlapping column if not exists
            if 'overlapping' not in parent_fish_dat.columns:
                parent_fish_dat['overlapping'] = np.float32(0)
            if 'overlapping' not in child_fish_dat.columns:
                child_fish_dat['overlapping'] = np.float32(0)
            
            # Process each bout for this fish
            for _, bout_row in parent_fish_bouts.iterrows():
                min_epoch = bout_row['min_epoch']
                max_epoch = bout_row['max_epoch']
                
                # Get detections in this bout window
                parent_in_bout = parent_fish_dat[
                    (parent_fish_dat['epoch'] >= min_epoch) & 
                    (parent_fish_dat['epoch'] <= max_epoch)
                ]
                child_in_bout = child_fish_dat[
                    (child_fish_dat['epoch'] >= min_epoch) & 
                    (child_fish_dat['epoch'] <= max_epoch)
                ]
                
                if len(parent_in_bout) == 0 or len(child_in_bout) == 0:
                    continue  # No overlap
                
                overlaps_processed += 1
                
                # Calculate confidence for each receiver
                if use_posteriors:
                    parent_confidence = parent_in_bout['posterior_T'].mean()
                    child_confidence = child_in_bout['posterior_T'].mean()
                    metric_name = "posterior_T"
                else:
                    # Fallback to normalized power (not recommended for mixed receivers)
                    parent_confidence = self._normalize_power(parent_in_bout['power'])
                    child_confidence = self._normalize_power(child_in_bout['power'])
                    metric_name = "normalized_power"
                
                confidence_diff = parent_confidence - child_confidence
                
                # Make classification decision
                if confidence_diff > confidence_threshold:
                    # Parent has higher confidence → fish near parent → remove child
                    decision = "remove_child"
                    parent_overlapping = np.float32(0)  # Keep parent
                    child_overlapping = np.float32(1)   # Remove child
                    
                elif confidence_diff < -confidence_threshold:
                    # Child has higher confidence → fish near child → remove parent
                    decision = "remove_parent"
                    parent_overlapping = np.float32(1)  # Remove parent
                    child_overlapping = np.float32(0)   # Keep child
                    
                else:
                    # Confidence difference too small → ambiguous → keep both
                    decision = "keep_both"
                    parent_overlapping = np.float32(0)
                    child_overlapping = np.float32(0)
                
                logger.debug(
                    f"    {fish_id} [{min_epoch}-{max_epoch}]: "
                    f"parent_{metric_name}={parent_confidence:.3f}, "
                    f"child_{metric_name}={child_confidence:.3f}, "
                    f"diff={confidence_diff:.3f} → {decision}"
                )
                
                # Update DataFrames using boolean indexing (safe from SettingWithCopyWarning)
                parent_mask = (
                    (parent_fish_dat['freq_code'] == fish_id) &
                    (parent_fish_dat['epoch'] >= min_epoch) &
                    (parent_fish_dat['epoch'] <= max_epoch)
                )
                child_mask = (
                    (child_fish_dat['freq_code'] == fish_id) &
                    (child_fish_dat['epoch'] >= min_epoch) &
                    (child_fish_dat['epoch'] <= max_epoch)
                )
                
                parent_fish_dat.loc[parent_mask, 'overlapping'] = parent_overlapping
                child_fish_dat.loc[child_mask, 'overlapping'] = child_overlapping
                
                if decision != "keep_both":
                    total_detections_marked += len(parent_in_bout) + len(child_in_bout)
            
            # Update the main DataFrames
            parent_dat.loc[parent_dat['freq_code'] == fish_id, 'overlapping'] = \
                parent_fish_dat['overlapping'].values
            child_dat.loc[child_dat['freq_code'] == fish_id, 'overlapping'] = \
                child_fish_dat['overlapping'].values
        
        # Ensure overlapping column exists before writing
        if 'overlapping' not in parent_dat.columns:
            parent_dat['overlapping'] = np.float32(0)
        if 'overlapping' not in child_dat.columns:
            child_dat['overlapping'] = np.float32(0)
        
        # Write results for this edge
        logger.debug(f"  Writing results for {parent}")
        self.write_results_to_hdf5(parent_dat)
        
        logger.debug(f"  Writing results for {child}")
        self.write_results_to_hdf5(child_dat)
        
        # Memory cleanup
        del parent_bouts, parent_dat, child_dat
        gc.collect()
    
    logger.info(f"✓ Unsupervised overlap removal complete")
    logger.info(f"  Processed {overlaps_processed} overlapping bouts")
    logger.info(f"  Marked {total_detections_marked} detections as overlapping")

def _normalize_power(self, power_series):
    """
    Helper method to normalize power values (fallback when posteriors not available).
    
    Parameters
    ----------
    power_series : pd.Series
        Power values to normalize
    
    Returns
    -------
    float
        Mean of normalized power values, or 0 if normalization fails
    """
    try:
        power_array = power_series.values
        power_min = power_array.min()
        power_max = power_array.max()
        power_range = power_max - power_min
        
        if power_range > 0:
            normalized = (power_array - power_min) / power_range
            return normalized.mean()
        else:
            # All powers identical - can't distinguish
            return 0.0
    except Exception as e:
        logger.warning(f"Power normalization failed: {e}")
        return 0.0
