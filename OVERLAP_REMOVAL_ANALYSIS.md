# Deep Dive Analysis: overlap_removal.py

## Executive Summary

The `overlap_removal.py` module addresses a critical challenge in radio telemetry: **distinguishing true fish detections from overlapping signals** when multiple receivers detect the same tagged fish. The current implementation uses a statistical approach (t-test), with extensive commented-out alternatives exploring machine learning methods (KMeans, GMM, KNN). 

**Overall Assessment**: The current `unsupervised_removal` algorithm is conceptually sound but has significant issues that likely prevent it from working correctly in production.

---

## Module Structure

### Two Main Classes

1. **`bout` class** (~400 lines)
   - Identifies temporal "bouts" of presence using piecewise exponential decay models
   - Fits 2-3 process exponential decay to inter-detection intervals
   - Calculates presence periods for downstream overlap analysis
   - **Status**: ✅ Appears complete and functional

2. **`overlap_reduction` class** (~500+ lines active + ~700 lines commented)
   - Manages network graph of receiver relationships (parent → child)
   - Implements two algorithms:
     - `nested_doll()` - Simple interval tree overlap detection
     - `unsupervised_removal()` - Statistical classification of overlaps
   - **Status**: ⚠️ Has critical issues (detailed below)

---

## Current `unsupervised_removal()` Algorithm (Lines 747-832)

### Conceptual Approach

The algorithm attempts to classify overlapping detections as "near" (keep) or "far" (remove) using:

1. **Graph traversal**: Iterate through parent→child receiver pairs
2. **Bout matching**: For each fish, find overlapping time periods between parent and child
3. **Power normalization**: Normalize signal power separately for each receiver
4. **Statistical test**: Use Welch's t-test to compare normalized power distributions
5. **Classification**: 
   - If parent power > child power AND p < 0.05 → mark parent as "near" (overlapping = 0)
   - Otherwise → mark as "far" (overlapping = 1)

### Code Flow

```python
for parent, child in edges:
    for fish_id in parent_fish:
        for bout in parent_bouts:
            # Extract overlapping detections
            parent_power = detections in bout window
            child_power = detections in bout window
            
            # Normalize within-receiver
            parent_norm = (parent_power - min) / (max - min)
            child_norm = (child_power - min) / (max - min)
            
            # Statistical test
            t_stat, p_value = ttest_ind(parent_norm, child_norm)
            
            # Classify
            if mean(parent_norm) > mean(child_norm) AND p < 0.05:
                overlapping = 0  # NEAR
            else:
                overlapping = 1  # FAR
```

---

## Critical Issues Identified

### 🔴 **Issue 1: Logic Inversion / Semantic Confusion**

**Problem**: The classification labels are backwards or confusing.

```python
if np.mean(parent_norm_power) > np.mean(child_norm_power) and p_value < 0.05:
    parent_classification = np.float32(0)  # "Near"
else:
    parent_classification = np.float32(1)  # "Far"
```

**Why this is wrong:**
- If the **parent has higher power**, the fish is likely **closer to the parent**
- Therefore, detections at the **child** should be marked as overlapping (redundant)
- But the code marks the **parent** as "far" (overlapping = 1) in this case
- This is backwards!

**What should happen:**
- High parent power → Fish is near parent → **Child detections should be removed**
- High child power → Fish is near child → **Parent detections might be overlapping**

**Expected logic:**
```python
if np.mean(parent_norm_power) > np.mean(child_norm_power) and p_value < 0.05:
    # Fish is closer to parent, remove child detections
    child_classification = 1  # Mark child as overlapping/remove
    parent_classification = 0  # Keep parent
else:
    # Ambiguous or fish closer to child
    parent_classification = 1  # Mark parent as overlapping
    child_classification = 0  # Keep child
```

### 🔴 **Issue 2: Only Parent Data is Updated**

**Problem**: The algorithm only modifies `parent_fish_dat` but never updates `child_fish_dat`.

```python
# Updates parent
parent_fish_dat.loc[...] = parent_classification

# Child is never updated! ❌
```

**Impact**: 
- Even if the logic was correct, child detections are never marked
- The classification is incomplete
- Half of the overlap problem remains unaddressed

### 🔴 **Issue 3: Normalization Issues**

**Problem**: Normalization is done **within bout** but using **receiver-wide** min/max:

```python
# Uses fish-specific power from THIS bout
parent_power = parent_fish_dat[in_bout_window].power.values

# But normalizes using ENTIRE receiver's min/max
max_parent_power = parent_fish_dat.power.max()  # All detections for this fish
parent_norm_power = (parent_power - np.min(parent_power)) / (max_parent_power - np.min(parent_power))
```

**Issues**:
- Mixing local (bout) and global (receiver) statistics
- `np.min(parent_power)` is the minimum **within this bout**
- But denominator uses **max across all bouts** for this fish
- This creates inconsistent scaling
- If bout has narrow power range, normalization may amplify noise

**Better approach**:
```python
# Option A: Normalize within bout only
parent_norm = (parent_power - parent_power.min()) / (parent_power.max() - parent_power.min())

# Option B: Normalize using receiver-wide stats consistently
parent_norm = (parent_power - all_parent_power.min()) / (all_parent_power.max() - all_parent_power.min())
```

### 🟡 **Issue 4: Statistical Test Validity**

**Problem**: Using Welch's t-test assumes:
1. **Independence**: Parent and child samples are independent
2. **Normality**: Power distributions are approximately normal
3. **Meaningful comparison**: Normalized powers are comparable

**Reality**:
- **Same fish, same time** → detections are NOT independent (correlation ~ 1.0)
- **Signal power distributions** are typically right-skewed, not normal
- **Different receivers** may have different noise floors, antenna gains
- Normalization helps but doesn't fully address receiver-specific biases

**Alternative approaches**:
- **Paired t-test** or **Wilcoxon signed-rank test** (accounts for pairing)
- **Effect size** (Cohen's d) might be more robust than p-value
- **Signal-to-noise ratio** comparison rather than raw power
- **Median absolute difference** (robust to outliers)

### 🟡 **Issue 5: Memory Safety Concerns**

**Problem**: 
```python
parent_fish_dat.loc[condition, 'overlapping'] = parent_classification
```

Using `.loc[]` on a filtered DataFrame can create `SettingWithCopyWarning` and may not persist changes back to the original `parent_dat` DataFrame.

**Better approach**:
```python
# Use explicit indexing
idx = parent_dat[condition].index
parent_dat.loc[idx, 'overlapping'] = parent_classification
```

### 🟡 **Issue 6: No Edge Cases Handled**

**Missing checks**:
- What if `parent_power` or `child_power` arrays are empty? ✅ (Handled with `continue`)
- What if all powers are identical (zero variance)? ❌ (Division by zero possible)
- What if t-test fails (e.g., single sample)? ❌ (No try/except)
- What if receivers have drastically different power ranges? ❌

**Example issue**:
```python
parent_norm_power = (parent_power - np.min(parent_power)) / (max_parent_power - np.min(parent_power))
# If max_parent_power == np.min(parent_power) → division by zero!
```

### 🟢 **Issue 7: Excessive Verbosity**

**Current**: Prints for every fish, every bout, every comparison
```python
print(f"Processing fish ID: {fish_id}")
print('Overlapping detections found')
print(f"Fish ID {fish_id}: Parent classified as NEAR with p-value: {p_value:.4f}")
```

**Impact**: 
- Output will be overwhelming for large datasets
- 100 fish × 10 bouts each = 1000+ print statements per edge
- Should use logging with adjustable verbosity (already established pattern in your codebase!)

---

## Alternative Approaches (Commented Code Analysis)

The file contains ~700 lines of commented-out code exploring different methods:

### **Approach 1: KMeans Clustering** (Lines 998-1105)
```python
# Cluster normalized power into 2 groups (near vs far)
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(normalized_power)
```

**Pros**:
- Unsupervised - no p-value threshold needed
- Natural grouping of power distributions
- Handles multimodal distributions better than t-test

**Cons**:
- Assumes exactly 2 clusters (may not always be true)
- Sensitive to outliers
- Doesn't account for temporal structure
- Commented code has similar logic issues (doesn't update both parent and child)

### **Approach 2: Dask-Based Parallel Processing** (Lines 532-680)
```python
# Use Dask dataframes for larger-than-memory processing
pres_data = dd.read_hdf(...)
```

**Pros**:
- Scales to very large datasets
- Parallel processing across cores
- Memory efficient

**Cons**:
- Added complexity
- Harder to debug
- Current pandas implementation should work for most datasets

### **Approach 3: `nested_doll()` Method** (Lines 860-895, active code)
```python
# Simple interval tree approach - mark any detection that overlaps with child bout
overlaps = (min_epochs[:, None] <= fish_epochs) & (max_epochs[:, None] > fish_epochs)
```

**Pros**:
- ✅ Simple and deterministic
- ✅ Fast (vectorized operations)
- ✅ No statistical assumptions
- ✅ Currently working correctly

**Cons**:
- Binary classification only (overlap / no overlap)
- Doesn't consider signal strength
- May be overly conservative (marks more overlaps than necessary)

---

## Comparison of Methods

| Aspect | Current `unsupervised_removal` | `nested_doll` | KMeans (commented) |
|--------|-------------------------------|---------------|-------------------|
| **Complexity** | High (t-test, normalization) | Low (interval checks) | Medium (clustering) |
| **Correctness** | ❌ Logic errors | ✅ Working | ⚠️ Similar logic errors |
| **Statistical Rigor** | ⚠️ Questionable assumptions | N/A | N/A |
| **Scalability** | ✅ Good with fixes | ✅ Excellent | ✅ Good |
| **Interpretability** | ⚠️ P-values hard to explain | ✅ Very clear | ⚠️ Clusters arbitrary |
| **Edge Cases** | ❌ Many unhandled | ✅ Robust | ⚠️ Assumes 2 clusters |
| **Power Info Used** | ✅ Yes | ❌ No | ✅ Yes |

---

## Recommendations

### **Option 1: Fix Current `unsupervised_removal` (Recommended)**

**Pros**: Preserves signal strength information, statistically motivated
**Effort**: Medium (2-3 hours)

**Changes needed**:
1. ✅ Fix logic inversion - mark child detections when parent has higher power
2. ✅ Update both parent AND child DataFrames
3. ✅ Fix normalization to be consistent
4. ✅ Add edge case handling (zero variance, empty arrays, NaN)
5. ✅ Add try/except around t-test
6. ✅ Replace print with logging (logger.debug for detailed info)
7. ✅ Consider more robust statistical test (Mann-Whitney U, paired test)
8. ✅ Add validation - check that overlapping column exists before writing

### **Option 2: Hybrid Approach (Best Long-term)**

**Pros**: Combines deterministic interval detection with optional power-based refinement
**Effort**: High (4-6 hours)

**Implementation**:
```python
def hybrid_removal(self, use_power_test=True):
    # Step 1: Use nested_doll to identify ALL overlaps (conservative)
    self.nested_doll()
    
    # Step 2: If requested, refine using power analysis
    if use_power_test:
        self._refine_with_power_test()
        
def _refine_with_power_test(self):
    # For each overlap identified by nested_doll:
    # - Compare power distributions
    # - Un-mark overlap if power difference is insignificant
    # - This makes the method less conservative
```

**Benefits**:
- Nested doll ensures we don't miss overlaps (safe default)
- Power test refines the results (optional, more sophisticated)
- Clear two-stage process is easier to debug and explain
- Users can choose conservative (interval only) vs refined (power adjusted)

### **Option 3: Simplified Power Comparison**

**Pros**: Simpler than t-test, fewer assumptions
**Effort**: Low (1-2 hours)

**Implementation**:
```python
# Instead of t-test, use median power + threshold
parent_median = np.median(parent_power)
child_median = np.median(child_power)
power_ratio = parent_median / child_median

if power_ratio > 1.5:  # Parent is significantly stronger
    # Fish is near parent, remove child detections
    child_classification = 1
elif power_ratio < 0.67:  # Child is significantly stronger
    # Fish is near child, remove parent detections  
    parent_classification = 1
else:
    # Ambiguous - keep both (conservative)
    pass
```

**Benefits**:
- No p-values to interpret
- More robust to non-normal distributions
- Easy to tune threshold based on empirical results
- Clear logic: receiver with stronger signal "wins"

---

## Specific Code Issues to Fix

### Line 805: Incorrect t-test usage
```python
# CURRENT (wrong - samples are paired, not independent)
t_stat, p_value = ttest_ind(parent_norm_power, child_norm_power, equal_var=False)

# BETTER (accounts for pairing, but requires equal sample sizes)
from scipy.stats import ttest_rel
if len(parent_norm_power) == len(child_norm_power):
    t_stat, p_value = ttest_rel(parent_norm_power, child_norm_power)
else:
    # Fall back to Mann-Whitney U (non-parametric, handles unequal sizes)
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(parent_norm_power, child_norm_power, alternative='greater')
```

### Lines 795-800: Normalization fix
```python
# CURRENT (inconsistent mixing of local and global)
parent_norm_power = (parent_power - np.min(parent_power)) / (max_parent_power - np.min(parent_power))

# BETTER (consistent receiver-wide normalization)
# Pre-compute at receiver level
parent_min = parent_fish_dat.power.min()
parent_max = parent_fish_dat.power.max()
parent_range = parent_max - parent_min

if parent_range > 0:  # Avoid division by zero
    parent_norm_power = (parent_power - parent_min) / parent_range
else:
    # All powers identical - can't distinguish, mark as ambiguous
    continue
```

### Lines 808-813: Fix classification logic
```python
# CURRENT (backwards)
if np.mean(parent_norm_power) > np.mean(child_norm_power) and p_value < 0.05:
    parent_classification = np.float32(0)  # Near - WRONG!

# CORRECTED
if np.mean(parent_norm_power) > np.mean(child_norm_power) and p_value < 0.05:
    # Parent has significantly higher power → fish is near parent
    parent_classification = np.float32(0)  # Keep parent (not overlapping)
    child_classification = np.float32(1)   # Remove child (overlapping)
else:
    # Ambiguous or child has higher power
    parent_classification = np.float32(1)  # Remove parent
    child_classification = np.float32(0)   # Keep child
```

### Lines 816-821: Update both DataFrames
```python
# CURRENT (only updates parent)
parent_fish_dat.loc[condition, 'overlapping'] = parent_classification

# CORRECTED (update both)
# Update parent
parent_idx = parent_dat.index[
    (parent_dat['freq_code'] == fish_id) &
    (parent_dat['epoch'] >= parent_row['min_epoch']) &
    (parent_dat['epoch'] <= parent_row['max_epoch'])
]
parent_dat.loc[parent_idx, 'overlapping'] = parent_classification

# Update child
child_idx = child_dat.index[
    (child_dat['freq_code'] == fish_id) &
    (child_dat['epoch'] >= parent_row['min_epoch']) &
    (child_dat['epoch'] <= parent_row['max_epoch'])
]
child_dat.loc[child_idx, 'overlapping'] = child_classification
```

---

## Testing Strategy

### Unit Tests Needed

1. **Test normalization edge cases**
   ```python
   # All powers identical
   powers = np.array([50.0, 50.0, 50.0])
   # Should not crash
   
   # Single value
   powers = np.array([50.0])
   # Should handle gracefully
   ```

2. **Test classification logic**
   ```python
   # Parent significantly stronger → child should be marked overlapping
   parent_power = np.array([80, 85, 90, 88])
   child_power = np.array([50, 52, 48, 51])
   # Expected: parent_overlapping=0, child_overlapping=1
   ```

3. **Test both DataFrames are updated**
   ```python
   # After processing, both parent_dat and child_dat should have 'overlapping' column
   assert 'overlapping' in parent_dat.columns
   assert 'overlapping' in child_dat.columns
   ```

### Integration Tests

1. **Small synthetic dataset**
   - 2 receivers with known overlap
   - 1 fish with clear power gradient
   - Verify correct detections are marked

2. **Real data validation**
   - Run on actual dataset
   - Compare with `nested_doll()` results
   - Check that power-based method is less conservative (marks fewer overlaps)

3. **Performance test**
   - Large dataset (10,000+ detections)
   - Measure runtime
   - Check memory usage

---

## Documentation Needs

### Current State
- ❌ No docstring for `unsupervised_removal`
- ❌ No explanation of statistical method
- ❌ No guidance on when to use vs `nested_doll`
- ❌ Algorithm assumptions not stated

### Needed Documentation

1. **Method docstring** with:
   - Statistical approach explanation
   - Assumptions (independence, normality)
   - When to use this vs `nested_doll`
   - Parameters and return values
   - Example usage

2. **Module-level documentation** with:
   - Comparison of nested_doll vs unsupervised_removal
   - Guidance on parameter selection
   - Performance characteristics
   - Known limitations

3. **Inline comments** for:
   - Normalization rationale
   - Statistical test choice
   - Classification logic
   - Edge case handling

---

## Related TODOs in Code

Found 2 TODOs in the module:

**Line 261** - Piecewise exponential decay math
```python
#TODO - is the math correct?
```
**Impact**: Medium - Affects bout detection accuracy

**Line 284** - Model selection
```python
#TODO - AIC or BIC possible expansion?
```
**Impact**: Low - Enhancement for model selection

---

## Final Verdict

### Current `unsupervised_removal` Implementation: 2/10

**What's good**:
- ✅ Conceptually sound approach (use power to distinguish near vs far)
- ✅ Attempts statistical rigor with t-test
- ✅ Iterates through all parent-child pairs

**What's broken**:
- ❌ **Logic is backwards** (marks wrong receiver as overlapping)
- ❌ **Only updates parent** (child never marked)
- ❌ **Inconsistent normalization** (mixes local and global stats)
- ❌ **Wrong statistical test** (assumes independence when samples are paired)
- ❌ **No edge case handling** (division by zero, NaN, empty arrays)
- ❌ **Excessive verbosity** (should use logging)

**Can it work?**: Not in current state - needs significant fixes

### Recommended Path Forward

**Short term** (1-2 weeks):
1. Fix the critical logic errors in `unsupervised_removal`
2. Add comprehensive logging (already established pattern in project)
3. Add edge case handling and validation
4. Write unit tests for normalization and classification logic

**Medium term** (1-2 months):
1. Implement hybrid approach (nested_doll + optional power refinement)
2. Add configuration option to choose algorithm
3. Create visualization tools to validate results
4. Document algorithm choices and trade-offs

**Long term** (research):
1. Investigate machine learning approaches (RF, XGBoost) trained on known overlaps
2. Consider temporal features (detection patterns, bout structure)
3. Explore Bayesian approaches for uncertainty quantification
4. Publish methodology if results are strong

---

## Questions for Discussion

1. **What's the empirical evidence?** 
   - Have you validated that power differences actually indicate distance?
   - Do you have ground truth data (known fish positions)?

2. **What's the use case priority?**
   - Is it more important to be **conservative** (don't miss overlaps) or **precise** (only mark true overlaps)?
   - How do downstream analyses handle overlapping vs non-overlapping detections?

3. **Performance requirements?**
   - How large are typical datasets?
   - Is runtime currently a bottleneck?
   - Would Dask parallelization help?

4. **User expectations?**
   - Do users understand what "unsupervised removal" means?
   - Are they comfortable with statistical thresholds (p < 0.05)?
   - Would simpler median-based method be easier to explain?

---

**Generated**: 2025-10-07  
**Reviewer**: Kevin Nebiolo  
**Status**: Analysis complete - awaiting feedback on recommended approach
