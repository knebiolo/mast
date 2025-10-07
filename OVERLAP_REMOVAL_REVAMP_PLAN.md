# Overlap Removal Module Revamp Plan

## Executive Summary

**Goal**: Revamp `overlap_removal.py` to use Naive Bayes posterior probabilities for overlap detection while preserving all working code.

**Key Insight**: Posterior probabilities are superior to raw power because:
- ✅ Already normalized per receiver (handles different manufacturers, gains, calibrations)
- ✅ Multi-factor decision (hit_ratio, noise, lag, cons_length, power)
- ✅ Scale-invariant (always 0-1, comparable across receivers)
- ✅ Already validated by classification process

**Scope**: 
- Keep ~550 lines of working code untouched
- Refactor ~200 lines of broken `unsupervised_removal()`
- Delete ~700 lines of commented dead code
- Add logging, validation, progress bars

---

## Phase 1: Preserve Working Code ✅

### Do NOT Touch (Verified Working):

**`bout` class (lines 41-450)** - ✅ Keep as-is
```python
class bout():
    def __init__(self, radio_project, node, lag_window, time_limit)  # ✅
    def prompt_for_params(self, model_type)                          # ✅
    def find_knot(self, initial_knot_guess)                          # ✅
    def find_knots(self, initial_knot_guesses)                       # ✅
    def fit_processes(self)                                          # ✅
    def presence(self, threshold)                                     # ✅
```
**Functionality**: Piecewise exponential decay models for bout detection
**Status**: Mathematically sound, no reported issues
**Action**: None

**`overlap_reduction` class initialization (lines 693-744)** - ✅ Keep as-is
```python
def __init__(self, nodes, edges, radio_project)
```
**Functionality**: Loads presence and classified data for each node
**Status**: Working correctly
**Action**: None (maybe add logging)

**`nested_doll()` method (lines 860-895)** - ✅ Keep as-is
```python
def nested_doll(self)
```
**Functionality**: Simple interval-tree based overlap detection
**Status**: Working correctly, fast, conservative
**Action**: None (maybe add logging)

**`write_results_to_hdf5()` helper (lines 897-918)** - ✅ Keep as-is
```python
def write_results_to_hdf5(self, df)
```
**Functionality**: Writes overlapping detections to HDF5
**Status**: Working correctly
**Action**: None

---

## Phase 2: Refactor `unsupervised_removal()` 🔧

### Current Issues (lines 747-832)

1. ❌ Logic backwards (marks wrong receiver)
2. ❌ Only updates parent (child never touched)
3. ❌ Uses t-test on paired data (wrong statistical test)
4. ❌ Uses raw power (incomparable across receivers)
5. ❌ Inconsistent normalization
6. ❌ No edge case handling
7. ❌ Excessive print statements (should use logging)

### New Implementation Strategy

**Replace with posterior-based algorithm:**

```python
def unsupervised_removal(self, confidence_threshold=0.1, use_posteriors=True,
                        fallback_to_power=False):
    """Use Naive Bayes posterior_T to identify overlaps."""
    
    # Key changes:
    # 1. Use posterior_T from classification (multi-factor, normalized)
    # 2. Update BOTH parent and child DataFrames
    # 3. No statistical test - simple threshold comparison
    # 4. Comprehensive logging
    # 5. Edge case handling
    # 6. Progress bars
```

**Algorithm**:
```
FOR each parent→child edge:
    FOR each fish:
        FOR each overlapping bout:
            parent_confidence = mean(parent.posterior_T)
            child_confidence = mean(child.posterior_T)
            
            IF parent_confidence - child_confidence > threshold:
                # Fish near parent
                mark child as overlapping (remove)
                keep parent
                
            ELIF child_confidence - parent_confidence > threshold:
                # Fish near child
                mark parent as overlapping (remove)
                keep child
                
            ELSE:
                # Ambiguous - keep both (conservative)
                keep both
```

**Why This Works**:
- **Heterogeneous receivers**: Posterior already accounts for receiver-specific characteristics
- **Multi-factor**: Uses all predictors (not just power)
- **Validated**: Same probabilities used for classification
- **Interpretable**: "Receiver with higher confidence wins"
- **Conservative**: Ambiguous cases keep both detections

---

## Phase 3: Clean Up Dead Code 🧹

### Delete Commented Code (Safe - in git history)

**Section 1: Lines 532-680** (~150 lines)
```python
# class overlap_reduction():  # OLD ATTEMPT WITH DASK
#     def __init__...
```
**Reason**: Superseded by working pandas implementation

**Section 2: Lines 920-997** (~80 lines)
```python
# def _plot_kmeans_results...
```
**Reason**: KMeans approach abandoned

**Section 3: Lines 998-1105** (~110 lines)
```python
# def unsupervised_removal():  # OLD KMEANS VERSION
#     ...KMeans clustering...
```
**Reason**: Posterior probability approach is better

**Section 4: Lines 1106-1254** (~150 lines)
```python
# # def unsupervised_removal():  # ANOTHER OLD VERSION
```
**Reason**: More old attempts

**Total to delete**: ~490 lines of commented code
**Safety**: All preserved in git history

---

## Phase 4: Add Modern Features 📝

### 1. Add Logging (Following radio_project.py pattern)

**Replace** (throughout file):
```python
print("Processing edge...")
print(f"Fish ID: {fish_id}")
print("No overlaps found")
```

**With**:
```python
logger.info("Processing edge...")
logger.debug(f"Fish ID: {fish_id}")
logger.info("No overlaps found")
```

**Benefits**:
- Adjustable verbosity (DEBUG/INFO/WARNING/ERROR)
- Can log to file for troubleshooting
- Professional package behavior
- Consistent with rest of codebase

### 2. Add Progress Bars

**Add tqdm** to long loops:
```python
from tqdm import tqdm

# Edge processing
for parent, child in tqdm(self.edges, desc="Processing edges", unit="edge"):
    ...

# Fish processing (if many fish)
for fish in tqdm(fishes, desc=f"  {parent}→{child}", unit="fish"):
    ...
```

### 3. Add Input Validation

**Check required columns**:
```python
if use_posteriors:
    required_cols = ['posterior_T', 'posterior_F']
    for node in self.nodes:
        if not all(col in self.node_recap_dict[node].columns for col in required_cols):
            if fallback_to_power:
                logger.warning("posterior_T not found, using power fallback")
                use_posteriors = False
            else:
                raise ValueError(
                    "posterior_T not found. Run classification first or set "
                    "use_posteriors=False or fallback_to_power=True"
                )
```

### 4. Add Comprehensive Docstrings

**Module level**:
```python
"""
Overlap removal for radio telemetry data.

This module identifies and removes redundant detections when multiple receivers
detect the same tagged fish. Two approaches are provided:

1. nested_doll: Simple interval-based overlap detection (conservative)
2. unsupervised_removal: Confidence-based using Naive Bayes posteriors (sophisticated)

Classes
-------
bout
    Identifies temporal bouts of presence using piecewise exponential decay
overlap_reduction
    Manages receiver network and implements overlap removal algorithms
"""
```

**Method level**:
- Complete parameter descriptions
- Return value documentation
- Algorithm explanation
- Examples
- See Also sections

---

## Phase 5: Testing Strategy 🧪

### Unit Tests

**Test 1: Posterior-based classification**
```python
def test_posterior_comparison():
    # Parent high confidence (0.9), child low (0.3)
    # Expected: keep parent, remove child
    parent_posterior_T = [0.88, 0.91, 0.92, 0.87]
    child_posterior_T = [0.28, 0.32, 0.31, 0.29]
    
    # Run algorithm
    result = classify_overlap(parent_posterior_T, child_posterior_T, threshold=0.1)
    
    assert result['parent_overlapping'] == 0  # Keep
    assert result['child_overlapping'] == 1   # Remove
```

**Test 2: Ambiguous case**
```python
def test_ambiguous_overlap():
    # Similar confidence → keep both
    parent_posterior_T = [0.75, 0.78, 0.76]
    child_posterior_T = [0.74, 0.77, 0.75]
    
    result = classify_overlap(parent_posterior_T, child_posterior_T, threshold=0.1)
    
    assert result['parent_overlapping'] == 0  # Keep
    assert result['child_overlapping'] == 0   # Keep
```

**Test 3: Edge cases**
```python
def test_edge_cases():
    # All posteriors identical
    parent_posterior_T = [0.5, 0.5, 0.5]
    child_posterior_T = [0.5, 0.5, 0.5]
    # Should not crash, keep both
    
    # Empty arrays
    parent_posterior_T = []
    child_posterior_T = [0.8]
    # Should handle gracefully
    
    # NaN values
    parent_posterior_T = [0.8, np.nan, 0.75]
    child_posterior_T = [0.3, 0.32, 0.31]
    # Should handle (use nanmean)
```

### Integration Tests

**Test with real data**:
```python
def test_with_project_data():
    # Load test project
    project = pymast.radio_project(...)
    
    # Run classification
    project.classify(...)
    
    # Create overlap object
    nodes = ['R01', 'R02', 'R03']
    edges = [('R01', 'R02'), ('R02', 'R03')]
    overlap_obj = overlap_reduction(nodes, edges, project)
    
    # Run unsupervised removal
    overlap_obj.unsupervised_removal(confidence_threshold=0.1)
    
    # Check results
    overlaps = pd.read_hdf(project.db, 'overlapping')
    assert 'overlapping' in overlaps.columns
    assert overlaps['overlapping'].isin([0, 1]).all()
    
    # Compare with nested_doll (should mark fewer overlaps)
    overlap_obj2 = overlap_reduction(nodes, edges, project)
    overlap_obj2.nested_doll()
    overlaps2 = pd.read_hdf(project.db, 'overlapping')
    
    # Posterior method should be less conservative
    assert (overlaps['overlapping'] == 1).sum() <= (overlaps2['overlapping'] == 1).sum()
```

---

## Phase 6: Documentation Updates 📚

### Update Module Docstring

Add comparison table:
```
Method Comparison
-----------------
                    nested_doll         unsupervised_removal
Complexity          Low                 Medium
Input Required      Presence bouts      Classified data + posteriors
Receiver Types      Any                 Any (especially mixed types)
Considers Power     No                  Yes (via posteriors)
Conservative        Very                Moderately
Speed               Fast                Moderate
Interpretability    Simple intervals    Confidence-based

When to use nested_doll:
- Quick conservative estimate
- No classification done yet
- Simple project structure

When to use unsupervised_removal:
- Mixed receiver types (different manufacturers)
- Want less conservative removal
- Have completed classification
- Want confidence-weighted decisions
```

### Add Usage Examples

```python
# Example 1: Basic usage after classification
import pymast
from pymast.overlap_removal import bout, overlap_reduction

# Setup project
project = pymast.radio_project(...)

# Run classification first
project.train(...)
project.classify(...)

# Define network
nodes = ['R01', 'R02', 'R03', 'R04']
edges = [('R01', 'R02'), ('R02', 'R03'), ('R03', 'R04')]

# Create overlap object
overlap_obj = overlap_reduction(nodes, edges, project)

# Run posterior-based removal (recommended)
overlap_obj.unsupervised_removal(confidence_threshold=0.1)

# Results written to project.db at /overlapping key
overlaps = pd.read_hdf(project.db, 'overlapping')

# Example 2: Conservative approach
overlap_obj.nested_doll()  # More conservative

# Example 3: Custom threshold
# Larger threshold = more conservative (keeps more detections)
overlap_obj.unsupervised_removal(confidence_threshold=0.2)

# Example 4: Fallback to power (not recommended for mixed receivers)
overlap_obj.unsupervised_removal(use_posteriors=False)
```

---

## Implementation Checklist

### Phase 1: Preparation
- [x] Create analysis document
- [x] Create revamp plan
- [x] Create proposed implementation
- [ ] Review with Kevin
- [ ] Agree on approach

### Phase 2: Code Changes
- [ ] Add logging import to top of file
- [ ] Add tqdm import
- [ ] Replace `unsupervised_removal()` method (lines 747-832)
- [ ] Add `_normalize_power()` helper method
- [ ] Update `__init__()` with optional logging
- [ ] Update `nested_doll()` with logging
- [ ] Update `write_results_to_hdf5()` with logging

### Phase 3: Cleanup
- [ ] Delete commented code (lines 532-680)
- [ ] Delete commented code (lines 920-997)
- [ ] Delete commented code (lines 998-1105)
- [ ] Delete commented code (lines 1106-1254)
- [ ] Update module docstring
- [ ] Add method docstrings

### Phase 4: Testing
- [ ] Write unit tests
- [ ] Test with small synthetic dataset
- [ ] Test with real project data
- [ ] Compare nested_doll vs unsupervised_removal
- [ ] Validate overlapping detections make sense

### Phase 5: Documentation
- [ ] Update README with overlap removal section
- [ ] Add examples to docs/TUTORIAL.md
- [ ] Update API_REFERENCE.md
- [ ] Add to CHANGELOG.md

---

## Expected Outcomes

### File Size Change
- **Before**: 1,254 lines
- **After**: ~750 lines
- **Reduction**: ~500 lines (40% smaller, much cleaner)

### Code Quality
- **Before**: Mix of working code, broken code, commented experiments
- **After**: Only working, tested, documented code

### Functionality
- ✅ Preserved: bout detection, nested_doll, data loading
- 🔧 Improved: unsupervised_removal (now actually works!)
- 🆕 Added: Posterior-based confidence comparison
- 🆕 Added: Logging, progress bars, validation

### User Experience
- Clear which method to use when
- Adjustable confidence threshold
- Real-time progress feedback
- Debug logging available
- Comprehensive error messages

---

## Risk Assessment

### Low Risk
- ✅ Preserving all working code
- ✅ Using established patterns (logging from radio_project.py)
- ✅ Posterior probabilities already validated
- ✅ Changes in version control (can rollback)

### Medium Risk
- ⚠️ New algorithm needs validation with real data
- ⚠️ Threshold value may need tuning per project
- ⚠️ Users need to run classification before overlap removal

### Mitigation
- Keep nested_doll as conservative fallback
- Provide clear documentation and examples
- Add validation that posteriors exist
- Test with multiple datasets before release

---

## Questions for Discussion

1. **Confidence threshold default**: Is 0.1 reasonable, or should it be higher (more conservative)?

2. **Fallback behavior**: If posteriors not available, should it:
   - Error (force user to classify first)
   - Warn and use power fallback
   - Automatically fall back to nested_doll

3. **Both receivers marked**: Currently if ambiguous, keep both. Alternative: mark both as overlapping (ultra-conservative)?

4. **Performance**: With large datasets, do we need Dask parallelization, or is pandas sufficient?

5. **Additional metrics**: Besides posterior_T, should we consider:
   - Posterior ratio (posterior_T / posterior_F)
   - Number of detections in bout
   - Bout duration

---

**Status**: Plan complete, ready for implementation  
**Next Step**: Get Kevin's approval and preferences, then implement  
**Estimated Time**: 4-6 hours for full implementation + testing
