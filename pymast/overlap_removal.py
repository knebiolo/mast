# -*- coding: utf-8 -*-
"""
Bout detection and overlapping detection resolution for radio telemetry data.

This module provides two main classes for identifying and resolving overlapping
detections in radio telemetry studies. Overlapping detections occur when:
1. Multiple receivers detect the same fish simultaneously (spatial ambiguity)
2. Fish movements violate spatial/temporal constraints (impossible transitions)
3. Receiver antenna bleed causes detections from "wrong" direction

Core Classes
------------
- **bout**: Detects spatially/temporally clustered detections using DBSCAN
- **overlap_reduction**: Resolves overlapping detections using signal quality

Bout Detection Workflow
-----------------------
1. **DBSCAN Clustering**: Group detections by time and space
2. **Bout Assignment**: Label each detection with bout number
3. **Presence Matrix**: Create presence/absence by receiver and bout
4. **Visualization**: Diagnostic plots for bout length distributions

Overlap Resolution Workflow
---------------------------
1. **Unsupervised Learning**: Compare signal power and posterior probabilities
2. **Decision Logic**: Mark weaker overlapping detections
3. **Bout Spatial Filter**: Identify bouts with temporal overlap across receivers
4. **Write Results**: Store decisions in HDF5 `/overlapping` table

Resolution Criteria
-------------------
- **Power Comparison**: If both detections have power, keep stronger signal
- **Posterior Comparison**: If both have classification scores, keep higher posterior
- **Ambiguous**: Mark ambiguous if signals equal or criteria unavailable
- **Bout Conflicts**: Identify temporally overlapping bouts at different receivers

Output Tables
-------------
Creates these HDF5 tables:

- `/bouts`: Bout summaries (bout_no, start_time, end_time, detection_count)
- `/presence`: Presence/absence matrix (fish x bout x receiver)
- `/overlapping`: Detection-level decisions (overlapping=0/1, ambiguous=0/1)

Typical Usage
-------------
>>> from pymast.overlap_removal import bout, overlap_reduction
>>> 
>>> # Detect bouts using DBSCAN
>>> bout_obj = bout(
...     db_dir='project.h5',
...     receiver_dat='receivers.csv',
...     eps=3600,  # 1 hour temporal window
...     min_samp=1
... )
>>> bout_obj.cluster()
>>> 
>>> # Resolve overlapping detections
>>> overlap_obj = overlap_reduction(db_dir='project.h5')
>>> overlap_obj.unsupervised()
>>> 
>>> # Visualize results
>>> overlap_obj.visualize_overlaps()
>>> bout_obj.visualize_bout_lengths()

Notes
-----
- DBSCAN parameters (eps, min_samp) control bout sensitivity
- eps should match expected fish residence time at receiver
- min_samp=1 treats every detection as potential bout start
- Bout spatial filter runs automatically after unsupervised()
- Power and posterior columns are optional (conditionally written)

See Also
--------
formatter.time_to_event : Uses presence/absence for TTE analysis
radio_project : Project database management
"""

# import modules required for function dependencies
import os
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
try:
    from tqdm import tqdm
except Exception:
    # tqdm is optional — provide a lightweight passthrough iterator when not installed
    def tqdm(iterable, **kwargs):
        return iterable

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
#from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import dask.dataframe as dd
import dask.array as da
try:
    from dask_ml.cluster import KMeans
    _KMEANS_IMPL = 'dask'
except Exception:
    # dask-ml may not be installed in all environments; fall back to scikit-learn
    from sklearn.cluster import KMeans
    _KMEANS_IMPL = 'sklearn'
from dask import delayed
import sys
import matplotlib
from dask import config
config.set({"dataframe.convert-string": False})
from dask.distributed import Client
#client = Client(processes=False, threads_per_worker=1, memory_limit = '8GB')  # Single-threaded mode
from intervaltree import Interval, IntervalTree
import gc
gc.collect()

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

# Non-interactive helper: if the environment variable PYMAST_NONINTERACTIVE is set, auto-answer prompts
_NON_INTERACTIVE = os.environ.get('PYMAST_NONINTERACTIVE', '0') in ('1', 'true', 'True')

def _prompt(prompt_text, default=None):
    if _NON_INTERACTIVE:
        return default
    try:
        return input(prompt_text)
    except Exception:
        return default

class bout():
    """
    DBSCAN-based bout detection for identifying continuous fish presence periods.
    
    Uses density-based spatial clustering (DBSCAN) to group detections into bouts
    based on temporal proximity. Each bout represents a period of continuous or
    near-continuous presence at a receiver.
    
    Attributes
    ----------
    db : str
        Path to project HDF5 database
    rec_id : str
        Receiver identifier to process
    eps_multiplier : int
        Multiplier for pulse rate to set DBSCAN epsilon (temporal threshold)
    lag_window : int
        Time window in seconds for lag calculations (legacy parameter)
    tags : pandas.DataFrame
        Tag metadata (freq_code, pulse_rate, tag_type, etc.)
    recaptures_df : pandas.DataFrame
        Detections for this receiver
    presence_df : pandas.DataFrame
        Bout presence/absence matrix (fish x bout x receiver)
    
    Methods
    -------
    cluster()
        Run DBSCAN clustering to assign bout numbers
    visualize_bout_lengths()
        Create diagnostic plots showing bout duration distributions
    
    Notes
    -----
    - Physics-based epsilon: pulse_rate * eps_multiplier
    - Default eps_multiplier=5 gives ~40-50 seconds for typical tags
    - min_samples=1 treats every detection as potential bout start
    - Bout numbers are unique per fish, not globally
    - Presence matrix tracks which bouts occurred at which receivers
    
    Examples
    --------
    >>> from pymast.overlap_removal import bout
    >>> bout_obj = bout(
    ...     radio_project=proj,
    ...     rec_id='R03',
    ...     eps_multiplier=5,
    ...     lag_window=9
    ... )
    >>> bout_obj.cluster()
    >>> bout_obj.visualize_bout_lengths()
    
    See Also
    --------
    overlap_reduction : Resolves overlapping detections between bouts
    formatter.time_to_event : Uses bout presence for TTE analysis
    """
    def __init__(self, radio_project, rec_id, eps_multiplier=5, lag_window=9):
        """
        Initialize bout detection for a specific receiver.
        
        Args:
            radio_project: Project object with database and tags
            rec_id (str): Receiver ID to process (e.g., 'R03')
            eps_multiplier (int): Multiplier for pulse rate to set DBSCAN epsilon
                                 Default 5 = ~40-50 sec for typical tags
            lag_window (int): Time window in seconds for lag calculations
                             Default 9 seconds (kept for compatibility, not used in DBSCAN)
        """
        from sklearn.cluster import DBSCAN
        
        self.db = radio_project.db
        self.rec_id = rec_id
        self.eps_multiplier = eps_multiplier
        self.lag_window = lag_window
        self.tags = radio_project.tags
        
        # Load classified data for this receiver
        print(f"[bout] Loading classified data for {rec_id}")
        rec_dat = pd.read_hdf(self.db, 'classified', where=f'rec_id == "{rec_id}"')
        rec_dat = rec_dat[rec_dat.iter == rec_dat.iter.max()]
        rec_dat = rec_dat[rec_dat.test == 1]
        rec_dat = rec_dat[['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id']]
        rec_dat = rec_dat.astype({
            'freq_code': 'object',
            'epoch': 'float32',
            'time_stamp': 'datetime64[ns]',
            'power': 'float32',
            'rec_id': 'object'
        })
        
        # Clean up
        rec_dat.drop_duplicates(keep='first', inplace=True)
        rec_dat.sort_values(by=['freq_code', 'time_stamp'], inplace=True)
        
        self.data = rec_dat
        self.fishes = self.data.freq_code.unique()
        
        print(f"[bout] Loaded {len(self.data)} detections for {len(self.fishes)} fish")
        
        # Run DBSCAN bout detection immediately
        self._detect_bouts()
        
    def _detect_bouts(self):
        """
        Run DBSCAN temporal clustering to identify bouts.
        Called automatically during __init__.
        """
        from sklearn.cluster import DBSCAN
        import logging
        
        logger = logging.getLogger(__name__)
        
        print(f"[bout] Running DBSCAN bout detection for {self.rec_id}")
        presence_list = []
        
        for fish in self.fishes:
            fish_dat = self.data[self.data.freq_code == fish].copy()
            
            if len(fish_dat) == 0:
                continue
            
            # Get pulse rate for this fish
            try:
                pulse_rate = self.tags.loc[fish, 'pulse_rate']
            except (KeyError, AttributeError):
                pulse_rate = 8.0  # Default if not found
            
            # Calculate epsilon: pulse_rate * multiplier
            eps = pulse_rate * self.eps_multiplier
            
            # DBSCAN clustering on epoch (1D temporal data)
            epochs = fish_dat[['epoch']].values
            clustering = DBSCAN(eps=eps, min_samples=1).fit(epochs)
            fish_dat['bout_no'] = clustering.labels_
            
            # Filter out noise points (label = -1, though shouldn't happen with min_samples=1)
            fish_dat = fish_dat[fish_dat.bout_no != -1]
            
            # Assign to each detection
            for idx, row in fish_dat.iterrows():
                presence_list.append({
                    'freq_code': row['freq_code'],
                    'epoch': row['epoch'],
                    'time_stamp': row['time_stamp'],
                    'power': row['power'],
                    'rec_id': row['rec_id'],
                    'bout_no': row['bout_no'],
                    'class': 'study',
                    'det_lag': 0  # Not meaningful for DBSCAN, kept for compatibility
                })
        
        # Store results
        if presence_list:
            self.presence_df = pd.DataFrame(presence_list)
            print(f"[bout] Detected {self.presence_df.bout_no.nunique()} bouts across {len(self.fishes)} fish")
        else:
            self.presence_df = pd.DataFrame()
            print(f"[bout] No bouts detected for {self.rec_id}")
    
    def presence(self):
        """
        Write bout results to /presence table in HDF5.
        Call this after __init__ to save results to database.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.presence_df.empty:
            print(f"[bout] No presence data to write for {self.rec_id}")
            return
        
        # Prepare data for storage
        presence_df = self.presence_df.astype({
            'freq_code': 'object',
            'rec_id': 'object',
            'epoch': 'float32',
            'time_stamp': 'datetime64[ns]',
            'power': 'float32',
            'bout_no': 'int32',
            'class': 'object',
            'det_lag': 'int32'
        })
        
        # Write to HDF5
        with pd.HDFStore(self.db, mode='a') as store:
            store.append(
                key='presence',
                value=presence_df[['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'class', 'bout_no', 'det_lag']],
                format='table',
                data_columns=True,
                min_itemsize={'freq_code': 20, 'rec_id': 20, 'class': 20}
            )
        
        logger.debug(f"Wrote {len(presence_df)} detections ({presence_df.bout_no.nunique()} bouts) to /presence for {self.rec_id}")
        print(f"[bout] ✓ Wrote {len(presence_df)} detections to database")
    
    def visualize_bout_lengths(self, output_dir=None):
        """
        Visualize bout length distributions for this receiver.
        
        Creates comprehensive plots showing:
        - Overall distribution of bout lengths
        - Bout lengths by fish
        - Detections vs duration scatter
        - Cumulative distribution
        
        Args:
            output_dir (str): Directory to save plots. If None, uses database directory.
        """
        if self.presence_df.empty:
            print(f"[bout] No bout data to visualize for {self.rec_id}")
            return
        
        # Calculate bout summaries from presence data
        bout_summary = self.presence_df.groupby(['freq_code', 'bout_no']).agg({
            'epoch': ['min', 'max', 'count'],
            'power': 'mean'
        }).reset_index()
        bout_summary.columns = ['freq_code', 'bout_no', 'first_epoch', 'last_epoch', 'num_detections', 'mean_power']
        
        bouts = bout_summary.copy()
        bouts['bout_duration_sec'] = bouts['last_epoch'] - bouts['first_epoch']
        bouts['bout_duration_hrs'] = bouts['bout_duration_sec'] / 3600
        
        print(f"\n{'='*80}")
        print(f"BOUT LENGTH STATISTICS - {self.rec_id}")
        print(f"{'='*80}")
        
        # Overall statistics
        print(f"\nTotal bouts: {len(bouts):,}")
        print(f"\nBout duration statistics (hours):")
        print(bouts['bout_duration_hrs'].describe())
        
        # Count by duration categories
        short_bouts = bouts[bouts['bout_duration_sec'] < 300]
        medium_bouts = bouts[(bouts['bout_duration_sec'] >= 300) & (bouts['bout_duration_sec'] < 1800)]
        expected_bouts = bouts[(bouts['bout_duration_sec'] >= 1800) & (bouts['bout_duration_sec'] < 7200)]
        long_bouts = bouts[bouts['bout_duration_sec'] >= 7200]
        
        print(f"\nBouts < 5 minutes: {len(short_bouts):,} ({100*len(short_bouts)/len(bouts):.1f}%)")
        print(f"Bouts 5-30 minutes: {len(medium_bouts):,} ({100*len(medium_bouts)/len(bouts):.1f}%)")
        print(f"Bouts 30min-2hrs: {len(expected_bouts):,} ({100*len(expected_bouts)/len(bouts):.1f}%)")
        print(f"Bouts > 2 hours: {len(long_bouts):,} ({100*len(long_bouts)/len(bouts):.1f}%)")
        
        # Per-fish statistics
        print(f"\nBout statistics by fish:")
        fish_stats = bouts.groupby('freq_code').agg({
            'bout_duration_hrs': ['count', 'median', 'mean', 'max'],
            'num_detections': ['median', 'mean']
        }).round(2)
        fish_stats.columns = ['_'.join(col).strip() for col in fish_stats.columns.values]
        print(fish_stats.head(10))
        
        # Sample very short bouts
        if len(short_bouts) > 0:
            print(f"\nSample of very short bouts (< 5 minutes):")
            print(short_bouts[['freq_code', 'num_detections', 'bout_duration_sec', 'bout_duration_hrs']].head(10).to_string())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Bout Length Analysis - {self.rec_id}', fontsize=14, fontweight='bold')
        
        # 1. Overall distribution (log scale)
        ax = axes[0, 0]
        bouts['bout_duration_hrs'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
        ax.set_xlabel('Bout Duration (hours)', fontsize=10)
        ax.set_ylabel('Frequency (log scale)', fontsize=10)
        ax.set_title('Distribution of Bout Lengths', fontsize=11)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Zoomed in on short bouts (0-2 hours)
        ax = axes[0, 1]
        short_data = bouts[bouts['bout_duration_hrs'] <= 2]['bout_duration_hrs']
        short_data.hist(bins=40, ax=ax, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
        ax.set_xlabel('Bout Duration (hours)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Bout Lengths 0-2 Hours (Zoomed)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Detections per bout vs bout duration
        ax = axes[1, 0]
        sample = bouts.sample(min(5000, len(bouts)))  # Sample for performance
        ax.scatter(sample['num_detections'], sample['bout_duration_hrs'], 
                   alpha=0.4, s=20, color='darkgreen')
        ax.set_xlabel('Number of Detections per Bout', fontsize=10)
        ax.set_ylabel('Bout Duration (hours)', fontsize=10)
        ax.set_title('Detections vs Bout Duration', fontsize=11)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='30 min')
        ax.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='1 hour')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax = axes[1, 1]
        sorted_durations = np.sort(bouts['bout_duration_hrs'])
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
        ax.plot(sorted_durations, cumulative, linewidth=2, color='purple')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='30 min')
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='1 hour')
        ax.set_xlabel('Bout Duration (hours)', fontsize=10)
        ax.set_ylabel('Cumulative % of Bouts', fontsize=10)
        ax.set_title('Cumulative Distribution', fontsize=11)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if output_dir is None:
            output_dir = os.path.dirname(self.db)
        output_path = os.path.join(output_dir, f'bout_lengths_{self.rec_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n[bout] Saved visualization to: {output_path}")
        
        plt.show()
        
        return bouts


class overlap_reduction:
    """
    Resolve overlapping detections using signal quality comparison.
    
    Identifies and resolves spatially/temporally overlapping detections by comparing
    signal power and posterior probabilities. Marks weaker detections as overlapping
    to prevent spatial ambiguity in downstream statistical models.
    
    Resolution Logic:
    1. **Power Comparison**: If both detections have power, keep stronger signal
    2. **Posterior Comparison**: If both have classification scores, keep higher posterior
    3. **Ambiguous**: Mark if signals equal or criteria unavailable
    4. **Bout Conflicts**: Identify temporally overlapping bouts at different receivers
    
    Attributes
    ----------
    db : str
        Path to project HDF5 database
    project : object
        Radio project instance with database and metadata
    nodes : list
        List of receiver IDs (nodes in network)
    edges : list of tuples
        Directed edges representing receiver relationships
    G : networkx.DiGraph
        Network graph of receiver connections
    node_pres_dict : dict
        Presence data for each receiver (fish x bout)
    node_recap_dict : dict
        Recapture data for each receiver (detections)
    
    Methods
    -------
    unsupervised()
        Resolve overlaps using power/posterior comparison, apply bout spatial filter
    visualize_overlaps()
        Create 8-panel diagnostic plots for overlap analysis
    
    Notes
    -----
    - Operates on bout-level and detection-level simultaneously
    - Bout spatial filter identifies temporally overlapping bouts (≥50% overlap)
    - Power and posterior columns are optional (conditionally checked)
    - Results written to `/overlapping` table in HDF5 database
    - Visualization includes network structure, temporal patterns, power distributions
    
    Examples
    --------
    >>> from pymast.overlap_removal import overlap_reduction
    >>> overlap_obj = overlap_reduction(
    ...     nodes=['R01', 'R02', 'R03'],
    ...     edges=[('R01', 'R02'), ('R02', 'R03')],
    ...     radio_project=proj
    ... )
    >>> overlap_obj.unsupervised()
    >>> overlap_obj.visualize_overlaps()
    
    See Also
    --------
    bout : DBSCAN-based bout detection
    formatter.time_to_event : Uses overlap decisions for filtering
    """

    def __init__(self, nodes, edges, radio_project):
        """
        Initializes the OverlapReduction class.

        Args:
            nodes (list): List of nodes (receiver IDs) in the network.
            edges (list of tuples): Directed edges representing relationships between receivers.
            radio_project (object): Object representing the radio project, containing database path.

        This method reads and filters data from the project database for each node and stores 
        the processed data in dictionaries (`node_pres_dict` and `node_recap_dict`).
        """
        logger = logging.getLogger(__name__)
        logger.info("Initializing overlap_reduction")
        
        self.db = radio_project.db
        self.project = radio_project
        self.nodes = nodes
        self.edges = edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        # Initialize dictionaries for presence and recapture data
        self.node_pres_dict = {}
        self.node_recap_dict = {}
        
        logger.info(f"  Loading data for {len(nodes)} nodes")
        
        # Read and preprocess data for each node
        for node in tqdm(nodes, desc="Loading nodes", unit="node"):
            # Read data from the HDF5 database for the given node, applying filters using the 'where' parameter
            pres_where = f"rec_id == '{node}'"
            used_full_presence_read = False
            try:
                pres_data = pd.read_hdf(
                    self.db,
                    'presence',
                    columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'],
                    where=pres_where
                )
            except (TypeError, ValueError):
                # Some stores are fixed-format and don't support column selection — read entire table
                used_full_presence_read = True
                pres_data = pd.read_hdf(self.db, 'presence')

            # If we read zero rows, attempt a few fast alternate WHERE clauses that
            # handle common formatting differences (e.g. 'R02' vs '2' vs '02') before
            # performing a full-table in-memory fallback which is expensive.
            if len(pres_data) == 0 and not used_full_presence_read:
                tried_variants = []
                node_str = str(node)
                # generate candidate rec_id variants
                variants = []
                variants.append(node_str)
                if node_str.startswith(('R', 'r')):
                    variants.append(node_str[1:])
                # strip leading zeros
                variants.append(node_str.lstrip('0'))
                variants.append(node_str.lstrip('R').lstrip('0'))
                # numeric candidate
                try:
                    variants.append(str(int(''.join(filter(str.isdigit, node_str)))))
                except Exception:
                    pass
                # dedupe while preserving order
                seen = set()
                variants_clean = []
                for v in variants:
                    if not v:
                        continue
                    if v not in seen:
                        seen.add(v)
                        variants_clean.append(v)

                for cand in variants_clean:
                    tried_variants.append(cand)
                    alt_where = f"rec_id == '{cand}'"
                    try:
                        alt_pres = pd.read_hdf(
                            self.db,
                            'presence',
                            columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'],
                            where=alt_where
                        )
                        if len(alt_pres) > 0:
                            pres_data = alt_pres
                            logger.info("Node %s: found %d presence rows using alternate WHERE rec_id == '%s'", node, len(pres_data), cand)
                            break
                    except (TypeError, ValueError):
                        # column/where not supported on this store — give up trying alternates
                        logger.debug("Node %s: alternate WHERE '%s' not supported by store", node, alt_where)
                        break
                    except Exception:
                        # If this specific candidate failed, try the next
                        logger.debug("Node %s: alternate WHERE '%s' did not match", node, alt_where)
                        continue

            classified_where = f"(rec_id == '{node}') & (test == 1)"
            # Try to read classified with posterior columns if present
            try:
                recap_data = pd.read_hdf(
                    self.db,
                    'classified',
                    columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'iter', 'test', 'posterior_T', 'posterior_F'],
                    where=classified_where
                )
            except (KeyError, TypeError, ValueError):
                # Fallback: read the whole classified table for the node
                try:
                    recap_data = pd.read_hdf(self.db, 'classified')
                    recap_data = recap_data.query(classified_where)
                except Exception:
                    # If classified isn't available, try recaptures
                    try:
                        recap_data = pd.read_hdf(self.db, 'recaptures')
                        recap_data = recap_data.query(f"rec_id == '{node}'")
                    except Exception:
                        recap_data = pd.DataFrame()
        
            # Further filter recap_data for the max iteration
            recap_data = recap_data[recap_data['iter'] == recap_data['iter'].max()]

            # Group presence data by frequency code and bout, then calculate min, max, and median
            # Ensure presence has a 'power' column by merging power from the
            # classified/recaptures table when available. We don't change how
            # presence is originally created (bouts), we only attach power here
            # for downstream aggregation.
            if 'power' not in pres_data.columns and not recap_data.empty and 'power' in recap_data.columns:
                try:
                    pres_data = pres_data.merge(
                        recap_data[['freq_code', 'epoch', 'rec_id', 'power']],
                        on=['freq_code', 'epoch', 'rec_id'],
                        how='left'
                    )
                except Exception:
                    # If merge fails for any reason, continue without power —
                    # grouping will produce NaNs for median_power which is OK.
                    logger.debug('Could not merge power from recap_data into pres_data; continuing without power')

            # Group presence data by frequency code and bout, then calculate min, max, and median power
            # Check if power column exists first
            if 'power' in pres_data.columns:
                summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
                    min_epoch=('epoch', 'min'),
                    max_epoch=('epoch', 'max'),
                    median_power=('power', 'median')
                ).reset_index()
            else:
                logger.warning(f'Node {node}: Power column not found in presence data. Run bout detection first to populate power.')
                summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
                    min_epoch=('epoch', 'min'),
                    max_epoch=('epoch', 'max')
                ).reset_index()
                summarized_data['median_power'] = None
            
            # Log detailed counts so users can see raw vs summarized presence lengths
            raw_count = len(pres_data)
            summarized_count = len(summarized_data)
            logger.info(f"Node {node}: raw presence rows={raw_count}, summarized bouts={summarized_count}")

            # If we had to read the full presence table, warn (this can be slow and surprising)
            if used_full_presence_read:
                logger.warning(
                    "Node %s: had to read entire 'presence' table (fixed-format store); this may be slow and cause large raw counts. WHERE used: %s",
                    node,
                    pres_where,
                )

            # If counts are zero or unexpectedly large, include a small sample and the WHERE clause to help debug
            if raw_count == 0 or raw_count > 100000:
                try:
                    sample_head = pres_data.head(10).to_dict(orient='list')
                except Exception:
                    sample_head = '<unavailable>'
                logger.debug(
                    "Node %s: pres_data sample (up to 10 rows)=%s; WHERE=%s",
                    node,
                    sample_head,
                    pres_where,
                )

            # If we got zero rows from the column/where read, try a safe in-memory
            # fallback: read the full presence table and match rec_id after
            # normalizing (strip/upper). This can detect formatting mismatches
            # (e.g. numeric vs string rec_id, padding, whitespace).
            if raw_count == 0 and not used_full_presence_read:
                try:
                    logger.debug(
                        "Node %s: attempting in-memory fallback full-table read to find rec_id matches",
                        node,
                    )
                    full_pres = pd.read_hdf(self.db, 'presence')
                    if 'rec_id' in full_pres.columns:
                        node_norm = str(node).strip().upper()
                        full_pres['_rec_norm'] = full_pres['rec_id'].astype(str).str.strip().str.upper()
                        candidate = full_pres[full_pres['_rec_norm'] == node_norm]
                        if len(candidate) > 0:
                            # select expected columns if present
                            cols = [c for c in ['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'] if c in candidate.columns]
                            pres_data = candidate[cols].copy()
                            raw_count = len(pres_data)
                            used_full_presence_read = True
                            logger.info(
                                "Node %s: found %d presence rows after in-memory normalization of rec_id",
                                node,
                                raw_count,
                            )
                        else:
                            logger.debug("Node %s: in-memory full-table read did not find rec_id matches", node)
                    else:
                        logger.debug("Node %s: 'rec_id' column not present in full presence table", node)
                except Exception as e:
                    logger.debug("Node %s: in-memory fallback failed: %s", node, str(e))

            # Store the processed data in the dictionaries
            self.node_pres_dict[node] = summarized_data
            # Don't store recap_data - load on-demand per edge (memory efficient)
            self.node_recap_dict[node] = len(recap_data)  # Just track count
            logger.debug(f"  {node}: {len(pres_data)} presence records, {len(recap_data)} detections")
        
        logger.info(f"✓ Data loaded for {len(nodes)} nodes")

    def unsupervised_removal(self, method='posterior', p_value_threshold=0.05, effect_size_threshold=0.3, 
                            power_threshold=0.2, min_detections=3, bout_expansion=0):
        """
        Unsupervised overlap removal supporting multiple methods with statistical testing.

        Parameters
        ----------
        method : {'posterior', 'power'}
            'posterior' (default) uses `posterior_T` columns produced by the
            Naive Bayes classifier (recommended for radio telemetry).
            'power' compares median power in overlapping bouts (fallback).
        p_value_threshold : float, default=0.05
            Maximum p-value for t-test to consider difference statistically significant.
            Only applies when method='posterior'.
        effect_size_threshold : float, default=0.3
            Minimum Cohen's d effect size required (in addition to statistical significance).
            0.2 = small, 0.5 = medium, 0.8 = large effect. Lower values (0.3) are more
            conservative for radio telemetry where small differences matter.
        power_threshold : float
            Relative difference threshold for power-based decisions; computed
            as (parent_median - child_median) / max(parent_median, child_median).
        min_detections : int, default=3
            Minimum number of detections required in a bout for statistical comparison.
        bout_expansion : int, default=0
            Seconds to expand bout windows before/after (0 = no expansion, recommended
            for cleaner movement trajectories).
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting unsupervised overlap removal (method={method})")
        overlaps_processed = 0
        detections_marked = 0
        decisions = {'remove_parent': 0, 'remove_child': 0, 'keep_both': 0}
        skip_reasons = {'parent_too_small': 0, 'no_overlap': 0, 'child_too_small': 0, 
                       'no_posterior_data': 0, 'insufficient_after_nan': 0}

        # Precompute per-node, per-bout summaries (indices, posterior means, median power)
        # and build IntervalTrees per fish for fast overlap queries. This avoids
        # repeated mean()/median() computations inside the tight edge loops.
        node_bout_index = {}   # node -> fish -> list of bout dicts
        node_bout_trees = {}   # node -> fish -> IntervalTree
        node_recap_cache = {}  # node -> recap DataFrame (cache for edge loop)
        for node, bouts in self.node_pres_dict.items():
            # Load recap data and cache it for use in edge loop
            try:
                recaps = pd.read_hdf(self.db, 'classified', where=f"(rec_id == '{node}') & (test == 1)",
                                    columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'iter', 'test', 'posterior_T', 'posterior_F'])
                recaps = recaps[recaps['iter'] == recaps['iter'].max()]
                node_recap_cache[node] = recaps  # Cache for later use
            except Exception:
                recaps = pd.DataFrame()
                node_recap_cache[node] = pd.DataFrame()
            
            node_bout_index[node] = {}
            node_bout_trees[node] = {}
            if bouts.empty or recaps.empty:
                continue
            # ensure epoch dtype numeric for comparisons
            recaps_epoch = recaps['epoch']
            for fish_id, fish_bouts in bouts.groupby('freq_code'):
                r_fish = recaps[recaps['freq_code'] == fish_id]
                bout_list = []
                intervals = []
                for b_idx, bout_row in fish_bouts.reset_index(drop=True).iterrows():
                    min_epoch = bout_row['min_epoch']
                    max_epoch = bout_row['max_epoch']
                    if bout_expansion and bout_expansion > 0:
                        min_epoch = min_epoch - bout_expansion
                        max_epoch = max_epoch + bout_expansion

                    if not r_fish.empty:
                        mask = (r_fish['epoch'] >= min_epoch) & (r_fish['epoch'] <= max_epoch)
                        in_df = r_fish.loc[mask]
                        indices = in_df.index.tolist()
                        posterior = in_df['posterior_T'].mean(skipna=True) if 'posterior_T' in in_df.columns else np.nan
                        median_power = in_df['power'].median() if 'power' in in_df.columns else np.nan
                    else:
                        indices = []
                        posterior = np.nan
                        median_power = np.nan

                    bout_list.append({'min_epoch': min_epoch, 'max_epoch': max_epoch, 'indices': indices, 'posterior': posterior, 'median_power': median_power})
                    intervals.append((min_epoch, max_epoch, b_idx))

                node_bout_index[node][fish_id] = bout_list
                # build IntervalTree for this fish (only include intervals with numeric bounds)
                try:
                    tree = IntervalTree(Interval(int(a), int(b), idx) for (a, b, idx) in intervals if not (pd.isna(a) or pd.isna(b)))
                    node_bout_trees[node][fish_id] = tree
                except Exception:
                    node_bout_trees[node][fish_id] = IntervalTree()

        for edge_idx, (parent, child) in enumerate(tqdm(self.edges, desc="Processing edges", unit="edge")):
            logger.info(f"Edge {edge_idx+1}/{len(self.edges)}: {parent} → {child}")

            parent_bouts = self.node_pres_dict.get(parent, pd.DataFrame())
            
            # Use cached recap data but make a COPY to avoid cross-edge contamination
            # Without .copy(), the ambiguous_overlap column carries over between edges
            parent_dat = node_recap_cache.get(parent, pd.DataFrame()).copy()
            child_dat = node_recap_cache.get(child, pd.DataFrame()).copy()

            # Quick skip when any required table is empty
            if parent_bouts.empty or parent_dat.empty or child_dat.empty:
                logger.debug(f"Skipping {parent}->{child}: empty data")
                continue

            # Normalize freq_code dtype and pre-split recapture tables by freq_code
            # to avoid repeated full-DataFrame boolean comparisons inside loops.
            if 'freq_code' in parent_dat.columns:
                parent_dat['freq_code'] = parent_dat['freq_code'].astype('object')
            if 'freq_code' in child_dat.columns:
                child_dat['freq_code'] = child_dat['freq_code'].astype('object')

            parent_by_fish = {k: v for k, v in parent_dat.groupby('freq_code')} if not parent_dat.empty else {}
            child_by_fish = {k: v for k, v in child_dat.groupby('freq_code')} if not child_dat.empty else {}

            # Initialize overlapping and ambiguous_overlap columns fresh for each edge
            # This prevents carryover from previous edges
            parent_dat['overlapping'] = np.float32(0)
            parent_dat['ambiguous_overlap'] = np.float32(0)
            child_dat['overlapping'] = np.float32(0)
            child_dat['ambiguous_overlap'] = np.float32(0)

            fishes = parent_bouts['freq_code'].unique()
            logger.debug(f"  Processing {len(fishes)} fish for edge {parent}->{child}")
            print(f"  [overlap] {parent}→{child}: processing {len(fishes)} fish")

            # Buffers for indices to mark as overlapping or ambiguous for this edge
            parent_mark_idx = []
            child_mark_idx = []
            parent_ambiguous_idx = []
            child_ambiguous_idx = []

            for fish_idx, fish_id in enumerate(fishes, 1):
                # Progress update every 10 fish or for the last fish
                if fish_idx % 10 == 0 or fish_idx == len(fishes):
                    print(f"    [overlap] {parent}→{child}: fish {fish_idx}/{len(fishes)} ({fish_id})", end='\r')
                # fast access to precomputed bout lists and trees
                p_bouts = node_bout_index.get(parent, {}).get(fish_id, [])
                c_tree = node_bout_trees.get(child, {}).get(fish_id, IntervalTree())

                if not p_bouts or c_tree is None:
                    continue

                for p_i, p_info in enumerate(p_bouts):
                    p_indices = p_info['indices']
                    p_conf = p_info['posterior']
                    p_power = p_info['median_power']

                    # skip bouts with insufficient detections
                    if (not p_indices) or len(p_indices) < min_detections:
                        decisions['keep_both'] += 1
                        skip_reasons['parent_too_small'] += 1
                        continue

                    # query overlapping child bouts via IntervalTree
                    overlaps = c_tree.overlap(int(p_info['min_epoch']), int(p_info['max_epoch']))
                    if not overlaps:
                        decisions['keep_both'] += 1
                        skip_reasons['no_overlap'] += 1
                        continue

                    overlaps_processed += 1

                    for iv in overlaps:
                        c_idx = iv.data
                        try:
                            c_info = node_bout_index[child][fish_id][c_idx]
                        except Exception:
                            continue

                        c_indices = c_info['indices']
                        c_conf = c_info['posterior']
                        c_power = c_info['median_power']

                        # require minimum detections on both
                        if (not c_indices) or len(c_indices) < min_detections:
                            decisions['keep_both'] += 1
                            skip_reasons['child_too_small'] += 1
                            continue

                        if method == 'posterior':
                            # Statistical test approach: use t-test and Cohen's d on posterior_T
                            # Get actual posterior_T values for both receivers
                            p_posteriors = parent_dat.loc[p_indices, 'posterior_T'].values if 'posterior_T' in parent_dat.columns else []
                            c_posteriors = child_dat.loc[c_indices, 'posterior_T'].values if 'posterior_T' in child_dat.columns else []
                            
                            # Validate we have data
                            if len(p_posteriors) == 0 or len(c_posteriors) == 0:
                                decisions['keep_both'] += 1
                                skip_reasons['no_posterior_data'] += 1
                                continue
                            
                            # Remove NaN values
                            p_posteriors = p_posteriors[~np.isnan(p_posteriors)]
                            c_posteriors = c_posteriors[~np.isnan(c_posteriors)]
                            
                            if len(p_posteriors) < min_detections or len(c_posteriors) < min_detections:
                                decisions['keep_both'] += 1
                                skip_reasons['insufficient_after_nan'] += 1
                                continue
                            
                            # Perform Welch's t-test (unequal variances)
                            t_stat, p_value = ttest_ind(p_posteriors, c_posteriors, equal_var=False)
                            
                            # Calculate Cohen's d effect size
                            mean_diff = np.mean(p_posteriors) - np.mean(c_posteriors)
                            n1, n2 = len(p_posteriors), len(c_posteriors)
                            var1, var2 = np.var(p_posteriors, ddof=1), np.var(c_posteriors, ddof=1)
                            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2-2) > 0 else 1.0
                            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
                            
                            # Decision: require BOTH statistical significance AND meaningful effect size
                            if p_value < p_value_threshold and abs(cohens_d) >= effect_size_threshold:
                                if cohens_d > 0:  # parent has significantly higher posterior_T
                                    child_mark_idx.extend(c_indices)
                                    decisions['remove_child'] += 1
                                    detections_marked += len(c_indices)
                                else:  # child has significantly higher posterior_T
                                    parent_mark_idx.extend(p_indices)
                                    decisions['remove_parent'] += 1
                                    detections_marked += len(p_indices)
                            else:
                                # No significant difference - use combined score as tiebreaker
                                # Weighted combination: 70% posterior_T (classifier confidence) + 30% normalized power
                                # This accounts for both detection quality AND signal strength
                                p_mean_posterior = np.mean(p_posteriors)
                                c_mean_posterior = np.mean(c_posteriors)
                                
                                # Normalize power relative to each other (handles different receiver types)
                                if not pd.isna(p_power) and not pd.isna(c_power) and (p_power + c_power) > 0:
                                    p_norm_power = p_power / (p_power + c_power)
                                    c_norm_power = c_power / (p_power + c_power)
                                else:
                                    # Power not available, use equal weights
                                    p_norm_power = c_norm_power = 0.5
                                
                                # Combined score: 70% posterior_T, 30% power
                                p_score = 0.7 * p_mean_posterior + 0.3 * p_norm_power
                                c_score = 0.7 * c_mean_posterior + 0.3 * c_norm_power
                                
                                if p_score > c_score:
                                    child_mark_idx.extend(c_indices)
                                    decisions['remove_child'] += 1
                                    detections_marked += len(c_indices)
                                else:
                                    parent_mark_idx.extend(p_indices)
                                    decisions['remove_parent'] += 1
                                    detections_marked += len(p_indices)

                        elif method == 'power':
                            # Hierarchical decision tree with normalized power, posterior
                            # Step 1: Power → Step 2: Posterior → Step 3: Keep_both (ambiguous)
                            
                            # Initialize ambiguous flag for both bouts
                            p_ambiguous = 0
                            c_ambiguous = 0
                            
                            # Extract posterior from bout info
                            p_posterior = p_info.get('posterior', np.nan)
                            c_posterior = c_info.get('posterior', np.nan)
                            
                            # Get receiver info for power normalization
                            parent_rec = self.project.receivers.loc[parent]
                            child_rec = self.project.receivers.loc[child]
                            
                            # Step 1: Normalized power comparison
                            # Normalize: (power - min) / (max - min) where higher = stronger
                            # Use reasonable defaults if receiver stats not available
                            p_max = getattr(parent_rec, 'max_power', -40) if hasattr(parent_rec, 'max_power') else -40
                            p_min = getattr(parent_rec, 'min_power', -100) if hasattr(parent_rec, 'min_power') else -100
                            c_max = getattr(child_rec, 'max_power', -40) if hasattr(child_rec, 'max_power') else -40
                            c_min = getattr(child_rec, 'min_power', -100) if hasattr(child_rec, 'min_power') else -100
                            
                            if pd.isna(p_power) or pd.isna(c_power):
                                # Missing power data - try posterior
                                if not pd.isna(p_posterior) and not pd.isna(c_posterior):
                                    posterior_diff = p_posterior - c_posterior
                                    if abs(posterior_diff) > 0.1:  # 10% difference in classification confidence
                                        if posterior_diff > 0:
                                            # Parent has higher confidence - remove child
                                            child_mark_idx.extend(c_indices)
                                            decisions['remove_child'] += 1
                                            detections_marked += len(c_indices)
                                        else:
                                            # Child has higher confidence - remove parent
                                            parent_mark_idx.extend(p_indices)
                                            decisions['remove_parent'] += 1
                                            detections_marked += len(p_indices)
                                    else:
                                        # Both power and posterior missing/ambiguous - keep both
                                        p_ambiguous = 1
                                        c_ambiguous = 1
                                        decisions['keep_both'] += 1
                                else:
                                    # No data - keep both and mark as ambiguous
                                    p_ambiguous = 1
                                    c_ambiguous = 1
                                    decisions['keep_both'] += 1
                            else:
                                # Normalize power to 0-1 scale (1 = strongest)
                                p_norm = (p_power - p_min) / (p_max - p_min) if (p_max - p_min) != 0 else 0.5
                                c_norm = (c_power - c_min) / (c_max - c_min) if (c_max - c_min) != 0 else 0.5
                                
                                # Clamp to 0-1 range
                                p_norm = max(0.0, min(1.0, p_norm))
                                c_norm = max(0.0, min(1.0, c_norm))
                                
                                power_diff = p_norm - c_norm
                                
                                if power_diff > power_threshold:
                                    # Parent significantly stronger - remove child
                                    child_mark_idx.extend(c_indices)
                                    decisions['remove_child'] += 1
                                    detections_marked += len(c_indices)
                                    # Clear decision, not ambiguous
                                    p_ambiguous = 0
                                    c_ambiguous = 0
                                elif power_diff < -power_threshold:
                                    # Child significantly stronger - remove parent
                                    parent_mark_idx.extend(p_indices)
                                    decisions['remove_parent'] += 1
                                    detections_marked += len(p_indices)
                                    # Clear decision, not ambiguous
                                    p_ambiguous = 0
                                    c_ambiguous = 0
                                else:
                                    # Power is ambiguous - try Step 2: Posterior_T
                                    if not pd.isna(p_posterior) and not pd.isna(c_posterior):
                                        posterior_diff = p_posterior - c_posterior
                                        if abs(posterior_diff) > 0.1:  # 10% difference in classification confidence
                                            if posterior_diff > 0:
                                                # Parent has higher confidence - remove child
                                                child_mark_idx.extend(c_indices)
                                                decisions['remove_child'] += 1
                                                detections_marked += len(c_indices)
                                                p_ambiguous = 0
                                                c_ambiguous = 0
                                            else:
                                                # Child has higher confidence - remove parent
                                                parent_mark_idx.extend(p_indices)
                                                decisions['remove_parent'] += 1
                                                detections_marked += len(p_indices)
                                                p_ambiguous = 0
                                                c_ambiguous = 0
                                        else:
                                            # Both power and posterior ambiguous - keep both
                                            p_ambiguous = 1
                                            c_ambiguous = 1
                                            decisions['keep_both'] += 1
                                    else:
                                        # No posterior data - keep both and mark as ambiguous
                                        p_ambiguous = 1
                                        c_ambiguous = 1
                                        decisions['keep_both'] += 1
                            
                            # Store ambiguous flags in the dataframe
                            if p_ambiguous == 1:
                                parent_ambiguous_idx.extend(p_indices)
                            if c_ambiguous == 1:
                                child_ambiguous_idx.extend(c_indices)

                        else:
                            raise ValueError(f"Unknown method: {method}")

            # After processing all fish/bouts for this edge, bulk-assign overlapping flags
            print(f"\n  [overlap] {parent}→{child}: marking {len(set(parent_mark_idx))} parent + {len(set(child_mark_idx))} child detections as overlapping")
            print(f"  [overlap] {parent}→{child}: marking {len(set(parent_ambiguous_idx))} parent + {len(set(child_ambiguous_idx))} child detections as ambiguous")
            if parent_mark_idx:
                parent_dat.loc[sorted(set(parent_mark_idx)), 'overlapping'] = np.float32(1)
            if child_mark_idx:
                child_dat.loc[sorted(set(child_mark_idx)), 'overlapping'] = np.float32(1)

            # Bulk-assign ambiguous_overlap flags
            if parent_ambiguous_idx:
                parent_dat.loc[sorted(set(parent_ambiguous_idx)), 'ambiguous_overlap'] = np.float32(1)
            if child_ambiguous_idx:
                child_dat.loc[sorted(set(child_ambiguous_idx)), 'ambiguous_overlap'] = np.float32(1)

            # Write ONLY the marked detections (overlapping=1 OR ambiguous_overlap=1)
            # Combine overlapping and ambiguous indices (use set to avoid duplicates)
            parent_write_idx = sorted(set(parent_mark_idx + parent_ambiguous_idx))
            child_write_idx = sorted(set(child_mark_idx + child_ambiguous_idx))
            
            logger.debug(f"  Writing results for {parent} and {child} (parent overlapping={len(parent_mark_idx)}, ambiguous={len(parent_ambiguous_idx)}, child overlapping={len(child_mark_idx)}, ambiguous={len(child_ambiguous_idx)})")
            print(f"  [overlap] {parent}→{child}: writing overlapping detections to HDF5...")
            
            # Only write detections that were marked as overlapping or ambiguous
            if parent_write_idx:
                parent_overlapping = parent_dat.loc[parent_write_idx]
                ambig_count = (parent_overlapping['ambiguous_overlap'] == 1).sum()
                if ambig_count > 0:
                    print(f"  [overlap] {parent}→{child}: writing {ambig_count} parent ambiguous detections")
                self.write_results_to_hdf5(parent_overlapping)
            if child_write_idx:
                child_overlapping = child_dat.loc[child_write_idx]
                ambig_count = (child_overlapping['ambiguous_overlap'] == 1).sum()
                if ambig_count > 0:
                    print(f"  [overlap] {parent}→{child}: writing {ambig_count} child ambiguous detections")
                self.write_results_to_hdf5(child_overlapping)
            print(f"  [overlap] ✓ {parent}→{child} complete\n")

            # cleanup
            del parent_bouts, parent_dat, child_dat
            gc.collect()

        # Calculate statistics from HDF5 overlapping table
        logger.info("Calculating final statistics from overlapping table...")
        try:
            with pd.HDFStore(self.project.db, mode='r') as store:
                if '/overlapping' in store:
                    overlapping_table = store.select('overlapping')
                    total_written = len(overlapping_table)
                    overlapping_count = (overlapping_table['overlapping'] == 1).sum()
                    ambiguous_count = (overlapping_table['ambiguous_overlap'] == 1).sum()
                    unique_fish = overlapping_table['freq_code'].nunique()
                    unique_receivers = overlapping_table['rec_id'].nunique()
                else:
                    total_written = overlapping_count = ambiguous_count = unique_fish = unique_receivers = 0
        except Exception as e:
            logger.warning(f"Could not read overlapping table for statistics: {e}")
            total_written = overlapping_count = ambiguous_count = unique_fish = unique_receivers = 0

        print("\n" + "="*80)
        logger.info("✓ Unsupervised overlap removal complete")
        logger.info(f"  Overlapping bouts processed: {overlaps_processed}")
        logger.info(f"  Detections marked as overlapping: {detections_marked}")
        logger.info(f"  Decision breakdown: {decisions}")
        logger.info(f"  Skip reasons: {skip_reasons}")
        print(f"[overlap] ✓ Complete: {overlaps_processed} overlaps processed")
        print(f"[overlap] Decisions: remove_parent={decisions['remove_parent']}, remove_child={decisions['remove_child']}, keep_both={decisions['keep_both']}")
        print(f"[overlap] Skip breakdown: parent_too_small={skip_reasons['parent_too_small']}, child_too_small={skip_reasons['child_too_small']}, no_posterior={skip_reasons['no_posterior_data']}, insufficient_after_nan={skip_reasons['insufficient_after_nan']}")
        print(f"\n[overlap] Written to HDF5:")
        print(f"  Total detections in /overlapping table: {total_written:,}")
        print(f"  Detections with overlapping=1: {overlapping_count:,} ({100*overlapping_count/total_written if total_written > 0 else 0:.1f}%)")
        print(f"  Detections with ambiguous_overlap=1: {ambiguous_count:,} ({100*ambiguous_count/total_written if total_written > 0 else 0:.1f}%)")
        print(f"  Unique fish affected: {unique_fish}")
        print(f"  Unique receivers affected: {unique_receivers}")
        print("="*80)
        
        # Apply bout-based spatial filter to handle antenna bleed
        self._apply_bout_spatial_filter()

    def _apply_bout_spatial_filter(self, temporal_overlap_threshold=0.5):
        """
        Apply bout-based spatial logic filter to handle antenna bleed in overlapping table.
        
        When a fish has simultaneous bouts at multiple receivers (temporal overlap),
        keep the longer/stronger bout and mark the shorter bout as overlapping=1.
        
        This addresses the problem where powerhouse antennas detect fish on the "wrong"
        side due to back lobes, reflections, or diffraction.
        
        Parameters
        ----------
        temporal_overlap_threshold : float
            Fraction of temporal overlap required to consider bouts conflicting (0-1)
            Default 0.5 = 50% overlap
        """
        import logging
        logger = logging.getLogger(__name__)
        
        print(f"\n{'='*80}")
        print(f"BOUT-BASED SPATIAL FILTER")
        print(f"{'='*80}")
        print(f"[overlap] Resolving antenna bleed using bout strength...")
        
        try:
            # Read overlapping table
            overlapping_data = pd.read_hdf(self.db, key='/overlapping')
            
            if overlapping_data.empty:
                print(f"[overlap] No data in overlapping table")
                return
            
            # Get bout summaries from presence table
            presence_data = pd.read_hdf(self.db, key='/presence')
            
            if presence_data.empty or 'bout_no' not in presence_data.columns:
                print(f"[overlap] No bout data available, skipping spatial filter")
                return
            
            # Build bout summary: min/max epoch, detection count per bout
            bout_summary = presence_data.groupby(['freq_code', 'rec_id', 'bout_no']).agg({
                'epoch': ['min', 'max', 'count']
            }).reset_index()
            
            bout_summary.columns = ['freq_code', 'rec_id', 'bout_no', 'min_epoch', 'max_epoch', 'num_detections']
            bout_summary['bout_duration'] = bout_summary['max_epoch'] - bout_summary['min_epoch']
            
            print(f"[overlap] Loaded {len(bout_summary):,} bouts from {bout_summary['freq_code'].nunique()} fish")
            
            # Track which detections to mark as overlapping
            detections_to_mark = []  # List of (freq_code, rec_id, bout_no) tuples
            conflicts_found = 0
            
            # For each fish, check for temporally overlapping bouts at different receivers
            for fish in bout_summary['freq_code'].unique():
                fish_bouts = bout_summary[bout_summary['freq_code'] == fish].copy()
                
                if len(fish_bouts) < 2:
                    continue  # Can't have conflicts with only one bout
                
                # Compare all pairs of bouts for this fish
                for i, bout_a in fish_bouts.iterrows():
                    for j, bout_b in fish_bouts.iterrows():
                        if i >= j:  # Skip self-comparison and duplicates
                            continue
                        
                        # Only consider bouts at different receivers
                        if bout_a['rec_id'] == bout_b['rec_id']:
                            continue
                        
                        # Calculate temporal overlap
                        overlap_start = max(bout_a['min_epoch'], bout_b['min_epoch'])
                        overlap_end = min(bout_a['max_epoch'], bout_b['max_epoch'])
                        overlap_duration = max(0, overlap_end - overlap_start)
                        
                        # Calculate overlap as fraction of shorter bout
                        min_duration = min(bout_a['bout_duration'], bout_b['bout_duration'])
                        
                        if min_duration > 0:
                            overlap_fraction = overlap_duration / min_duration
                        else:
                            overlap_fraction = 0
                        
                        # If significant temporal overlap exists, we have a conflict
                        if overlap_fraction >= temporal_overlap_threshold:
                            conflicts_found += 1
                            
                            # Decide which bout to mark as overlapping based on:
                            # 1. Number of detections (primary - longer bout is more reliable)
                            # 2. Duration (secondary - longer time = more confidence)
                            
                            if bout_a['num_detections'] > bout_b['num_detections']:
                                # Keep A, mark B as overlapping
                                loser = (bout_b['freq_code'], bout_b['rec_id'], bout_b['bout_no'])
                                winner_rec = bout_a['rec_id']
                                loser_rec = bout_b['rec_id']
                                winner_dets = bout_a['num_detections']
                                loser_dets = bout_b['num_detections']
                            elif bout_b['num_detections'] > bout_a['num_detections']:
                                # Keep B, mark A as overlapping
                                loser = (bout_a['freq_code'], bout_a['rec_id'], bout_a['bout_no'])
                                winner_rec = bout_b['rec_id']
                                loser_rec = bout_a['rec_id']
                                winner_dets = bout_b['num_detections']
                                loser_dets = bout_a['num_detections']
                            else:
                                # Same detection count - use duration as tiebreaker
                                if bout_a['bout_duration'] > bout_b['bout_duration']:
                                    loser = (bout_b['freq_code'], bout_b['rec_id'], bout_b['bout_no'])
                                    winner_rec = bout_a['rec_id']
                                    loser_rec = bout_b['rec_id']
                                    winner_dets = bout_a['num_detections']
                                    loser_dets = bout_b['num_detections']
                                else:
                                    loser = (bout_a['freq_code'], bout_a['rec_id'], bout_a['bout_no'])
                                    winner_rec = bout_b['rec_id']
                                    loser_rec = bout_a['rec_id']
                                    winner_dets = bout_b['num_detections']
                                    loser_dets = bout_a['num_detections']
                            
                            detections_to_mark.append(loser)
                            logger.debug(f"  Fish {fish}: {winner_rec} ({winner_dets} dets) vs {loser_rec} ({loser_dets} dets, {overlap_fraction*100:.0f}% overlap) → Marking {loser_rec} as overlapping")
            
            # Mark conflicting bouts as overlapping=1 in overlapping table
            if len(detections_to_mark) > 0:
                # Need to join overlapping_data with presence to get bout_no
                overlapping_with_bouts = overlapping_data.merge(
                    presence_data[['freq_code', 'rec_id', 'epoch', 'bout_no']],
                    on=['freq_code', 'rec_id', 'epoch'],
                    how='left'
                )
                
                # Mark detections from losing bouts as overlapping=1
                initial_overlapping = (overlapping_with_bouts['overlapping'] == 1).sum()
                
                for fish, rec, bout in detections_to_mark:
                    mask = (
                        (overlapping_with_bouts['freq_code'] == fish) &
                        (overlapping_with_bouts['rec_id'] == rec) &
                        (overlapping_with_bouts['bout_no'] == bout)
                    )
                    overlapping_with_bouts.loc[mask, 'overlapping'] = 1
                
                final_overlapping = (overlapping_with_bouts['overlapping'] == 1).sum()
                newly_marked = final_overlapping - initial_overlapping
                
                # Drop bout_no before writing back (not needed in overlapping table)
                overlapping_with_bouts = overlapping_with_bouts.drop(columns=['bout_no'])
                
                # Write back to HDF5 (replace entire table)
                with pd.HDFStore(self.project.db, mode='a') as store:
                    # Remove old table
                    if '/overlapping' in store:
                        store.remove('overlapping')
                    
                    # Write updated table
                    store.append(
                        key='overlapping',
                        value=overlapping_with_bouts,
                        format='table',
                        data_columns=True,
                        min_itemsize={'freq_code': 20, 'rec_id': 20}
                    )
                
                print(f"\n[overlap] Bout spatial filter complete:")
                print(f"  Found {conflicts_found} temporal bout conflicts")
                print(f"  Marked {len(detections_to_mark)} conflicting bouts as overlapping")
                print(f"  Newly marked {newly_marked:,} detections ({newly_marked/len(overlapping_data)*100:.1f}%)")
                print(f"  Total overlapping detections: {final_overlapping:,} ({final_overlapping/len(overlapping_data)*100:.1f}%)")
                
                logger.info(f"Bout spatial filter marked {newly_marked} additional detections as overlapping")
            else:
                print(f"[overlap] No temporally overlapping bouts found across different receivers")
                logger.info("Bout spatial filter: no conflicts found")
        
        except Exception as e:
            logger.error(f"Error in bout spatial filter: {e}")
            print(f"[overlap] Error in bout spatial filter: {e}")
            import traceback
            traceback.print_exc()

    def nested_doll(self):
        """
        Identify and mark overlapping detections between parent and child nodes.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting nested_doll overlap detection")
        logger.info("  Method: Interval-based (conservative)")
        
        overlaps_found = False
        overlap_count = 0
        
        for i in tqdm(self.node_recap_dict, desc="Processing nodes", unit="node"):
            fishes = self.node_recap_dict[i].freq_code.unique()

            for j in fishes:
                children = list(self.G.successors(i))
                fish_dat = self.node_recap_dict[i][self.node_recap_dict[i].freq_code == j]
                fish_dat['overlapping'] = 0.0

                if len(children) > 0:
                    for k in children:
                        child_dat = self.node_pres_dict[k][self.node_pres_dict[k].freq_code == j]
                        if len(child_dat) > 0:
                            min_epochs = child_dat.min_epoch.values
                            max_epochs = child_dat.max_epoch.values
                            
                            fish_epochs = fish_dat.epoch.values
                            overlaps = np.any(
                                (min_epochs[:, None] <= fish_epochs) & (max_epochs[:, None] > fish_epochs), axis=0
                            )
                            overlap_indices = np.where(overlaps)[0]
                            if overlap_indices.size > 0:
                                overlaps_found = True
                                overlap_count += overlap_indices.size
                                fish_dat.loc[overlaps, 'overlapping'] = 1.0
                                #fish_dat.loc[overlaps, 'parent'] = i

                # fish_dat = fish_dat.astype({
                #     'freq_code': 'object',
                #     'epoch': 'int32',
                #     'rec_id': 'object',
                #     'overlapping': 'int32',
                # })
                fish_dat = fish_dat[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping']]
                self.write_results_to_hdf5(fish_dat)

                # with pd.HDFStore(self.db, mode='a') as store:
                #     store.append(key='overlapping',
                #                   value=fish_dat,
                #                   format='table',
                #                   index=False,
                #                   min_itemsize={'freq_code': 20,
                #                                 'rec_id': 20},
                #                   append=True,
                #                   data_columns=True,
                #                   chunksize=1000000)

        if overlaps_found:
            logger.info(f"✓ Nested doll complete")
            logger.info(f"  Total overlaps found: {overlap_count}")
        else:
            logger.info("✓ Nested doll complete - no overlaps found")

    def write_results_to_hdf5(self, df):
        """
        Writes the processed DataFrame to the HDF5 database.

        Args:
            df (DataFrame): The DataFrame containing processed detection data.
        
        The function appends data to the 'overlapping' table in the HDF5 database, ensuring 
        that each record is written incrementally to minimize memory usage.
        """
        logger = logging.getLogger(__name__)
        try:
            # Initialize ambiguous_overlap column if not present
            if 'ambiguous_overlap' not in df.columns:
                df['ambiguous_overlap'] = np.float32(0)
            
            # Determine which columns to write
            base_columns = ['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping', 'ambiguous_overlap']
            optional_columns = ['power', 'posterior_T', 'posterior_F']
            
            columns_to_write = base_columns.copy()
            for col in optional_columns:
                if col in df.columns:
                    columns_to_write.append(col)
            
            # Set data types for base columns
            dtype_dict = {
                'freq_code': 'object',
                'epoch': 'int32',
                'rec_id': 'object',
                'overlapping': 'int32',
                'ambiguous_overlap': 'float32',
            }
            
            # Add optional column types if present
            if 'power' in df.columns:
                dtype_dict['power'] = 'float32'
            if 'posterior_T' in df.columns:
                dtype_dict['posterior_T'] = 'float32'
            if 'posterior_F' in df.columns:
                dtype_dict['posterior_F'] = 'float32'
            
            df = df.astype(dtype_dict)
            
            with pd.HDFStore(self.project.db, mode='a') as store:
                store.append(
                    key='overlapping',
                    value=df[columns_to_write],
                    format='table',
                    data_columns=True,
                    min_itemsize={'freq_code': 20, 'rec_id': 20}
                )
            logger.debug(f"    Wrote {len(df)} detections to /overlapping (ambiguous: {df['ambiguous_overlap'].sum()})")
        except Exception as e:
            logger.error(f"Error writing to HDF5: {e}")
            raise




                
#     def _plot_kmeans_results(self, combined, centers, fish_id, node_a, node_b, project_dir):
#         """
#         Plots and saves the K-means clustering results to the project directory.
#         """
#         plt.figure(figsize=(10, 6))
#         plt.hist(combined['norm_power'], bins=30, alpha=0.5, label='Normalized Power')
#         plt.axvline(centers[0], color='r', linestyle='dashed', linewidth=2, label='Cluster Center 1')
#         plt.axvline(centers[1], color='b', linestyle='dashed', linewidth=2, label='Cluster Center 2')
#         plt.title(f"K-means Clustering between Nodes {node_a} and {node_b}")
#         plt.xlabel("Normalized Power")
#         plt.ylabel("Frequency")
#         plt.legend()

#         output_path = os.path.join(project_dir, 'Output', 'Figures', f'kmeans_nodes_{node_a}_{node_b}.png')
#         plt.savefig(output_path)
#         plt.close()
#         print(f"K-means plot saved")


            
# class overlap_reduction():
#     def __init__(self, nodes, edges, radio_project, n_clusters=2):
#         self.db = radio_project.db
#         self.G = nx.DiGraph()
#         self.G.add_edges_from(edges)
        
#         self.node_pres_dict = {}
#         self.node_recap_dict = {}
#         self.nodes = nodes
#         self.edges = edges
#         self.n_clusters = n_clusters

#         for node in nodes:
#             pres_data = dd.read_hdf(self.db, 'presence', columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'])
#             recap_data = dd.read_hdf(self.db, 'classified', columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'iter', 'test'])

#             pres_data['epoch'] = ((pres_data['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype('int64')
#             recap_data['epoch'] = ((recap_data['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype('int64')

#             pres_data = pres_data[pres_data['rec_id'] == node]
#             recap_data = recap_data[(recap_data['rec_id'] == node) & 
#                                     (recap_data['iter'] == recap_data['iter'].max()) & 
#                                     (recap_data['test'] == 1)]
#             recap_data = recap_data[['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id']]
            
#             pres_data = pres_data.compute()

#             summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg({
#                 'epoch': ['min', 'max'],
#                 'power': 'median'
#             }).reset_index()

#             summarized_data.columns = ['freq_code', 'bout_no', 'rec_id', 
#                                        'min_epoch', 'max_epoch', 'median_power']

#             rec_ids = summarized_data['rec_id'].values
#             median_powers = summarized_data['median_power'].values
#             normalized_power = np.zeros_like(median_powers)

#             for rec_id in np.unique(rec_ids):
#                 mask = rec_ids == rec_id
#                 norm_power = median_powers[mask]
#                 normalized_power[mask] = (norm_power - norm_power.min()) / (norm_power.max() - norm_power.min())

#             summarized_data['norm_power'] = normalized_power

#             self.node_pres_dict[node] = dd.from_pandas(summarized_data, npartitions=10)
#             self.node_recap_dict[node] = recap_data
#             print(f"Completed data management process for node {node}")

#         # Debugging step to check initialized keys
#         print("Initialized nodes in node_pres_dict:", list(self.node_pres_dict.keys()))
    

#     def unsupervised_removal(self):
#         final_classifications = {}
#         combined_recaps_list = []
    
#         def process_pair(parent, child):
#             parent_bouts = self.node_pres_dict[parent]
#             child_bouts = self.node_pres_dict[child]
    
#             overlapping = parent_bouts.merge(
#                 child_bouts,
#                 on='freq_code',
#                 suffixes=('_parent', '_child')
#             ).query('(min_epoch_child <= max_epoch_parent) & (max_epoch_child >= min_epoch_parent)').compute()
    
#             if overlapping.empty:
#                 return None
    
#             parent_recaps = self.node_recap_dict[parent].merge(
#                 overlapping[['freq_code', 'min_epoch_parent', 'max_epoch_parent']],
#                 on='freq_code'
#             ).query('epoch >= min_epoch_parent and epoch <= max_epoch_parent').compute()
    
#             child_recaps = self.node_recap_dict[child].merge(
#                 overlapping[['freq_code', 'min_epoch_child', 'max_epoch_child']],
#                 on='freq_code'
#             ).query('epoch >= min_epoch_child and epoch <= max_epoch_child').compute()
    
#             if parent_recaps.empty or child_recaps.empty:
#                 return None
    
#             combined_recaps = pd.concat([parent_recaps, child_recaps])
#             combined_recaps['norm_power'] = (combined_recaps['power'] - combined_recaps['power'].min()) / (combined_recaps['power'].max() - combined_recaps['power'].min())
#             return combined_recaps
    
#         # Process receiver pairs in parallel
#         with ProcessPoolExecutor() as executor:
#             results = executor.map(lambda pair: process_pair(pair[0], pair[1]), self.edges)
    
#         for combined_recaps in results:
#             if combined_recaps is not None:
#                 combined_recaps_list.append(combined_recaps)
    
#         if combined_recaps_list:
#             all_combined_recaps = pd.concat(combined_recaps_list, ignore_index=True)
#             best_bout_mask = self.apply_kmeans(all_combined_recaps)
    
#             all_combined_recaps['overlapping'] = np.where(best_bout_mask, 0, 1)
#             for _, rec in all_combined_recaps.iterrows():
#                 key = (rec['freq_code'], rec['epoch'])
#                 if key not in final_classifications:
#                     final_classifications[key] = rec['overlapping']
#                 else:
#                     final_classifications[key] = max(final_classifications[key], rec['overlapping'])
    
#         final_detections = []
#         for parent in self.node_pres_dict.keys():
#             recaps_chunk = self.node_recap_dict[parent].compute()
#             recaps_chunk['overlapping'] = 1
    
#             for (freq_code, epoch), overlap_value in final_classifications.items():
#                 recaps_chunk.loc[(recaps_chunk['epoch'] == epoch) & (recaps_chunk['freq_code'] == freq_code), 'overlapping'] = overlap_value
    
#             final_detections.append(recaps_chunk)
    
#         final_result = pd.concat(final_detections, ignore_index=True)
#         final_result['epoch'] = final_result['epoch'].astype('int64')
    
#         string_columns = final_result.select_dtypes(include=['string']).columns
#         final_result[string_columns] = final_result[string_columns].astype('object')
    
#         with pd.HDFStore(self.db, mode='a') as store:
#             store.append(key='overlapping',
#                          value=final_result,
#                          format='table',
#                          index=False,
#                          min_itemsize={'freq_code': 20, 'rec_id': 20},
#                          append=True,
#                          data_columns=True,
#                          chunksize=1000000)
    
#         print(f'Processed overlap for all receiver pairs.')

    
#     def apply_kmeans(self, combined_recaps):
#         """
#         Applies KMeans clustering to identify 'near' and 'far' clusters.
#         If KMeans cannot find two distinct clusters, falls back to a simple power comparison.
#         """
#         # Convert to NumPy arrays directly from the DataFrame
#         features = combined_recaps[['norm_power']].values
    
#         kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
#         kmeans.fit(features)
    
#         # Ensure labels are a NumPy array
#         labels = np.array(kmeans.labels_)
    
#         # Check if KMeans found fewer than 2 clusters
#         if len(np.unique(labels)) < 2:
#             print("Found fewer than 2 clusters. Falling back to selecting the recapture with the highest power.")
#             return combined_recaps['power'].values >= combined_recaps['power'].mean()
    
#         # Determine which cluster corresponds to 'near' based on median power
#         cluster_medians = combined_recaps.groupby(labels)['norm_power'].median()
#         near_cluster = cluster_medians.idxmax()  # Cluster with the higher median power is 'near'
    
#         return labels == near_cluster
   
#     # def unsupervised_removal(self):
#     #     """
#     #     Identifies and removes overlapping detections across receivers using KMeans for clustering.
#     #     Ensures each detection is classified only once, with the most conservative (i.e., 'far') classification.
#     #     """
#     #     final_classifications = {}
    
#     #     for parent, child in self.edges:
#     #         print(f"Processing parent: {parent}")
    
#     #         if parent not in self.node_pres_dict:
#     #             raise KeyError(f"Parent {parent} not found in node_pres_dict. Available keys: {list(self.node_pres_dict.keys())}")
    
#     #         parent_bouts = self.node_pres_dict[parent].compute()
#     #         child_bouts = self.node_pres_dict[child].compute()
    
#     #         # Merge and detect overlaps between parent and child
#     #         overlapping = parent_bouts.merge(
#     #             child_bouts,
#     #             on='freq_code',
#     #             suffixes=('_parent', '_child')
#     #         ).query('(min_epoch_child <= max_epoch_parent) & (max_epoch_child >= min_epoch_parent)')
    
#     #         if not overlapping.empty:
#     #             # Apply KMeans clustering or fallback to greater-than analysis
#     #             best_bout_mask = self.apply_kmeans(overlapping)
#     #             overlapping['overlapping'] = np.where(best_bout_mask, 0, 1)
    
#     #             # Update the final classification for each detection
#     #             for _, bout in overlapping.iterrows():
#     #                 key = (bout['freq_code'], bout['min_epoch_parent'], bout['max_epoch_parent'])
#     #                 if key not in final_classifications:
#     #                     final_classifications[key] = bout['overlapping']
#     #                 else:
#     #                     final_classifications[key] = max(final_classifications[key], bout['overlapping'])
    
#     #     # Prepare final result based on the most conservative classification
#     #     final_detections = []
#     #     for parent in self.node_pres_dict.keys():
#     #         recaps_chunk = self.node_recap_dict[parent].compute()
    
#     #         # Initialize 'overlapping' column as 1 (conservative)
#     #         recaps_chunk['overlapping'] = 1
    
#     #         # Update based on the final classifications
#     #         for (freq_code, min_epoch, max_epoch), overlap_value in final_classifications.items():
#     #             in_bout = (recaps_chunk['epoch'] >= min_epoch) & (recaps_chunk['epoch'] <= max_epoch) & (recaps_chunk['freq_code'] == freq_code)
#     #             recaps_chunk.loc[in_bout, 'overlapping'] = overlap_value
    
#     #         final_detections.append(recaps_chunk)
    
#     #     # Combine all detections
#     #     final_result = pd.concat(final_detections, ignore_index=True)
#     #     final_result['epoch'] = final_result['epoch'].astype('int64')
    
#     #     # Convert StringDtype columns to object dtype
#     #     string_columns = final_result.select_dtypes(include=['string']).columns
#     #     final_result[string_columns] = final_result[string_columns].astype('object')
    
#     #     # Save the final results to the HDF5 store
#     #     with pd.HDFStore(self.db, mode='a') as store:
#     #         store.append(key='overlapping',
#     #                       value=final_result,
#     #                       format='table',
#     #                       index=False,
#     #                       min_itemsize={'freq_code': 20, 'rec_id': 20},
#     #                       append=True,
#     #                       data_columns=True,
#     #                       chunksize=1000000)
    
#     #     print(f'Processed overlap for all receiver pairs.')
    
#     def _plot_kmeans_results(self, combined, centers, fish_id, node_a, node_b, project_dir):
#         """
#         Plots and saves the K-means clustering results to the project directory.
#         """
#         plt.figure(figsize=(10, 6))
#         plt.hist(combined['norm_power'], bins=30, alpha=0.5, label='Normalized Power')
#         plt.axvline(centers[0], color='r', linestyle='dashed', linewidth=2, label='Cluster Center 1')
#         plt.axvline(centers[1], color='b', linestyle='dashed', linewidth=2, label='Cluster Center 2')
#         plt.title(f"K-means Clustering between Nodes {node_a} and {node_b}")
#         plt.xlabel("Normalized Power")
#         plt.ylabel("Frequency")
#         plt.legend()

#         output_path = os.path.join(project_dir, 'Output', 'Figures', f'kmeans_nodes_{node_a}_{node_b}.png')
#         plt.savefig(output_path)
#         plt.close()
#         print(f"K-means plot saved")

#     def nested_doll(self):
#         """
#         Identify and mark overlapping detections between parent and child nodes.
#         """
#         overlaps_found = False
#         overlap_count = 0
        
#         for i in self.node_recap_dict:
#             fishes = self.node_recap_dict[i].freq_code.unique().compute()

#             for j in fishes:
#                 children = list(self.G.successors(i))
#                 fish_dat = self.node_recap_dict[i][self.node_recap_dict[i].freq_code == j].compute().copy()
#                 fish_dat['overlapping'] = 0
#                 fish_dat['parent'] = ''

#                 if len(children) > 0:
#                     for k in children:
#                         child_dat = self.node_pres_dict[k][self.node_pres_dict[k].freq_code == j].compute()
#                         if len(child_dat) > 0:
#                             min_epochs = child_dat.min_epoch.values
#                             max_epochs = child_dat.max_epoch.values
                            
#                             fish_epochs = fish_dat.epoch.values
#                             overlaps = np.any(
#                                 (min_epochs[:, None] <= fish_epochs) & (max_epochs[:, None] > fish_epochs), axis=0
#                             )
#                             overlap_indices = np.where(overlaps)[0]
#                             if overlap_indices.size > 0:
#                                 overlaps_found = True
#                                 overlap_count += overlap_indices.size
#                                 fish_dat.loc[overlaps, 'overlapping'] = 1
#                                 fish_dat.loc[overlaps, 'parent'] = i

#                 fish_dat = fish_dat.astype({
#                     'freq_code': 'object',
#                     'epoch': 'int32',
#                     'rec_id': 'object',
#                     'node': 'object',
#                     'overlapping': 'int32',
#                     'parent': 'object'
#                 })

#                 with pd.HDFStore(self.db, mode='a') as store:
#                     store.append(key='overlapping',
#                                  value=fish_dat,
#                                  format='table',
#                                  index=False,
#                                  min_itemsize={'freq_code': 20,
#                                                'rec_id': 20,
#                                                'parent': 20},
#                                  append=True,
#                                  data_columns=True,
#                                  chunksize=1000000)

#         if overlaps_found:
#             print(f"Overlaps were found and processed. Total number of overlaps: {overlap_count}.")
#         else:
#             print("No overlaps were found.")

    def visualize_overlaps(self, output_dir=None):
        """
        Visualize overlap patterns, decisions, and network structure.
        
        Creates comprehensive plots showing:
        - Network graph of receiver relationships with overlap counts
        - Decision breakdown (remove_parent, remove_child, keep_both)
        - Overlap distribution by receiver and fish
        - Temporal patterns of overlaps
        - Power distributions for overlapping vs non-overlapping detections
        
        Args:
            output_dir (str): Directory to save plots. If None, uses database directory.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        print(f"\n{'='*80}")
        print("OVERLAP VISUALIZATION")
        print(f"{'='*80}")
        
        # Load overlapping data
        try:
            overlapping = pd.read_hdf(self.db, key='/overlapping')
            print(f"Loaded {len(overlapping):,} detections from /overlapping table")
        except Exception as e:
            print(f"Error loading overlapping data: {e}")
            return
        
        if overlapping.empty:
            print("No overlap data to visualize")
            return
        
        # Calculate statistics
        total_detections = len(overlapping)
        overlapping_count = overlapping['overlapping'].sum() if 'overlapping' in overlapping.columns else 0
        ambiguous_count = overlapping['ambiguous_overlap'].sum() if 'ambiguous_overlap' in overlapping.columns else 0
        
        print(f"\nOverlap Statistics:")
        print(f"  Total detections: {total_detections:,}")
        print(f"  Overlapping detections: {overlapping_count:,} ({100*overlapping_count/total_detections:.1f}%)")
        print(f"  Ambiguous overlaps: {ambiguous_count:,} ({100*ambiguous_count/total_detections:.1f}%)")
        print(f"  Unique fish: {overlapping['freq_code'].nunique()}")
        print(f"  Unique receivers: {overlapping['rec_id'].nunique()}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Network graph showing receiver relationships
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_network_graph(ax1)
        
        # 2. Decision breakdown pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_decision_breakdown(ax2, overlapping)
        
        # 3. Overlaps by receiver (bar chart)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_overlaps_by_receiver(ax3, overlapping)
        
        # 4. Overlaps by fish (top 15)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_overlaps_by_fish(ax4, overlapping)
        
        # 5. Temporal pattern of overlaps
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_temporal_patterns(ax5, overlapping)
        
        # 6. Power distribution comparison
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_power_distributions(ax6, overlapping)
        
        # 7. Detection count per fish
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_detection_counts(ax7, overlapping)
        
        # 8. Overlap percentage by receiver pair
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_receiver_pair_heatmap(ax8, overlapping)
        
        fig.suptitle('Overlap Removal Analysis', fontsize=16, fontweight='bold')
        
        # Save figure
        if output_dir is None:
            output_dir = os.path.dirname(self.db)
        output_path = os.path.join(output_dir, 'overlap_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n[overlap] Saved visualization to: {output_path}")
        
        plt.show()
    
    def _plot_network_graph(self, ax):
        """Plot the receiver network graph with edge weights showing overlap counts."""
        pos = nx.spring_layout(self.G, seed=42, k=0.5, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.9, ax=ax)
        
        # Draw edges with varying thickness based on overlap count
        edges = self.G.edges()
        if len(edges) > 0:
            nx.draw_networkx_edges(self.G, pos, width=2, alpha=0.6, 
                                  edge_color='gray', arrows=True, 
                                  arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title('Receiver Network Structure', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_decision_breakdown(self, ax, overlapping):
        """Plot pie chart of overlap decisions."""
        # Count decisions from the data
        decisions = {
            'Overlapping': int(overlapping['overlapping'].sum()) if 'overlapping' in overlapping.columns else 0,
            'Ambiguous': int(overlapping['ambiguous_overlap'].sum()) if 'ambiguous_overlap' in overlapping.columns else 0,
            'Clean': int((overlapping['overlapping'] == 0).sum()) if 'overlapping' in overlapping.columns else len(overlapping)
        }
        
        # Filter out zero counts
        decisions = {k: v for k, v in decisions.items() if v > 0}
        
        if decisions:
            colors = {'Overlapping': '#ff6b6b', 'Ambiguous': '#ffd93d', 'Clean': '#6bcf7f'}
            ax.pie(decisions.values(), labels=decisions.keys(), autopct='%1.1f%%',
                   colors=[colors.get(k, 'gray') for k in decisions.keys()],
                   startangle=90)
            ax.set_title('Detection Categories', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Categories', fontsize=12, fontweight='bold')
    
    def _plot_overlaps_by_receiver(self, ax, overlapping):
        """Bar chart of overlap counts by receiver."""
        if 'overlapping' in overlapping.columns:
            overlap_by_rec = overlapping[overlapping['overlapping'] == 1].groupby('rec_id').size().sort_values(ascending=False)
            
            if len(overlap_by_rec) > 0:
                overlap_by_rec.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel('Receiver ID', fontsize=10)
                ax.set_ylabel('Overlapping Detections', fontsize=10)
                ax.set_title('Overlaps by Receiver', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No overlapping detections', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Overlaps by Receiver', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overlaps by Receiver', fontsize=12, fontweight='bold')
    
    def _plot_overlaps_by_fish(self, ax, overlapping):
        """Bar chart of overlap counts by fish (top 15)."""
        if 'overlapping' in overlapping.columns:
            overlap_by_fish = overlapping[overlapping['overlapping'] == 1].groupby('freq_code').size().sort_values(ascending=False).head(15)
            
            if len(overlap_by_fish) > 0:
                overlap_by_fish.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
                ax.set_xlabel('Overlapping Detections', fontsize=10)
                ax.set_ylabel('Fish ID', fontsize=10)
                ax.set_title('Top 15 Fish with Overlaps', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No overlapping detections', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Top 15 Fish with Overlaps', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Top 15 Fish with Overlaps', fontsize=12, fontweight='bold')
    
    def _plot_temporal_patterns(self, ax, overlapping):
        """Plot posterior ratio distributions to see if weaker classifications correlate with overlaps."""
        if 'posterior_T' in overlapping.columns and 'posterior_F' in overlapping.columns and 'overlapping' in overlapping.columns:
            # Calculate posterior ratio (T/F) - higher = stronger classification
            overlapping_copy = overlapping.copy()
            overlapping_copy['posterior_ratio'] = overlapping_copy['posterior_T'] / (overlapping_copy['posterior_F'] + 1e-10)
            
            overlap_ratio = overlapping_copy[overlapping_copy['overlapping'] == 1]['posterior_ratio'].dropna()
            clean_ratio = overlapping_copy[overlapping_copy['overlapping'] == 0]['posterior_ratio'].dropna()
            
            if len(overlap_ratio) > 0 and len(clean_ratio) > 0:
                # Use log scale for ratio
                overlap_log = np.log10(overlap_ratio + 1e-10)
                clean_log = np.log10(clean_ratio + 1e-10)
                
                ax.hist([clean_log, overlap_log], bins=30, label=['Clean', 'Overlapping'],
                       color=['lightblue', 'salmon'], alpha=0.7, edgecolor='black')
                ax.set_xlabel('log10(Posterior_T / Posterior_F)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title('Classification Strength: Overlapping vs Clean', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                median_overlap = np.median(overlap_log)
                median_clean = np.median(clean_log)
                ax.axvline(median_clean, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'Clean median: {median_clean:.2f}')
                ax.axvline(median_overlap, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Overlap median: {median_overlap:.2f}')
                ax.legend()
                
                print(f"\n[overlap] Posterior ratio analysis:")
                print(f"  Clean detections - median log10(T/F): {median_clean:.3f} (ratio: {10**median_clean:.2f})")
                print(f"  Overlapping detections - median log10(T/F): {median_overlap:.3f} (ratio: {10**median_overlap:.2f})")
                if median_overlap < median_clean:
                    print(f"  ✓ Overlapping detections have WEAKER classifications (lower T/F ratio)")
                else:
                    print(f"  ⚠ Overlapping detections do NOT have weaker classifications")
            else:
                ax.text(0.5, 0.5, 'Insufficient posterior data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Classification Strength', fontsize=12, fontweight='bold')
        else:
            # Fallback to temporal patterns if no posterior data
            if 'time_stamp' in overlapping.columns and 'overlapping' in overlapping.columns:
                overlap_data = overlapping[overlapping['overlapping'] == 1].copy()
                
                if len(overlap_data) > 0:
                    overlap_data['date'] = pd.to_datetime(overlap_data['time_stamp']).dt.date
                    daily_overlaps = overlap_data.groupby('date').size()
                    
                    daily_overlaps.plot(ax=ax, color='darkgreen', linewidth=2)
                    ax.set_xlabel('Date', fontsize=10)
                    ax.set_ylabel('Overlapping Detections', fontsize=10)
                    ax.set_title('Overlaps Over Time', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No overlapping detections', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Overlaps Over Time', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Overlaps Over Time', fontsize=12, fontweight='bold')
    
    def _plot_power_distributions(self, ax, overlapping):
        """Compare power distributions for overlapping vs non-overlapping detections."""
        if 'power' in overlapping.columns and 'overlapping' in overlapping.columns:
            overlap_power = overlapping[overlapping['overlapping'] == 1]['power'].dropna()
            clean_power = overlapping[overlapping['overlapping'] == 0]['power'].dropna()
            
            if len(overlap_power) > 0 and len(clean_power) > 0:
                ax.hist([clean_power, overlap_power], bins=30, label=['Clean', 'Overlapping'],
                       color=['lightblue', 'salmon'], alpha=0.7, edgecolor='black')
                ax.set_xlabel('Power (dB)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title('Power Distribution', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'Insufficient power data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Power Distribution', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No power data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Power Distribution', fontsize=12, fontweight='bold')
    
    def _plot_detection_counts(self, ax, overlapping):
        """Plot detection counts: total, overlapping, ambiguous."""
        categories = ['Total', 'Overlapping', 'Ambiguous', 'Clean']
        counts = [
            len(overlapping),
            int(overlapping['overlapping'].sum()) if 'overlapping' in overlapping.columns else 0,
            int(overlapping['ambiguous_overlap'].sum()) if 'ambiguous_overlap' in overlapping.columns else 0,
            int((overlapping['overlapping'] == 0).sum()) if 'overlapping' in overlapping.columns else len(overlapping)
        ]
        
        colors = ['#4a90e2', '#ff6b6b', '#ffd93d', '#6bcf7f']
        bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Detection Counts', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    
    def _plot_receiver_pair_heatmap(self, ax, overlapping):
        """Heatmap showing overlap percentages between receiver pairs."""
        if 'overlapping' in overlapping.columns:
            # This is simplified - would need parent-child relationship data for full heatmap
            overlap_by_rec = overlapping.groupby('rec_id').agg({
                'overlapping': 'sum',
                'freq_code': 'count'
            })
            overlap_by_rec['pct_overlap'] = 100 * overlap_by_rec['overlapping'] / overlap_by_rec['freq_code']
            
            receivers = overlap_by_rec.index.tolist()
            pct_values = overlap_by_rec['pct_overlap'].values
            
            if len(receivers) > 0:
                bars = ax.barh(receivers, pct_values, color='purple', alpha=0.7, edgecolor='black')
                ax.set_xlabel('% Detections Overlapping', fontsize=10)
                ax.set_ylabel('Receiver ID', fontsize=10)
                ax.set_title('Overlap % by Receiver', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Overlap % by Receiver', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overlap % by Receiver', fontsize=12, fontweight='bold')
