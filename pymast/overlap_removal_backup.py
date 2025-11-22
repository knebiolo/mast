# -*- coding: utf-8 -*-
'''
Module contains all of the methods and classes required to identify and remove
overlapping detections from radio telemetry data.
'''

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
    '''
    DBSCAN-based bout detection for a single receiver.
    
    Uses temporal clustering to identify continuous presence periods.
    Physics-based epsilon: pulse_rate * eps_multiplier (typically 5-10x pulse rate)
    '''
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
                value=presence_df[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'class', 'bout_no', 'det_lag']],
                format='table',
                data_columns=True,
                min_itemsize={'freq_code': 20, 'rec_id': 20, 'class': 20}
            )
        
        logger.debug(f"Wrote {len(presence_df)} detections ({presence_df.bout_no.nunique()} bouts) to /presence for {self.rec_id}")
        print(f"[bout] ✓ Wrote {len(presence_df)} detections to database")


from scipy.stats import ttest_ind


class overlap_reduction:
    """
            print("Please enter the values for the initial quantity (y0), the quantity at time t (yt), and the time t.")
            y0 = float(_prompt("Enter the initial quantity (y0): ", default=1.0))
            yt = float(_prompt("Enter the quantity at time t (yt): ", default=0.1))
            t = float(_prompt("Enter the time at which yt is observed (t): ", default=10.0))
        
            # Calculate decay rate b1 using the provided y0, yt, and t
            b1 = -np.log(yt / y0) / t
        
            # Assume that the decay rate b2 after the knot is the same as b1 before the knot
            # This is a simplifying assumption; you may want to calculate b2 differently based on additional data or domain knowledge
            b2 = b1
        
            # For the two-process model, we'll assume a1 is the initial quantity y0
            a1 = y0
        
            # We'll calculate a2 such that the function is continuous at the knot
            # This means solving the equation a1 * exp(-b1 * k) = a2 * exp(-b2 * k)
            # Since we've assumed b1 = b2, this simplifies to a2 = a1 * exp(-b1 * k)
            a2 = a1 * np.exp(-b1 * t)
        
            return [a1, b1, a2, b2, t]
        
        else:
            logger = logging.getLogger(__name__)
            logger.warning("Unsupported model type requested in prompt_for_params()")
            return None
        
    def find_knot(self, initial_knot_guess):
        # get lag frequencies
        lags = np.arange(0, self.time_limit, 2)
        freqs, bins = np.histogram(np.sort(self.data.lag_binned), lags)
        bins = bins[:-1][freqs > 0]  # Ensure the bins array is the right length
        freqs = freqs[freqs > 0]
        log_freqs = np.log(freqs)
        
        # Define a two-segment exponential decay function
        def two_exp_decay(x, a1, b1, a2, b2, k):
            condlist = [x < k, x >= k]
            funclist = [
                lambda x: a1 * np.exp(-b1 * x),
                lambda x: a2 * np.exp(-b2 * (x - k))
            ]
            return np.piecewise(x, condlist, funclist)


        # Objective function for two-segment model
        def objective_function(knot, bins, log_freqs):
            # Fit the model without bounds on the knot
            try:
                params, _ = curve_fit(lambda x, a1, b1, a2, b2: two_exp_decay(x, a1, b1, a2, b2, knot),
                                      bins, 
                                      log_freqs, 
                                      p0=[log_freqs[0], 0.001, 
                                          log_freqs[0], 0.001])
                y_fit = two_exp_decay(bins, *params, knot)
                error = np.sum((log_freqs - y_fit) ** 2)
                return error
            except RuntimeError:
                return np.inf

        # Use minimize to find the optimal knot
        result = minimize(
            fun=objective_function,
            x0=[initial_knot_guess],
            args=(bins, log_freqs),
            bounds=[(bins.min(), bins.max())]
        )

        # Check if the optimization was successful and extract the results
        if result.success:
            optimized_knot = result.x[0]

            # Refit with the optimized knot to get all parameters
            p0 = [log_freqs[0], 0.001, 
                  log_freqs[0], 0.001,
                  optimized_knot]
            
            bounds_lower = [0, 0, 
                            0, 0, 
                            bins.min()]
            
            bounds_upper = [np.inf, np.inf, 
                            np.inf, np.inf, 
                            bins.max()]

            optimized_params, _ = curve_fit(
                two_exp_decay,
                bins,
                log_freqs,
                p0=p0,
                bounds=(bounds_lower, bounds_upper)
            )

            # Visualization of the final fit with the estimated knot
            plt.figure(figsize=(12, 6))
            
            plt.scatter(bins, 
                        log_freqs, 
                        label='Data', 
                        alpha=0.6)
            
            x_range = np.linspace(bins.min(), bins.max(), 1000)
            
            plt.plot(x_range, 
                     two_exp_decay(x_range, *optimized_params),
                     label='Fitted function', 
                     color='red')
            
            plt.axvline(x=optimized_knot, color='orange', linestyle='--')#, label=f'Knot at x={optimized_knot:.2f}')
            plt.title('Fitted Two-Process Model')
            plt.xlabel('Lag Time')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            
            return optimized_params[-1]
        
        else:
            print("Optimization failed:", result.message)
            return None
    
    def find_knots(self, initial_knot_guesses):
        # get lag frequencies
        lags = np.arange(0, self.time_limit, 2)
        freqs, bins = np.histogram(np.sort(self.data.lag_binned), lags)
        bins = bins[:-1][freqs > 0]  # Ensure the bins array is the right length
        freqs = freqs[freqs > 0]
        log_freqs = np.log(freqs)
        # Assuming initial_knot_guesses is a list of two knot positions
        # This method should fit a three-process model

        # Define bounds for parameters outside of objective_function
        bounds_lower = [0, 0, 0, 0, 0, 0, bins.min(), initial_knot_guesses[0]]
        bounds_upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, initial_knot_guesses[1], bins.max()]
    
        # Define a three-segment exponential decay function
        #TODO - is the math correct?
        def three_exp_decay(x, a1, b1, a2, b2, a3, b3, k1, k2):
            condlist = [x < k1, (k1 <= x) & (x < k2), x >= k2]
            funclist = [
                lambda x: a1 * np.exp(-b1 * x),
                lambda x: a2 * np.exp(-b2 * (x - k1)),
                lambda x: a3 * np.exp(-b3 * (x - k2))
            ]
            return np.piecewise(x, condlist, funclist)
    
        # Objective function for three-segment model
        def objective_function(knots, bins, log_freqs):
            # Unpack knots
            k1, k2 = knots
            # Initial parameter guesses (3 amplitudes, 3 decay rates)
            p0 = [log_freqs[0], 0.001, log_freqs[0], 0.001, log_freqs[0], 0.001, k1, k2]

            # Fit the three-segment model
            params, _ = curve_fit(three_exp_decay,
                                  bins, 
                                  log_freqs,
                                  p0=p0, 
                                  bounds=(bounds_lower, bounds_upper))            # Calculate the errors
            #TODO - AIC or BIC possible expansion?
            y_fit = three_exp_decay(bins, *params)
            error = np.sum((log_freqs - y_fit) ** 2)
            return error
    
        # Perform the optimization with the initial guesses
        result = minimize(
            fun=objective_function,
            x0=initial_knot_guesses,  # Initial guesses for the knot positions
            args=(bins, log_freqs),
            bounds=[(bins.min(), initial_knot_guesses[1]),  # Ensure k1 < k2
                    (initial_knot_guesses[0], bins.max())]
        )

            
    
        # Check if the optimization was successful and extract the results
        if result.success:
            optimized_knots = result.x  # These should be the optimized knots
    
            # Now we refit the model with the optimized knots to get all the parameters
            p0 = [log_freqs[0], 0.001, 
                  log_freqs[0], 0.001, 
                  log_freqs[0], 0.001, 
                  optimized_knots[0], optimized_knots[1]]
            
            bounds_lower = [0, 0, 
                            0, 0, 
                            0, 0, 
                            bins.min(), optimized_knots[0]]
            
            bounds_upper = [np.inf, np.inf, 
                            np.inf, np.inf, 
                            np.inf, np.inf, 
                            optimized_knots[1], bins.max()]
    
            optimized_params, _ = curve_fit(
                three_exp_decay,
                bins,
                log_freqs,
                p0=p0,
                bounds=(bounds_lower, bounds_upper)
            )           

            # Visualization of the final fit with the estimated knots
            plt.figure(figsize=(12, 6))
            plt.scatter(bins, log_freqs, label='Data', alpha=0.6)

            # Create a range of x values for plotting the fitted function
            x_range = np.linspace(bins.min(), bins.max(), 1000)
            plt.plot(x_range, 
                     three_exp_decay(x_range, *optimized_params), 
                     label='Fitted function', color='red')

            plt.axvline(x=optimized_knots[0], color='orange', linestyle='--')
            plt.axvline(x=optimized_knots[1], color='green', linestyle='--')

            plt.title('Fitted Three-Process Model')
            plt.xlabel('Lag Time')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            plt.pause(5)
        else:
            print("Try again, optimization failed:", result.message)
            sys.exit()

        return optimized_params[-1]
    
    def fit_processes(self):
        # Step 1: Plot bins vs log frequencies
        lags = np.arange(0, self.time_limit, 2)
        freqs, bins = np.histogram(np.sort(self.data.lag_binned), lags)
        bins = bins[:-1][freqs > 0]  # Ensure the bins array is the right length
        freqs = freqs[freqs > 0]
        log_freqs = np.log(freqs)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(bins, log_freqs, label='Log of Frequencies', alpha=0.6)
        plt.title('Initial Data Plot')
        plt.xlabel('Bins')
        plt.ylabel('Log Frequencies')
        plt.legend()
        plt.show(block=True)
        plt.pause(5)
        # Step 2: Ask user for initial knots
        num_knots = int(_prompt("Enter the number of knots (1 for two-process, 2 for three-process): ", default=1))
        initial_knots = []
        for i in range(num_knots):
            knot = float(_prompt(f"Enter initial guess for knot {i+1}: ", default=5.0))
            initial_knots.append(knot)
    
        # Step 3 & 4: Determine the number of processes and fit accordingly
        if num_knots == 1:
            # Fit a two-process model
            self.initial_knot_guess = initial_knots[0]
            self.find_knot(self.initial_knot_guess)
        elif num_knots == 2:
            # Fit a three-process model (you will need to implement this method)
            self.initial_knot_guesses = initial_knots
            self.find_knots(self.initial_knot_guesses)
        else:
            print("Invalid number of knots. Please enter 1 or 2.")
            
        # Fit the model based on the number of knots
        optimized_knots = None
        if num_knots == 1:
            # Fit a two-process model
            optimized_knots = self.find_knot(initial_knots[0])
        elif num_knots == 2:
            # Fit a three-process model
            optimized_knots = self.find_knots(initial_knots)
        else:
            print("Invalid number of knots. Please enter 1 or 2.")
        
        # Return the optimized knot(s)
        return optimized_knots
      
    def presence(self, threshold):
        '''Function takes the break point between a continuous presence and new presence,
        enumerates the presence number at a receiver and writes the data to the
        analysis database.'''
        response = _prompt('Satisfied with the bout fitting?', default='yes')
        if str(response) in ['yes', 'y', 'T', 'True']:

            fishes = self.data.freq_code.unique()

            for fish in fishes:
                fish_dat = self.data[self.data.freq_code == fish]

                # Vectorized classification
                classifications = np.where(fish_dat.det_lag <= threshold, 'within_bout', 'start_new_bout')

                # Generating bout numbers
                # Increment bout number each time a new bout starts
                bout_changes = np.where(classifications == 'start_new_bout', 1, 0)
                bout_no = np.cumsum(bout_changes)

                # Assigning classifications and bout numbers to the dataframe
                fish_dat['class'] = classifications
                fish_dat['bout_no'] = bout_no

                fish_dat = fish_dat.astype({'freq_code': 'object',
                                            # 'epoch': 'float64',
                                            'epoch': 'float32',
                                            'time_stamp': 'datetime64[ns]',
                                            'power': 'float32',
                                            'rec_id': 'object',
                                            'class': 'object',
                                            'bout_no': 'int32',
                                            'det_lag': 'int32'})

                # append to hdf5
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(key='presence',
                                 value=fish_dat,
                                 format='table',
                                 index=False,
                                 min_itemsize={'freq_code': 20,
                                               'rec_id': 20,
                                               'class': 20},
                                 append=True,
                                 data_columns=True,
                                 chunksize=1000000)

                logger = logging.getLogger(__name__)
                logger.info('bouts classified for fish %s', fish)
        else:
            logger = logging.getLogger(__name__)
            logger.error('Give the fitting another try')
            sys.exit()
 
# class overlap_reduction():
#     """
#     Class to reduce redundant detections at overlapping receivers.

#     Attributes:
#         db (str): Path to the project database.
#         G (networkx.DiGraph): Directed graph representing the relationships between receivers.
#         node_pres_dict (dict): Dictionary storing presence data for each node (receiver).
#         node_recap_dict (dict): Dictionary storing recapture data for each node (receiver).
#     """

#     def __init__(self, nodes, edges, radio_project):
#         """
#         Initializes the OverlapReduction class by creating a directed graph and
#         importing the necessary data for each node (receiver).
#         """
#         self.db = radio_project.db
#         self.project = radio_project
#         # Create a directed graph from the list of edges
#         self.G = nx.DiGraph()
#         self.G.add_edges_from(edges)
        
#         # Initialize dictionaries for presence and recap data
#         self.node_pres_dict = {}
#         self.node_recap_dict = {}
#         self.nodes = nodes
#         self.edges = edges
        
#         for node in nodes:
#             # Load the entire dataset or relevant columns, and then filter it in memory
#             pres_data = dd.read_hdf(self.db, 'presence', columns=['freq_code', 'epoch','time_stamp', 'power', 'rec_id', 'bout_no'])
#             recap_data = dd.read_hdf(self.db, 'classified', columns=['freq_code', 'epoch','time_stamp', 'power', 'rec_id', 'iter', 'test'])
#             pres_data['epoch'] = (pres_data['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#             recap_data['epoch'] = (recap_data['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

#             # Filter the data for the specific node
#             pres_data = pres_data[pres_data['rec_id'] == node]
#             recap_data = recap_data[recap_data['rec_id'] == node]

#             # We only want good data in our recaps
#             recap_data = recap_data[recap_data['iter'] == recap_data['iter'].max()]
#             recap_data = recap_data[recap_data['test'] == 1]
#             recap_data = recap_data[['freq_code', 'epoch','time_stamp', 'power', 'rec_id']]            
            
#             # Grouping with shuffle-based aggregation
#             summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg({
#                 'epoch': ['min', 'max'],
#                 'power': 'median'
#             }, shuffle='tasks').reset_index()

#             # Flatten multi-index columns
#             summarized_data.columns = ['freq_code', 'bout_no', 'rec_id', 
#                                         'min_epoch', 'max_epoch', 'median_power']
            
#             # normalize power
#             recap_data['norm_power'] = (recap_data.power - recap_data.power.min()) / (recap_data.power.max() - recap_data.power.min())
#             summarized_data['norm_power'] = (summarized_data.median_power - summarized_data.median_power.min()) / \
#                 (summarized_data.median_power.max() - summarized_data.median_power.min())

#             self.node_pres_dict[node] = summarized_data
#             self.node_recap_dict[node] = recap_data
#             print(f"Completed data management process for node {node}")

#         #self.visualize_graph()

#         # Visualize the graph
#         #self.visualize_graph()

#     def visualize_graph(self):
#         """
#         Visualizes the directed graph representing the relationships between nodes.
#         """
#         pos = nx.circular_layout(self.G)
#         nx.draw(self.G, pos, node_color='r', node_size=400, with_labels=True)
#         plt.axis('off')
#         plt.show()

#     def unsupervised_removal(self):
#         for i in self.edges:
#             parent = i[0]
#             child = i[1]
            
#             # get bout data
#             parent_bouts = self.node_pres_dict[parent]
#             child_bouts = self.node_pres_dict[child]
            
#             # get parent data to label
#             parent_dat = self.node_recap_dict[parent]
            
#             if child_bouts.shape[0].compute() == 0:
#                 continue
    
#             print (f'Identifying overlapping bouts between {parent} and {child}')
            
#             # 1. Perform an inner merge on 'freq_code'
#             merged_ddf = dd.merge(parent_bouts, child_bouts, on='freq_code', suffixes=('_parent', '_child'))
            
#             # 2. Define a function to check for overlaps
#             def check_overlap(row):
#                 return max(row['min_epoch_parent'], row['min_epoch_child']) <= min(row['max_epoch_parent'], row['max_epoch_child'])
            
#             # 3. Apply overlap check row-wise
#             merged_ddf['overlap'] = merged_ddf.apply(check_overlap, axis=1, meta=('overlap', 'bool'))
            
#             # 4. Filter rows where overlap is True
#             overlap_ddf = merged_ddf[merged_ddf['overlap']]
            
#             # *** Delete intermediate object 'merged_ddf' since it's no longer needed ***
#             del merged_ddf
#             gc.collect()
            
#             print ('Creating intermediate dataframes and removing duplicates created in previous step')
    
#             # get overlapping parent and child dataframes
#             ovlp_parent_bouts = overlap_ddf[['freq_code','bout_no_parent','rec_id_parent', 'min_epoch_parent','max_epoch_parent','norm_power_parent']]
#             ovlp_parent_bouts = ovlp_parent_bouts.rename(columns={
#                 'bout_no_parent':'bout_no', 'rec_id_parent':'rec_id',
#                 'min_epoch_parent':'min_epoch', 'max_epoch_parent':'max_epoch',
#                 'norm_power_parent':'norm_power'})
#             ovlp_parent_bouts = ovlp_parent_bouts.drop_duplicates(subset=['freq_code', 'bout_no'])
    
#             ovlp_child_bouts = overlap_ddf[['freq_code','bout_no_child','rec_id_child', 'min_epoch_child','max_epoch_child','norm_power_child']]
#             ovlp_child_bouts = ovlp_child_bouts.rename(columns={
#                 'bout_no_child':'bout_no', 'rec_id_child':'rec_id',
#                 'min_epoch_child':'min_epoch', 'max_epoch_child':'max_epoch',
#                 'norm_power_child':'norm_power'})
#             ovlp_child_bouts = ovlp_child_bouts.drop_duplicates(subset=['freq_code', 'bout_no'])
    
#             # *** Delete 'overlap_ddf' after use ***
#             del overlap_ddf
#             gc.collect()
    
#             # get power and create an array for processing with gmm
#             parent_column = ovlp_parent_bouts['norm_power']
#             child_column = ovlp_child_bouts['norm_power']
    
#             combined_columns = dd.concat([parent_column, child_column])
#             combined_array = combined_columns.to_dask_array(lengths=True)
#             combined_array = combined_array.rechunk({0: 10000})
    
#             # *** Delete 'ovlp_parent_bouts' and 'ovlp_child_bouts' after their usage ***
#             del ovlp_parent_bouts, ovlp_child_bouts
#             gc.collect()
    
#             print ('Performing Gaussian Mixture to identify differences between detection power')
#             try:
#                 gmm = GaussianMixture(n_components=2)
#                 gmm.fit(combined_array.reshape(-1, 1))
    
#                 means = gmm.means_.flatten()
#                 sorted_means = np.sort(means)
#                 split_point = (sorted_means[0] + sorted_means[1]) / 2
#                 print(f'Split point: {split_point}')
                
#                 # *** Delete 'combined_array' after GaussianMixture ***
#                 del combined_array
#                 gc.collect()
                
#             except ValueError:
#                 print(f'GMM failed')
#                 split_point = 0.0
#                 continue
            
#             print ('Classifying rows in parent dataframe')
#             parent_bouts['overlapping_bout'] = parent_bouts['norm_power'].apply(
#                 lambda x: 1 if x < split_point else 0, meta=('classification', 'int')
#             )
    
#             # Step 1: Merge on 'freq_code'
#             merged_df = dd.merge(parent_dat, parent_bouts, on='freq_code', how='left')
    
#             # *** Delete 'parent_dat' and 'parent_bouts' after merging ***
#             del parent_dat, parent_bouts
#             gc.collect()
    
#             # Step 2: Filter where 'epoch' falls within the 'min_epoch' and 'max_epoch'
#             merged_df['in_bout'] = (merged_df['epoch'] >= merged_df['min_epoch']) & (merged_df['epoch'] <= merged_df['max_epoch'])
    
#             # Step 3: Classify the detections
#             merged_df['overlapping'] = merged_df['overlapping_bout'].where(merged_df['in_bout'], other=None)
    
#             # Step 1: Keep necessary columns
#             merged_df = merged_df[['freq_code', 'epoch', 'time_stamp', 'rec_id_x', 'in_bout', 'overlapping']]
#             merged_df = merged_df.rename(columns={'rec_id_x': 'rec_id'})
    
#             # Step 2: Filter rows
#             merged_df = merged_df[merged_df['in_bout'] == True]
    
#             # Step 3: Convert string columns to object-type strings
#             merged_df['freq_code'] = merged_df['freq_code'].astype('object')
#             merged_df['rec_id'] = merged_df['rec_id'].astype('object')
    
#             # Repartition the DataFrame
#             merged_df = merged_df.repartition(npartitions=500)
    
#             # Write each partition to HDF5
#             for i, partition in enumerate(merged_df.partitions):
#                 partition_df = partition.compute()
#                 partition_df['freq_code'] = partition_df['freq_code'].astype('object')
#                 partition_df['rec_id'] = partition_df['rec_id'].astype('object')
    
#                 with pd.HDFStore(self.project.db, mode='a') as store:
#                     store.append(
#                         key='overlapping',
#                         value=partition_df,
#                         format='table',
#                         index=False,
#                         min_itemsize={'freq_code': 20, 'rec_id': 20},
#                         data_columns=True
#                     )
#                 print(f"Partition {i + 1} written to HDF5.")
            
#             # *** Delete 'merged_df' after writing to HDF5 ***
#             del merged_df
#             gc.collect()


from scipy.stats import ttest_ind


class overlap_reduction:
    """
    A class to manage and reduce redundant detections at overlapping radio receivers.
    
    The class processes data from multiple receivers, identifies overlapping 
    detections, and applies statistical tests to determine which receiver has 
    the higher signal strength for a given animal.
    
    Attributes:
        db (str): Path to the project database.
        project (object): An object representing the radio project, providing access to the database.
        nodes (list): A list of nodes (receivers) in the network.
        edges (list of tuples): Directed edges representing the relationships between nodes.
        node_pres_dict (dict): Dictionary storing processed presence data for each node (receiver).
        node_recap_dict (dict): Dictionary storing processed recapture data for each node (receiver).
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
            summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
                min_epoch=('epoch', 'min'),
                max_epoch=('epoch', 'max'),
                median_power=('power', 'median')
            ).reset_index()
            
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
            self.node_recap_dict[node] = recap_data
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
        for node, bouts in self.node_pres_dict.items():
            recaps = self.node_recap_dict.get(node, pd.DataFrame())
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
            parent_dat = self.node_recap_dict.get(parent, pd.DataFrame()).copy()
            child_dat = self.node_recap_dict.get(child, pd.DataFrame()).copy()

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

            # Initialize overlapping columns if missing
            if 'overlapping' not in parent_dat.columns:
                parent_dat['overlapping'] = np.float32(0)
            if 'overlapping' not in child_dat.columns:
                child_dat['overlapping'] = np.float32(0)
            
            # Initialize ambiguous_overlap columns if missing
            if 'ambiguous_overlap' not in parent_dat.columns:
                parent_dat['ambiguous_overlap'] = np.float32(0)
            if 'ambiguous_overlap' not in child_dat.columns:
                child_dat['ambiguous_overlap'] = np.float32(0)

            fishes = parent_bouts['freq_code'].unique()
            logger.debug(f"  Processing {len(fishes)} fish for edge {parent}->{child}")
            print(f"  [overlap] {parent}→{child}: processing {len(fishes)} fish")

            # Buffers for indices to mark as overlapping for this edge
            parent_mark_idx = []
            child_mark_idx = []

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
                                for idx in p_indices:
                                    parent_dat.loc[idx, 'ambiguous_overlap'] = np.float32(1)
                            if c_ambiguous == 1:
                                for idx in c_indices:
                                    child_dat.loc[idx, 'ambiguous_overlap'] = np.float32(1)

                        else:
                            raise ValueError(f"Unknown method: {method}")

            # After processing all fish/bouts for this edge, bulk-assign overlapping flags
            print(f"\n  [overlap] {parent}→{child}: marking {len(set(parent_mark_idx))} parent + {len(set(child_mark_idx))} child detections as overlapping")
            if parent_mark_idx:
                parent_dat.loc[sorted(set(parent_mark_idx)), 'overlapping'] = np.float32(1)
            if child_mark_idx:
                child_dat.loc[sorted(set(child_mark_idx)), 'overlapping'] = np.float32(1)

            # Write ONLY the marked detections (not all data for this receiver pair)
            logger.debug(f"  Writing results for {parent} and {child} (parent marks={len(parent_mark_idx)}, child marks={len(child_mark_idx)})")
            print(f"  [overlap] {parent}→{child}: writing overlapping detections to HDF5...")
            
            # Only write detections that were marked as overlapping
            if parent_mark_idx:
                parent_overlapping = parent_dat.loc[sorted(set(parent_mark_idx))]
                self.write_results_to_hdf5(parent_overlapping)
            if child_mark_idx:
                child_overlapping = child_dat.loc[sorted(set(child_mark_idx))]
                self.write_results_to_hdf5(child_overlapping)
            print(f"  [overlap] ✓ {parent}→{child} complete\n")

            # cleanup
            del parent_bouts, parent_dat, child_dat
            gc.collect()

        print("\n" + "="*80)
        logger.info("✓ Unsupervised overlap removal complete")
        logger.info(f"  Overlapping bouts processed: {overlaps_processed}")
        logger.info(f"  Detections marked as overlapping: {detections_marked}")
        logger.info(f"  Decision breakdown: {decisions}")
        logger.info(f"  Skip reasons: {skip_reasons}")
        print(f"[overlap] ✓ Complete: {overlaps_processed} overlaps processed, {detections_marked} detections marked")
        print(f"[overlap] Decisions: remove_parent={decisions['remove_parent']}, remove_child={decisions['remove_child']}, keep_both={decisions['keep_both']}")
        print(f"[overlap] Skip breakdown: parent_too_small={skip_reasons['parent_too_small']}, child_too_small={skip_reasons['child_too_small']}, no_posterior={skip_reasons['no_posterior_data']}, insufficient_after_nan={skip_reasons['insufficient_after_nan']}")
        print("="*80)

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
                fish_dat['overlapping'] = np.float32(0)

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
                                fish_dat.loc[overlaps, 'overlapping'] = np.float32(1)
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
            
            df = df.astype({
                'freq_code': 'object',
                'epoch': 'int32',
                'rec_id': 'object',
                'overlapping': 'int32',
                'ambiguous_overlap': 'float32',
            })
            with pd.HDFStore(self.project.db, mode='a') as store:
                store.append(
                    key='overlapping',
                    value=df[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping', 'ambiguous_overlap']],
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