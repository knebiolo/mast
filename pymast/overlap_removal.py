# -*- coding: utf-8 -*-
'''
Module contains all of the methods and classes required to identify and remove
overlapping detections from radio telemetry data.
'''

# import modules required for function dependencies
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

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
from dask_ml.cluster import KMeans
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

class bout():
    '''Python class object to delineate when bouts occur at receiver.'''
    def __init__ (self, radio_project, node, lag_window, time_limit):
        self.lag_window = lag_window
        self.time_limit = time_limit
        self.db = radio_project.db

        # get the receivers associated with this particular network node
        recs = radio_project.receivers[radio_project.receivers.index == node]
        self.receivers = recs.index  # get the unique receivers associated with this node
        self.data = pd.DataFrame(columns = ['freq_code','epoch','power','rec_id'])            # set up an empty data frame

        # for every receiver
        for i in self.receivers:
            # get this receivers data from the classified key
            rec_dat = pd.read_hdf(self.db,
                                  'classified',
                                  where = 'rec_id = %s'%(i))
            rec_dat = rec_dat[rec_dat.iter == rec_dat.iter.max()]
            rec_dat = rec_dat[rec_dat.test == 1]
            rec_dat = rec_dat[['freq_code','epoch','time_stamp','power','rec_id']]
            rec_dat = rec_dat.astype({'freq_code':'object',
                            'epoch':'float32',
                            'time_stamp':'datetime64[ns]',
                            'power':'float32',
                            'rec_id':'object'})
            
            self.data = pd.concat([self.data,rec_dat])

        # Define the bin size (5 seconds)
        bin_size = 30
    
        # clean up and bin the lengths
        self.data.drop_duplicates(keep = 'first', inplace = True)
        self.data.sort_values(by = ['freq_code','time_stamp'], inplace = True)
        self.data['epoch'] = (self.data['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.data['det_lag'] = self.data.groupby('freq_code')['epoch'].diff().abs() // lag_window * lag_window
        self.data['lag_binned'] = (self.data['det_lag'] // bin_size) * bin_size
        self.data.dropna(axis = 0, inplace = True)   # drop Nan from the data
        self.node = node
        self.fishes = self.data.freq_code.unique()
        
    def prompt_for_params(self, model_type):
        if model_type == 'two_process':
            print("Please enter the values for the initial quantity (y0), the quantity at time t (yt), and the time t.")
            y0 = float(input("Enter the initial quantity (y0): "))
            yt = float(input("Enter the quantity at time t (yt): "))
            t = float(input("Enter the time at which yt is observed (t): "))
        
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
            print ("Sorry, we don't yet support that model type")
        # get lag frequencies
        lags = np.arange(0,self.time_limit,2)
        freqs, bins = np.histogram(np.sort(self.data.lag_binned),lags)
        bins = bins[np.where(freqs > 0)]
        freqs = freqs[np.where(freqs > 0)]
        log_freqs = np.log(freqs)
        
        # Plot the raw data
        plt.scatter(bins, log_freqs, label='Data')
        plt.xlabel('Lag')
        plt.ylabel('Lag Frequency')
        plt.title('Raw Data for Two-Process Model')
        plt.legend()
        plt.show()
        plt.pause(5)
        # Prompt the user for initial parameters
        initial_guess = self.prompt_for_params(model_type = 'two_process')
    
        # Perform the curve fitting
        try:
            params, params_covariance = curve_fit(self.two_process_model, 
                                                  bins, 
                                                  log_freqs, 
                                                  p0=initial_guess)
            
            # Plot the fitted function
            plt.plot(bins, self.two_process_model(bins, *params), 
                     label='Fitted function',
                     color='red')
            
            plt.scatter(bins, log_freqs, label='Data')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Fitted Two-Process Model')
            plt.show()
            plt.pause(1)
            # Return the fitted parameters
            return params
        except RuntimeError as e:
            print(f"An error occurred during fitting: {e}")
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
        num_knots = int(input("Enter the number of knots (1 for two-process, 2 for three-process): "))
        initial_knots = []
        for i in range(num_knots):
            knot = float(input(f"Enter initial guess for knot {i+1}: "))
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
        response = input ('Satisfied with the bout fitting?')
        if response in ['yes', 'y', 'T','True']:
            
            
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
                                            'time_stamp':'datetime64[ns]',
                                            'power': 'float32',
                                            'rec_id': 'object',
                                            'class': 'object',
                                            'bout_no':'int32',
                                            'det_lag':'int32'})
        
                # append to hdf5
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(key = 'presence',
                                 value = fish_dat, 
                                 format = 'table', 
                                 index = False,
                                 min_itemsize = {'freq_code':20,
                                                 'rec_id':20,
                                                 'class':20},
                                 append = True, 
                                 data_columns = True,
                                 chunksize = 1000000)
            
                print ('bouts classified for fish %s'%(fish))
        else:
            print ('Give the fitting another try')
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
        self.db = radio_project.db
        self.project = radio_project
        self.nodes = nodes
        self.edges = edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        # Initialize dictionaries for presence and recapture data
        self.node_pres_dict = {}
        self.node_recap_dict = {}
        
        # Read and preprocess data for each node
        for node in nodes:
            # Read data from the HDF5 database for the given node, applying filters using the 'where' parameter
            pres_data = pd.read_hdf(
                self.db,
                'presence',
                columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'bout_no'],
                where=f"rec_id == {node}"
            )
            recap_data = pd.read_hdf(
                self.db,
                'classified',
                columns=['freq_code', 'epoch', 'time_stamp', 'power', 'rec_id', 'iter', 'test'],
                where=f"(rec_id == {node}) & (test == 1)"
            )
        
            # Further filter recap_data for the max iteration
            recap_data = recap_data[recap_data['iter'] == recap_data['iter'].max()]

            # Group presence data by frequency code and bout, then calculate min, max, and median
            summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'rec_id']).agg(
                min_epoch=('epoch', 'min'),
                max_epoch=('epoch', 'max'),
                median_power=('power', 'median')
            ).reset_index()

            # Store the processed data in the dictionaries
            self.node_pres_dict[node] = summarized_data
            self.node_recap_dict[node] = recap_data
            print(f"Completed data management process for node {node}")

    def unsupervised_removal(self):
        """
        Iterates through each presence bout in the parent data and identifies overlaps
        in the child data for each fish (freq_code). Uses statistical tests to determine 
        which receiver has the higher power for the overlapping detections and stores the result.
        """
        for i, (parent, child) in enumerate(self.edges):
            print(f"Processing edge {i+1}/{len(self.edges)}: {parent} -> {child}")
    
            parent_bouts = self.node_pres_dict[parent]
            parent_dat = self.node_recap_dict[parent]
            child_dat = self.node_recap_dict[child]
    
            if parent_bouts.empty or parent_dat.empty or child_dat.empty:
                continue
    
            # Iterate over each unique fish (freq_code) in the parent data
            for fish_id in parent_bouts['freq_code'].unique():
                print(f"Processing fish ID: {fish_id}")
    
                # Filter the parent and child data for the current fish
                parent_fish_bouts = parent_bouts[parent_bouts['freq_code'] == fish_id]
                parent_fish_dat = parent_dat[parent_dat['freq_code'] == fish_id]
                child_fish_dat = child_dat[child_dat['freq_code'] == fish_id]
    
                max_parent_power = parent_fish_dat.power.max()
                max_child_power = child_fish_dat.power.max()
    
                # Skip if there's no data for this fish in either parent or child
                if parent_fish_bouts.empty or parent_fish_dat.empty or child_fish_dat.empty:
                    print(f"No data for fish ID {fish_id} in parent or child receiver")
                    continue
    
                # Iterate over each bout for the current fish in the parent data
                for _, parent_row in parent_fish_bouts.iterrows():
                    # Extract and normalize the power values separately for parent and child bouts
                    parent_power = parent_fish_dat[
                        (parent_fish_dat['epoch'] <= parent_row['max_epoch']) & 
                        (parent_fish_dat['epoch'] >= parent_row['min_epoch'])
                    ].power.values
                    
                    child_power = child_fish_dat[
                        (child_fish_dat['epoch'] <= parent_row['max_epoch']) & 
                        (child_fish_dat['epoch'] >= parent_row['min_epoch'])
                    ].power.values
                    
                    if len(parent_power) == 0 or len(child_power) == 0:
                        print(f'No overlapping detections found for fish ID {fish_id} between {parent_row["min_epoch"]} and {parent_row["max_epoch"]}')
                        continue
                    else:
                        print('Overlapping detections found')
    
                    # Normalize the power values separately for parent and child
                    parent_norm_power = (parent_power - np.min(parent_power)) / (max_parent_power - np.min(parent_power))
                    child_norm_power = (child_power - np.min(child_power)) / (max_child_power - np.min(child_power))
    
                    # Perform a t-test to check if the parent power is significantly higher than the child power
                    t_stat, p_value = ttest_ind(parent_norm_power, child_norm_power, equal_var=False)
    
                    # Classify based on t-test and power mean comparison
                    if np.mean(parent_norm_power) > np.mean(child_norm_power) and p_value < 0.05:
                        parent_classification = 0  # Near
                        print(f"Fish ID {fish_id}: Parent classified as NEAR with p-value: {p_value:.4f}")
                    else:
                        parent_classification = 1  # Far
                        print(f"Fish ID {fish_id}: Parent classified as FAR with p-value: {p_value:.4f}")
    
                    # Update parent data with classification result
                    parent_dat.loc[
                        (parent_dat['freq_code'] == fish_id) &
                        (parent_dat['epoch'] >= parent_row['min_epoch']) &
                        (parent_dat['epoch'] <= parent_row['max_epoch']),
                        'overlapping'
                    ] = parent_classification
    
            # Write the updated parent data to the HDF5 store incrementally
            self.write_results_to_hdf5(parent_dat)
    
            # Cleanup and memory management
            del parent_bouts, parent_dat, child_dat
            gc.collect()

    def nested_doll(self):
        """
        Identify and mark overlapping detections between parent and child nodes.
        """
        overlaps_found = False
        overlap_count = 0
        
        for i in self.node_recap_dict:
            fishes = self.node_recap_dict[i].freq_code.unique()

            for j in fishes:
                children = list(self.G.successors(i))
                fish_dat = self.node_recap_dict[i][self.node_recap_dict[i].freq_code == j]
                #fish_dat = np.repeat(i,len(fish_dat))
                fish_dat['overlapping'] = 0
                fish_dat['parent'] = ''

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
                                fish_dat.loc[overlaps, 'overlapping'] = 1
                                fish_dat.loc[overlaps, 'parent'] = i

                fish_dat = fish_dat.astype({
                    'freq_code': 'object',
                    'epoch': 'int32',
                    'rec_id': 'object',
                    'node': 'object',
                    'overlapping': 'int32',
                    'parent': 'object'
                })

                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(key='overlapping',
                                  value=fish_dat,
                                  format='table',
                                  index=False,
                                  min_itemsize={'freq_code': 20,
                                                'rec_id': 20,
                                                'parent': 20},
                                  append=True,
                                  data_columns=True,
                                  chunksize=1000000)

        if overlaps_found:
            print(f"Overlaps were found and processed. Total number of overlaps: {overlap_count}.")
        else:
            print("No overlaps were found.")

    def write_results_to_hdf5(self, df):
        """
        Writes the processed DataFrame to the HDF5 database.

        Args:
            df (DataFrame): The DataFrame containing processed detection data.
        
        The function appends data to the 'overlapping' table in the HDF5 database, ensuring 
        that each record is written incrementally to minimize memory usage.
        """
        try:
            with pd.HDFStore(self.project.db, mode='a') as store:
                store.append(
                    key='overlapping',
                    value=df[['freq_code', 'epoch', 'time_stamp', 'rec_id', 'overlapping']],
                    format='table',
                    data_columns=True,
                    min_itemsize={'freq_code': 20, 'rec_id': 20}
                )
        except Exception as e:
            print(f"Error writing to HDF5: {e}")




                
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