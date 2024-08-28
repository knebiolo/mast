# -*- coding: utf-8 -*-
'''
Module contains all of the methods and classes required to identify and remove
overlapping detections from radio telemetry data.
'''

# import modules required for function dependencies
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

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
        recs = radio_project.receivers[radio_project.receivers.node == node]
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
            rec_dat = rec_dat[['freq_code','epoch','power','rec_id']]
            rec_dat = rec_dat.astype({'freq_code':'object',
                            'epoch':'float32',
                            'power':'float32',
                            'rec_id':'object'})
            
            self.data = pd.concat([self.data,rec_dat])

        # clean up and bin the lengths
        self.data.drop_duplicates(keep = 'first', inplace = True)
        self.data.sort_values(by = ['freq_code','epoch'], inplace = True)
        self.data['det_lag'] = self.data.groupby('freq_code')['epoch'].diff().abs() // lag_window * lag_window
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
        freqs, bins = np.histogram(np.sort(self.data.det_lag),lags)
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
            
            # Return the fitted parameters
            return params
        except RuntimeError as e:
            print(f"An error occurred during fitting: {e}")
            return None
        
    def find_knot(self, initial_knot_guess):
        # get lag frequencies
        lags = np.arange(0, self.time_limit, 2)
        freqs, bins = np.histogram(np.sort(self.data.det_lag), lags)
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
        freqs, bins = np.histogram(np.sort(self.data.det_lag), lags)
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
        else:
            print("Optimization failed:", result.message)
            return None

        return optimized_params[-1]
    
    def fit_processes(self):
        # Step 1: Plot bins vs log frequencies
        lags = np.arange(0, self.time_limit, 2)
        freqs, bins = np.histogram(np.sort(self.data.det_lag), lags)
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
                                        'epoch': 'float32',
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
            
class overlap_reduction():
    """
    Class to reduce redundant detections at overlapping receivers.

    This class identifies and removes overlapping detections, which occur when a
    fish is detected by multiple receivers with overlapping detection zones. By
    identifying the true receiver based on signal power, this class helps to remove
    biases in time-to-event modeling when studying fish movement.

    Attributes:
        db (str): Path to the project database.
        G (networkx.DiGraph): Directed graph representing the relationships between receivers.
        node_pres_dict (dict): Dictionary storing presence data for each node (receiver).
        node_recap_dict (dict): Dictionary storing recapture data for each node (receiver).
    """

    def __init__(self, nodes, edges, radio_project):
        """
        Initializes the OverlapReduction class by creating a directed graph and
        importing the necessary data for each node (receiver).

        Args:
            nodes (list): List of nodes (receivers) to be analyzed.
            edges (list of tuples): List of edges defining the relationship between nodes.
                Format should be [(from_node, to_node)] representing (outer, inner) nodes.
            radio_project (object): Object containing the radio telemetry project data.
        """
        self.db = radio_project.db

        # Step 1: Create a directed graph from the list of edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        
        # Step 2: Import data and create dictionaries for node dataframes
        self.node_pres_dict = {}
        self.node_recap_dict = {}
        
        for node in nodes:
            # Import data for each node and add to dictionaries
            node_recs = radio_project.receivers[radio_project.receivers.node == node].index
            pres_data = pd.DataFrame(columns=['freq_code', 'epoch', 'power', 'node', 'rec_id', 'presence'])
            recap_data = pd.DataFrame(columns=['freq_code', 'epoch', 'power', 'node', 'rec_id'])
            
            for rec_id in node_recs:
                # Load presence data for the receiver
                presence_dat = pd.read_hdf(radio_project.db, 'presence', where=f'rec_id = {rec_id}')
                presence_dat['node'] = np.repeat(node, len(presence_dat))
                
                # Load recapture data for the receiver
                class_dat = pd.read_hdf(radio_project.db, 'classified', where=f'rec_id = {rec_id}')
                class_dat = class_dat[class_dat.iter == class_dat.iter.max()]
                class_dat = class_dat[class_dat.test == 1][['freq_code', 'epoch', 'power', 'rec_id']]
                class_dat['node'] = np.repeat(node, len(class_dat))
                
                # Append to node-specific dataframes
                pres_data = pd.concat([pres_data, presence_dat])
                recap_data = pd.concat([recap_data, class_dat])

            # Summarize presence data
            summarized_data = pres_data.groupby(['freq_code', 'bout_no', 'node', 'rec_id']).agg(
                min_epoch=('epoch', 'min'),
                max_epoch=('epoch', 'max'),
                median_power=('power', 'median'),
            ).reset_index()

            self.node_pres_dict[node] = summarized_data
            self.node_recap_dict[node] = recap_data
            
            # Clean up intermediate dataframes
            del pres_data, recap_data, summarized_data, presence_dat, class_dat
            print(f"Completed data management process for node {node}")

        # Visualize the graph
        self.visualize_graph()

    def visualize_graph(self):
        """
        Visualizes the directed graph representing the relationships between nodes.
        """
        pos = nx.circular_layout(self.G)
        nx.draw(self.G, pos, node_color='r', node_size=400, with_labels=True)
        plt.axis('off')
        plt.show()

    def unsupervised_removal(self, project):
        """
        Identifies and removes overlapping detections across receivers using signal power.
    
        This method performs pairwise comparisons between nodes (receivers) to determine the true
        receiver where the fish is located based on median signal power. Overlapping detections with
        lower signal power are marked and can be removed from further analysis. A visualization of 
        the K-means results is saved to the project directory.
    
        Args:
            project (object): The project object containing the project directory path.
        """
    
        # Combine bouts from all receivers
        bout_summaries = []
        for node, df in self.node_pres_dict.items():
            fish_bouts = df.copy()  # No need to filter by fish_id now
            fish_bouts['node'] = node
            bout_summaries.append(fish_bouts)
    
        if len(bout_summaries) == 0:
            return
    
        bout_summaries = pd.concat(bout_summaries, ignore_index=True)
    
        # Normalize the power values within each receiver's bouts
        bout_summaries['norm_power'] = bout_summaries.groupby('node')['median_power'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
    
        # Perform pairwise comparison of nodes for overlapping bouts
        nodes = bout_summaries['node'].unique()
        classified_detections = []
    
        # Track which detections have already been classified
        already_classified = set()
    
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                bouts_a = bout_summaries[bout_summaries['node'] == node_a]
                bouts_b = bout_summaries[bout_summaries['node'] == node_b]
    
                # Check if there are bouts to compare
                if len(bouts_a) == 0 or len(bouts_b) == 0:
                    print(f"Skipping comparison between node {node_a} and {node_b} because one has no bouts.")
                    continue
    
                # Identify overlapping bouts between nodes
                overlapping_bouts_a = bouts_a[bouts_a.apply(
                    lambda row: ((row['min_epoch'] <= bouts_b['max_epoch']) &
                                 (row['max_epoch'] >= bouts_b['min_epoch'])).any(),
                    axis=1)]
    
                overlapping_bouts_b = bouts_b[bouts_b.apply(
                    lambda row: ((row['min_epoch'] <= bouts_a['max_epoch']) &
                                 (row['max_epoch'] >= bouts_a['min_epoch'])).any(),
                    axis=1)]
    
                if len(overlapping_bouts_a) == 0 or len(overlapping_bouts_b) == 0:
                    print(f"No overlapping bouts found between node {node_a} and {node_b}.")
                    continue
    
                print(f"Processing overlapping bouts between node {node_a} and {node_b}.")
    
                # Combine overlapping bouts for K-means clustering
                combined_bouts = pd.concat([overlapping_bouts_a, overlapping_bouts_b])
    
                combined_power = combined_bouts['norm_power'].values.reshape(-1, 1)
    
                kmeans = KMeans(n_clusters=2, random_state=42).fit(combined_power)
                centers = kmeans.cluster_centers_.flatten()
                combined_bouts['cluster'] = kmeans.labels_
    
                # Determine which cluster is 'near' (higher power)
                near_cluster = np.argmax(centers)
                combined_bouts['assigned_label'] = combined_bouts['cluster'].apply(
                    lambda x: 'near' if x == near_cluster else 'far'
                )
    
                # Now, map the clustering back to individual detections in node_recap_dict
                for node in [node_a, node_b]:
                    recaps = self.node_recap_dict[node].copy()
    
                    for _, bout in combined_bouts[combined_bouts['node'] == node].iterrows():
                        in_bout = (recaps['epoch'] >= bout['min_epoch']) & (recaps['epoch'] <= bout['max_epoch'])
                        bout_ids = set(recaps.loc[in_bout].index)
    
                        # Only classify detections that haven't been classified yet
                        if not bout_ids.intersection(already_classified):
                            recaps.loc[in_bout, 'overlapping'] = 1 if bout['assigned_label'] == 'far' else 0
                            recaps.loc[in_bout, 'parent'] = node_a if bout['assigned_label'] == 'far' else node_b
    
                            # Mark these detections as classified
                            already_classified.update(bout_ids)
    
                    classified_detections.append(recaps)
    
                # Plot and save the K-means results for visualization
                self._plot_kmeans_results(combined_bouts, centers, None, node_a, node_b, project.project_dir)
    
        if classified_detections:
            final_result = pd.concat(classified_detections, ignore_index=True)
            final_result.fillna(0, inplace=True)
            final_result = final_result.astype({
                'freq_code': 'object',
                'epoch': 'int32',
                'rec_id': 'object',
                'node': 'object',
                'overlapping': 'int32',
                'parent': 'object'
            })
    
            # Save the final results to the HDF5 store
            with pd.HDFStore(self.db, mode='a') as store:
                store.append(key='overlapping',
                             value=final_result,
                             format='table',
                             index=False,
                             min_itemsize={'freq_code': 20, 'rec_id': 20, 'parent': 20},
                             append=True,
                             data_columns=True,
                             chunksize=1000000)
            print(f'Processed overlap for all receiver pairs.')
            
    def _plot_kmeans_results(self, combined, centers, fish_id, node_a, node_b, project_dir):
        """
        Plots and saves the K-means clustering results to the project directory.
    
        Args:
            combined (pd.DataFrame): The combined DataFrame of detections from two nodes.
            centers (np.array): The cluster centers from K-means.
            fish_id (str): The fish identifier.
            node_a (str): The first node involved in the comparison.
            node_b (str): The second node involved in the comparison.
            project_dir (str): The directory where the plot will be saved.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(combined['norm_power'], bins=30, alpha=0.5, label='Normalized Power')
        plt.axvline(centers[0], color='r', linestyle='dashed', linewidth=2, label='Cluster Center 1')
        plt.axvline(centers[1], color='b', linestyle='dashed', linewidth=2, label='Cluster Center 2')
        plt.title(f"K-means Clustering for Fish {fish_id} between Nodes {node_a} and {node_b}")
        plt.xlabel("Normalized Power")
        plt.ylabel("Frequency")
        plt.legend()
    
        # Create the output path
        output_path = os.path.join(project_dir,'Output','Figures', f'kmeans_fish_{fish_id}_nodes_{node_a}_{node_b}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"K-means plot saved")
        
    def nested_doll(self):
        """
        Identify and mark overlapping detections between parent and child nodes.
    
        This method checks for overlaps between detections at a parent node and its 
        child nodes based on the presence data. If an overlap is found, the method 
        marks these detections and stores the updated information in the project database.
        """
        overlaps_found = False
        overlap_count = 0
        
        for i in self.node_recap_dict:
            fishes = self.node_recap_dict[i].freq_code.unique()

            for j in fishes:
                children = list(self.G.successors(i))
                fish_dat = self.node_recap_dict[i][self.node_recap_dict[i].freq_code == j].copy()
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
            
