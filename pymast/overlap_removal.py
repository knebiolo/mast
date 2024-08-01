# -*- coding: utf-8 -*-
'''
Module contains all of the methods and classes required to identify and remove
overlapping detections from radio telemetry data.
'''

# import modules required for function dependencies
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams

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
        self.data = pd.DataFrame(columns = ['freq_code','epoch','rec_id'])            # set up an empty data frame

        # for every receiver
        for i in self.receivers:
            # get this receivers data from the classified key
            rec_dat = pd.read_hdf(self.db,
                                  'classified',
                                  where = 'rec_id = %s'%(i))
            rec_dat = rec_dat[rec_dat.iter == rec_dat.iter.max()]
            rec_dat = rec_dat[rec_dat.test == 1]
            rec_dat = rec_dat[['freq_code','epoch','rec_id']]
            rec_dat = rec_dat.astype({'freq_code':'object',
                            'epoch':'float32',
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
    '''Python class  to reduce redundant dections at overlappin receivers.
    More often than not, a large aerial yagi will be placed adjacent to a dipole.
    The yagi has a much larger detection area and will pick up fish in either detection
    zone.  The dipole is limited in its coverage area, therefore if a fish is
    currently present at a dipole and is also detected at the yagi, we can safely
    assume that the detection at the Yagi are overlapping and we can place the fish
    at the dipole antenna.  By identifying and removing these overlapping detections
    we remove bias in time-to-event modeling when we want to understand movement
    from detection areas with limited coverage to those areas with larger aerial
    coverages.

    This class object contains a series of methods to identify overlapping detections
    and import a table for joining into the project database.'''

    def __init__(self, nodes, edges, radio_project):
        '''The initialization module imports data and creates a networkx graph object.

        The end user supplies a list of nodes, and a list of edges with instructions
        on how to connect them and the function does the rest.  NO knowlege of networkx
        is required.

        The nodes and edge relationships should start with the outermost nodes and
        eventually end with the inner most node/receiver combinations.

        Nodes must be a list of nodes and edges must be a list of tuples.
        Edge example: [(1,2),(2,3)],
        Edges always in format of [(from,to)] or [(outer,inner)] or [(parent,child)]'''
        self.db = radio_project.db

        # Step 1, create a directed graph from list of edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        
        # Step 2, import data and create a dictionary of node dataframes
        self.node_pres_dict = dict()
        self.node_recap_dict = dict()
        
        for i in nodes:
            #import data and add to node dict
            node_recs = radio_project.receivers[radio_project.receivers.node == i]
            node_recs = node_recs.index         # get the unique receivers associated with this node
            pres_data = pd.DataFrame(columns = ['freq_code','epoch','node','rec_id','presence'])        # set up an empty data frame
            recap_data = pd.DataFrame(columns = ['freq_code','epoch','node','rec_id'])
            
            for j in node_recs:
                # get presence data and final classifications for this receiver
                presence_dat = pd.read_hdf(radio_project.db,'presence', where = 'rec_id = %s'%(j))
                presence_dat['node'] = np.repeat(i,len(presence_dat))
                class_dat = pd.read_hdf(radio_project.db,'classified', where = 'rec_id = %s'%(j))
                class_dat = class_dat[class_dat.iter == class_dat.iter.max()]
                class_dat = class_dat[class_dat.test == 1]
                class_dat = class_dat[['freq_code', 'epoch', 'rec_id']]
                class_dat['node'] = np.repeat(i,len(class_dat))
                # append to node specific dataframe
                pres_data = pd.concat([pres_data,presence_dat])
                recap_data = pd.concat([recap_data,class_dat])

            # now that we have data, we need to summarize it, use group by to get min ans max epoch by freq code, recID and presence_number
            dat = pres_data.groupby(['freq_code','bout_no','node','rec_id'])['epoch'].agg(['min','max'])
            dat.reset_index(inplace = True, drop = False)
            dat.rename(columns = {'min':'min_epoch','max':'max_epoch'},inplace = True)
            self.node_pres_dict[i] = dat
            self.node_recap_dict[i] = recap_data
            
            # clean up
            del pres_data, recap_data, dat, presence_dat, class_dat
            print ("Completed data management process for node %s"%(i))

        # visualize the graph
        shells = []
        for n in list(self.G.nodes()):
            successors = list(self.G.succ[n].keys())
            shells.append(successors)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4));
        pos= nx.circular_layout(self.G)
        nx.draw_networkx_nodes(self.G,pos,list(self.G.nodes()),node_color = 'r',node_size = 400)
        nx.draw_networkx_edges(self.G,pos,list(self.G.edges()),edge_color = 'k')
        nx.draw_networkx_labels(self.G,pos,font_size=8)
        plt.axis('off')
        plt.show()

    def nested_doll(self):
        '''Identify and mark overlapping detections.'''
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