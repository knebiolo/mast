# -*- coding: utf-8 -*-
'''
Module contains all of the methods and classes required to identify and remove
overlapping detections from radio telemetry data.
'''

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import sqlite3
import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx
from matplotlib import rcParams
from scipy import interpolate

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

class bout():
    '''Python class object to delineate when bouts occur at receiver.'''
    def __init__ (self,node,dBase,lag_window, time_limit):
        self.lag_window = lag_window
        self.time_limit = time_limit
        conn = sqlite3.connect(dBase)
        recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(node)   # SQL code to import data from this node
        receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)        # import data
        receivers = receivers[receivers.Node == node].recID.unique()           # get the unique receivers associated with this node
        data = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])            # set up an empty data frame
        c = conn.cursor()
        for i in receivers:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            max_iter = 1
            while curr_idx <= len(tblList) - 1:
                for j in tblList:
                    if int(j[-1]) >= max_iter:
                        max_iter = int(j[-1])
                        max_iter_dict[i] = j
                    curr_idx = curr_idx + 1
            curr_idx = 0

            # once we have a hash table of receiver to max classification, extract the classification dataset
            for j in max_iter_dict:
                sql = "SELECT FreqCode, Epoch, recID, test FROM %s"%(max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn)                                 # get data for this receiver
                dat = dat[(dat.test == 1)] # query
                dat.drop(['test'],axis = 1, inplace = True)
                data = data.append(dat)

        c.close()
        data.drop_duplicates(keep = 'first', inplace = True)
        data['det_lag'] = (data.Epoch.diff()//lag_window * lag_window)
        data.dropna(axis = 0, inplace = True)                                  # drop Nan from the data
        self.bout_data = data[(data.det_lag > 0) & (data.det_lag <= time_limit)]
        self.node = node
        self.fishes = data.FreqCode.unique()

    def broken_stick_3behavior(self, plotOutputWS = None):
        '''Function to find the bout length using a broken stick/ piecewise linear
        regression technique.  We do not know the knot criteria ahead of time, therefore
        our procedure finds the knot that minimizes error.

        Method will attempt to minimize the sum of the sum of squared residuals for each
        each line.  If we assume there are two behaviors, there are two equations and one
        breakpoint.  The position of the breakpoint is where the sum of the sum
        squares of the residuals is minimized.

        Further, we are only testing breakpoints that exist in our data, therefore lags can only
        occur on 2 second intervals, not continuously'''

        # get data
        det_lag_DF = self.bout_data
        det_lag_DF = det_lag_DF.groupby('det_lag')['det_lag'].count()          # group by the detection lag and count each of the bins
        det_lag_DF = pd.Series(det_lag_DF, name = 'lag_freq')                  # rename the series field
        det_lag_DF = pd.DataFrame(det_lag_DF).reset_index()                    # convert to a dataframe
        det_lag_DF['log_lag_freq'] = np.log(det_lag_DF.lag_freq)               # log the lal frequencies - make things nice and normal for OLS
        knots = det_lag_DF.det_lag.values[2:-2]                                # extract every potential detection lag values that can be a knot - need at least 3 points for an OLS regression, therefore we extract every node except the first three and last three
        knot_ssr = dict()                                                      # create empty dictionary to hold results

        if len(knots) <= 5: # middle behavior segment must be at least 3 points long
            minKnot = 7200
            return minKnot
        else:
            # test all potential knots and write MSE to knot dictionary
            for i in np.arange(2,len(knots),1):
                k1 = i
                for j in np.arange(i+2,len(knots)-2,1):
                    k2 = j
                    first = det_lag_DF.iloc[0:k1]                              # get data
                    second = det_lag_DF[k1:k2]
                    third = det_lag_DF.iloc[k2:]
                    mod1 = smf.ols('log_lag_freq~det_lag', data = first).fit() # fit models to the data
                    mod2 = smf.ols('log_lag_freq~det_lag', data = second).fit()
                    mod3 = smf.ols('log_lag_freq~det_lag', data = third).fit()
                    mod1ssr = mod1.ssr                                         # get the sum of the squared residuals from each model
                    mod2ssr = mod2.ssr
                    mod3ssr = mod3.ssr
                    print ("With knots at %s and %s seconds, the SSR of each model was %s, %s and %s respectivley"%(det_lag_DF.det_lag.values[k1],det_lag_DF.det_lag.values[k2],round(mod1ssr,4),round(mod2ssr,4),round(mod3ssr,4)))
                    ssr_sum = np.sum([mod1ssr,mod2ssr,mod3ssr])                # simple sum is fine here
                    knot_ssr[(k1,k2)] = ssr_sum                                     # add that sum to the dictionary
            knotDF = pd.DataFrame.from_dict(knot_ssr, orient = 'index').reset_index()
            knotDF.rename(columns = {'index':'knots',0:'SSR'}, inplace = True)
            print(knotDF)
            if len(knotDF) == 0:
                minKnot = 7200
                return minKnot
            else:
                min_knot_idx = knotDF['SSR'].idxmin()
                minKnot = knotDF.iloc[min_knot_idx,0]
                # You Can Plot If You Want To
                if plotOutputWS != None:
                    k1 = minKnot[0]
                    k2 = minKnot[1]
                    dat1 = det_lag_DF.iloc[0:k1]
                    dat2 = det_lag_DF.iloc[k1:k2]
                    dat3 = det_lag_DF.iloc[k2:]
                    mod1 = smf.ols('log_lag_freq~det_lag', data = dat1).fit()     # fit models to the data
                    mod2 = smf.ols('log_lag_freq~det_lag', data = dat2).fit()
                    mod3 = smf.ols('log_lag_freq~det_lag', data = dat3).fit()
                    x1 = np.linspace(dat1.det_lag.min(), dat1.det_lag.max(), 100)
                    x2 = np.linspace(dat2.det_lag.min(), dat2.det_lag.max(), 100)
                    x3 = np.linspace(dat3.det_lag.min(), dat3.det_lag.max(), 100)
                    plt.figure(figsize = (3,3))
                    plt.plot(det_lag_DF.det_lag.values, det_lag_DF.log_lag_freq.values, "o")
                    plt.plot(x1, mod1.params[1]*x1+mod1.params[0], color = 'red')
                    plt.plot(x2, mod2.params[1]*x2+mod2.params[0], color = 'red')
                    plt.plot(x3, mod3.params[1]*x3+mod3.params[0], color = 'red')
                    plt.xlabel('Detection Lag (s)')
                    plt.ylabel('Log Lag Count')
                    plt.title('Bout Length Site %s \n %s seconds'%(self.node, det_lag_DF.det_lag.values[minKnot[1]]))
                    plt.ylim(0,max(det_lag_DF.log_lag_freq.values)+1)
                    plt.savefig(os.path.join(plotOutputWS,"BoutsAtSite%s.png"%(self.node)))

            return det_lag_DF.det_lag.values[minKnot[1]]

    def presence(self, fish, bout_length, dBase, scratch):
        '''Function takes the break point between a continuous presence and new presence,
        enumerates the presence number at a receiver and writes the data to the
        analysis database.'''
        # get data
        conn = sqlite3.connect(dBase)
        recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(self.node)            # SQL code to import data from this node
        receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)                 # import data
        receivers = receivers[receivers.Node == self.node].recID.unique()                    # get the unique receivers associated with this node
        presence = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])                             # set up an empty data frame
        c = conn.cursor()
        for i in receivers:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            # iterate over the receivers to find the final classification (aka the largest _n)
            max_iter_dict = {} # receiver:max iter
            curr_idx = 0
            max_iter = 1
            while curr_idx <= len(tblList) - 1:
                for j in tblList:
                    if int(j[-1]) >= max_iter:
                        max_iter = int(j[-1])
                        max_iter_dict[i] = j
                    curr_idx = curr_idx + 1
            curr_idx = 0
            del j

            # once we have a hash table of receiver to max classification, extract the classification dataset
            for j in max_iter_dict:
                sql = "SELECT FreqCode, Epoch, recID, test FROM %s WHERE FreqCode == '%s'"%(max_iter_dict[j],fish)
                dat = pd.read_sql(sql, con = conn)                                 # get data for this receiver
                dat = dat[(dat.test == 1)] # query
                dat.drop(['test'],axis = 1, inplace = True)
                presence = presence.append(dat)

        c.close()
        presence.drop_duplicates(keep = 'first', inplace = True)
        # get fish data
        presence['det_lag'] = (presence.Epoch.diff()//self.lag_window * self.lag_window)
        presence['det_lag'].fillna(0, inplace = True)
        presence.sort_values(by = ['Epoch'], axis = 0, inplace = True)
        presence.set_index('Epoch', drop = False, inplace = True)
        # determine new presences
        # step 1: apply new presence criteria to lag column using numpy where
        presence['new_presence'] = np.where(presence.det_lag.values > bout_length,1,0)
        # step 2: extract new presences
        new_presence = presence[presence.new_presence == 1]
        new_presence.reset_index(inplace = True, drop = True)
        new_presence.sort_values('Epoch', inplace = True)
        # step 3: rank each presence by epoch
        new_presence['presence_number'] = new_presence.Epoch.rank()
        new_presence.set_index('Epoch', drop = False, inplace = True)

        # step 4: join presence number to presence data frame on epoch
        presence = presence.join(new_presence[['presence_number']], how = 'left')
        # step 5: forward fill presence numbers until the next new presence, then fill initial presence as zero
        presence.fillna(method = 'ffill',inplace = True)
        presence.fillna(value = 0, inplace = True)
        presence.to_csv(os.path.join(scratch,'%s_at_%s_presence.csv'%(fish,self.node)), index = False)

def manage_node_presence_data(inputWS, projectDB):
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f), dtype = {'FreqCode':str,'Epoch':np.int32,'recID':str,'det_lag':np.int32,'presence_number':np.float64})
        dat.to_sql('tblPresence',con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
    c.close()

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

    def __init__(self,curr_node,nodes,edges, projectDB, outputWS,figureWS = None):
        '''The initialization module imports data and creates a networkx graph object.

        The end user supplies a list of nodes, and a list of edges with instructions
        on how to connect them and the function does the rest.  NO knowlege of networkx
        is required.

        The nodes and edge relationships should start with the outermost nodes and
        eventually end with the inner most node/receiver combinations.

        Nodes must be a list of nodes and edges must be a list of tuples.
        Edge example: [(1,2),(2,3)],
        Edges always in format of [(from,to)] or [(outer,inner)] or [(child,parent)]'''

        # Step 1, create a directed graph from list of edges
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.curr_node = curr_node
        self.outputWS = outputWS
        # Step 2, import data and create a dictionary of node dataframes
        conn = sqlite3.connect(projectDB)
        c = conn.cursor()
        self.node_pres_dict = dict()
        self.node_recap_dict = dict()
        for i in nodes:
            #import data and add to node dict
            recSQL = "SELECT * FROM tblMasterReceiver WHERE Node == '%s'"%(i)  # SQL code to import data from this node
            receivers = pd.read_sql(recSQL,con = conn, coerce_float = True)    # import data
            node_recs = receivers[receivers.Node == i].recID.unique()          # get the unique receivers associated with this node
            pres_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID','presence_number'])        # set up an empty data frame
            recap_data = pd.DataFrame(columns = ['FreqCode','Epoch','recID'])
            for j in node_recs:
                c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                tbls = c.fetchall()
                tblList = []
                for k in tbls:
                    if j in k[0]:
                        tblList.append(k[0])
                del k
                # iterate over the receivers to find the final classification (aka the largest _n)
                max_iter_dict = {} # receiver:max iter
                curr_idx = 0
                max_iter = 1
                while curr_idx <= len(tblList) - 1:
                    for k in tblList:
                        if int(k[-1]) >= max_iter:
                            max_iter = int(k[-1])
                            max_iter_dict[j] = k
                        curr_idx = curr_idx + 1
                curr_idx = 0
                #print (tblList)
                #print (max_iter_dict)
                # once we have a hash table of receiver to max classification, extract the classification dataset
                for k in max_iter_dict:

                    #print "Start selecting classified and presence data that matches the current receiver (%s)"%(j)
                    presence_sql = "SELECT * FROM tblPresence WHERE recID = '%s'"%(j)
                    presenceDat = pd.read_sql(presence_sql, con = conn)
                    recap_sql = "SELECT FreqCode, Epoch, recID, test from %s"%(max_iter_dict[j])
                    recapDat = pd.read_sql(recap_sql, con = conn)
                    recapDat = recapDat[(recapDat.test == 1)]

                    recapDat.drop(labels = ['test'], axis = 1, inplace = True)
                    # now that we have data, we need to summarize it, use group by to get min ans max epoch by freq code, recID and presence_number
                    pres_data = pres_data.append(presenceDat)
                    recap_data = recap_data.append(recapDat)
                    del presence_sql, recap_sql

            dat = pres_data.groupby(['FreqCode','presence_number'])['Epoch'].agg(['min','max'])
            dat.reset_index(inplace = True, drop = False)
            dat.rename(columns = {'min':'min_Epoch','max':'max_Epoch'},inplace = True)
            self.node_pres_dict[i] = dat
            self.node_recap_dict[i] = recap_data
            del pres_data, recap_data, dat, recSQL, receivers, node_recs
        print ("Completed data management process for node %s"%(self.curr_node))
        c.close()

        # visualize the graph
        if figureWS != None:
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
            plt.savefig(os.path.join(figureWS,'overlap_model.png'),bbox_inches = 'tight')
            plt.show()

def russian_doll(overlap):
    '''Function iterates through matching recap data from successors to see if
    current recapture row at predeccesor overlaps with successor presence.'''
    #print "Starting Russing Doll algorithm for node %s, hope you aren't busy, you will be watching this process for hours"%(overlap.curr_node)
    # create function that we can vectorize over list of epochs (i)
    def overlap_check(i, min_epoch, max_epoch):
        return np.logical_and(min_epoch >= i, max_epoch < i).any()
    nodeDat = overlap.node_recap_dict[overlap.curr_node]
    fishes = nodeDat.FreqCode.unique()
    for i in fishes:
        overlap.fish = i
        #print "Let's start sifting through fish %s at node %s"%(i, overlap.curr_node)
        children = overlap.G.succ[overlap.curr_node]
        fishDat = nodeDat[nodeDat.FreqCode == i]
        fishDat['overlapping'] = np.zeros(len(fishDat))
        fishDat['successor'] = np.repeat('',len(fishDat))
        fishDat.set_index('Epoch', inplace = True, drop = False)
        if len(children) > 0:                                            # if there is no child node, who cares? there is no overlapping detections, we are at the middle doll
            for j in children:
                child_dat = overlap.node_pres_dict[j]
                if len(child_dat) > 0:
                    child_dat = child_dat[child_dat.FreqCode == i]
                    if len(child_dat) > 0:
                        min_epochs = child_dat.min_Epoch.values
                        max_epochs = child_dat.max_Epoch.values
                        for k in fishDat.Epoch.values:                          # for every row in the fish data
                            print ("Fish %s epoch %s overlap check at child %s"%(i,k,j))
                            if np.logical_and(min_epochs <= k, max_epochs >k).any(): # if the current epoch is within a presence at a child receiver
                                print ("Overlap Found, at %s fish %s was recaptured at both %s and %s"%(k,i,overlap.curr_node,j))
                                fishDat.at[k,'overlapping'] = 1
                                fishDat.at[k,'successor'] = j
        fishDat.reset_index(inplace = True, drop = True)
        fishDat.to_csv(os.path.join(overlap.outputWS,'%s_at_%s_soverlap.csv'%(i,overlap.curr_node)), index = False)

def manage_node_overlap_data(inputWS, projectDB):
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f), dtype = {'FreqCode':str,'Epoch':np.int32,'recID':str,'overlapping':np.int32})
        dat.to_sql('tblOverlap',con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
    c.close()
