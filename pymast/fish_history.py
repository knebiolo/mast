# -*- coding: utf-8 -*-
"""
Modules contains all of the functions to plot fish location in time using
3d Matplotlib plots.  Useful for identifying remaining false positive and overlap
detections prior to running the big merge.
"""

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
from sklearn import metrics

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

class fish_history():
    """
    Interactive visualization of fish movement histories through space and time.
    
    Provides 3D matplotlib plots showing fish tracks through receiver network,
    helping identify remaining false positives, overlapping detections, and
    movement anomalies before statistical analysis.
    
    Attributes
    ----------
    filtered : bool
        If True, shows only filtered detections (test==0)
    nodes : pandas.DataFrame
        Node locations (X, Y, Node, Seconds)
    receivers : pandas.DataFrame
        Receiver metadata (rec_id, node, coordinates)
    detections : pandas.DataFrame
        Fish detections (time_stamp, rec_id, freq_code, power, etc.)
    current_fish : str
        Currently displayed freq_code
    
    Methods
    -------
    __init__(projectDB, filtered=True, overlapping=False, rec_list=None, filter_date=None)
        Initialize connection to project database and load detections
    
    change_fish(freq_code)
        Switch to different fish and update plots
    
    plot_3d_trajectory()
        Create 3D visualization of fish movement through network
    
    Notes
    -----
    - Uses matplotlib 3D plotting (Axes3D)
    - X/Y coordinates from node locations
    - Time on Z-axis for temporal progression
    - Color-coded by receiver or detection quality
    - Useful for quality control before final analysis
    
    Examples
    --------
    >>> from pymast.fish_history import fish_history
    >>> 
    >>> # Load fish tracks (filtered only)
    >>> fh = fish_history(
    ...     projectDB='C:/project/study.h5',
    ...     filtered=True,
    ...     overlapping=False
    ... )
    >>> 
    >>> # View specific fish
    >>> fh.change_fish('166.380 7')
    >>> fh.plot_3d_trajectory()
    
    See Also
    --------
    overlap_removal.visualize_overlaps : Overlap analysis plots
    formatter.time_to_event : Statistical model output
    """

    def __init__(self,projectDB,filtered = True, overlapping = False, rec_list = None,filter_date = None):
        ''' when this class is initialized, we connect to the project databae and
        default filter the detections (test == 0) and overlapping = 0'''
        self.filtered = filtered
        conn = sqlite3.connect(projectDB,timeout = 30.0)
        c = conn.cursor()
        # read and import nodes
        sql = "SELECT X, Y, Node FROM tblNodes"
        self.nodes = pd.read_sql(sql,con = conn, coerce_float = True)
        self.nodes['Seconds'] = np.zeros(len(self.nodes))
        del sql
        if rec_list == None:
            # read and import receivers
            sql = "SELECT * FROM tblMasterReceiver"
            self.receivers = pd.read_sql(sql,con = conn, coerce_float = True)
            del sql
            receivers = self.receivers.recID.unique()
        else:
            sql = "SELECT * FROM tblMasterReceiver WHERE recID = '%s' "%(rec_list[0])
            for i in rec_list[1:]:
                sql = sql + "OR recID = '%s' "%(i)
            self.receivers = pd.read_sql(sql,con = conn, coerce_float = True)
            receivers = rec_list
        # read and import recaptures
        data = pd.DataFrame(columns = ['FreqCode','Epoch','timeStamp','recID','test'])
        for i in receivers:
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tbls = c.fetchall()
            tblList = []
            for j in tbls:
                if i in j[0]:
                    tblList.append(j[0])
            del j
            #print (tblList)
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
           # print (max_iter_dict)
            if filtered == False and overlapping == False:
                sql = "SELECT FreqCode, Epoch, timeStamp, recID, test FROM tblClassify_%s_1"%(i)
                dat = pd.read_sql(sql,con = conn)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            elif filtered == True and overlapping == False:
                sql = "SELECT FreqCode, Epoch,  timeStamp, recID, test FROM %s WHERE test = 1"%(max_iter_dict[i])
                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                data = data.append(dat)
                print ("Data from reciever %s imported"%(i))
                del sql
            else:
                sql = "SELECT %s.FreqCode, %s.Epoch,  %s.timeStamp, %s.recID, overlapping, test FROM %s LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID WHERE test = 1"%(max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i],max_iter_dict[i])
                dat = pd.read_sql(sql,con = conn, coerce_float = True)
                dat['overlapping'].fillna(0,inplace = True)
                dat = dat[dat.overlapping == 0]
                data = data.append(dat)
                #print data.head()
                #fuck
                print ("Data from reciever %s imported"%(i))
                del sql
        data.drop_duplicates(keep = 'first', inplace = True)
        self.recaptures = data

        del data
        # read and import tags
        sql = "SELECT * FROM tblMasterTag"
        self.tags = pd.read_sql(sql,con = conn, coerce_float = True)
        del sql

        # calculate first recaptures
        firstRecap = pd.DataFrame(self.recaptures.groupby(['FreqCode'])['Epoch'].min().reset_index())
        firstRecap = firstRecap.rename(columns = {'Epoch':'firstRecap'})

        self.recaptures = pd.merge(self.recaptures,firstRecap,how = u'left',left_on =['FreqCode'], right_on = ['FreqCode'])
        self.recaptures = pd.merge(self.recaptures,self.receivers[['recID','Node']], how = u'left', left_on = ['recID'], right_on = ['recID'])
        self.recaptures = pd.merge(self.recaptures,self.nodes, how = u'left', left_on = ['Node'], right_on = ['Node'])
        self.recaptures['TimeSinceRecap'] = (self.recaptures.Epoch - self.recaptures.firstRecap)
        self.recaptures['HoursSinceRecap'] = self.recaptures.TimeSinceRecap/3600.0
        c.close()

    def fish_plot(self, fish):
        '''Pick a tag and plot'''
        tagDat = self.recaptures[self.recaptures.FreqCode == fish]
        tagDat.sort_values(['Epoch'], inplace = True)
        tagFirstRecap = tagDat.firstRecap.values[0]
        print ("Fish %s had No. of Recaptures = %s with first recapture at %s" %(fish,len(tagDat),tagFirstRecap))
        # plot data
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111,projection = '3d')
        ax.plot(tagDat.X.values, tagDat.Y.values, tagDat.HoursSinceRecap.values, c = 'k')
        ax.scatter(self.nodes.X.values,self.nodes.Y.values,self.nodes.Seconds.values)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Hours')
        for row in self.nodes.iterrows():
            label = row[1][2]
            x = row[1][0]
            y = row[1][1]
            ax.text(x,y,0,label,fontsize = 8)
        plt.show()
