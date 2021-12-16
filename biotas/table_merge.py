# -*- coding: utf-8 -*-
"""
Modules contains all of the functions to merge individual radio telemetry receiver
tables into a tblRecaptures.
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

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

def the_big_merge(outputWS,projectDB, hitRatio_Filter = False, pre_release_Filter = False, rec_list = None, con_rec_filter = None):
    '''function takes classified data, merges across sites and then joins presence
    and overlapping data into one big file for model building.'''
    conn = sqlite3.connect(projectDB)                                              # connect to the database
    if rec_list != None:
        recSQL = "SELECT * FROM tblMasterReceiver WHERE recID = '%s'"%(rec_list[0])
        for i in rec_list[1:]:
            recSQL = recSQL + " OR recID = '%s'"%(i)
    else:
        recSQL = "SELECT * FROM tblMasterReceiver"                                 # SQL code to import data from this node
    receivers = pd.read_sql(recSQL,con = conn)                                 # import data
    receivers = receivers.recID.unique()                                       # get the unique receivers associated with this node
    recapdata = pd.DataFrame(columns = ['FreqCode','Epoch','recID','timeStamp','fileName'])                # set up an empty data frame
    c = conn.cursor()

    bouts = False
    for i in receivers:                                                            # for every receiver

        print ("Start selecting and merging data for receiver %s"%(i))
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tbls = c.fetchall()
        tblList = []
        for j in tbls:
            if i in j[0]:
                tblList.append(j[0])
            if j[0] == 'tblPresence' or j[0]== 'tblOverlap':
                bouts = True
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

            cursor = conn.execute('select * from %s'%(max_iter_dict[j]))
            names = [description[0] for description in cursor.description]

            if bouts == True:
                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,presence_number, overlapping, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode
                    LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID
                    LEFT JOIN tblPresence ON %s.FreqCode = tblPresence.FreqCode AND %s.Epoch = tblPresence.Epoch AND %s.recID = tblPresence.recID'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp,presence_number, overlapping,test, RelDate
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode
                    LEFT JOIN tblOverlap ON %s.FreqCode = tblOverlap.FreqCode AND %s.Epoch = tblOverlap.Epoch AND %s.recID = tblOverlap.recID
                    LEFT JOIN tblPresence ON %s.FreqCode = tblPresence.FreqCode AND %s.Epoch = tblPresence.Epoch AND %s.recID = tblPresence.recID'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver
                dat['overlapping'].fillna(0,inplace = True)
                #dat = dat[dat.overlapping == 0]


            else:

                if 'hitRatio_A' in names:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp, hitRatio_A, hitRatio_M, detHist_A, detHist_M, conRecLength_A, conRecLength_M, lag, lagDiff, test, RelDate
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])

                else:
                    sql = '''SELECT %s.FreqCode, %s.Epoch, %s.recID, timeStamp, test, RelDate
                    FROM %s
                    LEFT JOIN tblMasterTag ON %s.FreqCode = tblMasterTag.FreqCode'''%(max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j],max_iter_dict[j])
                dat = pd.read_sql(sql, con = conn, coerce_float = True)                     # get data for this receiver


            dat = dat[dat.test == 1]
            dat['RelDate'] = pd.to_datetime(dat.RelDate)
            dat['timeStamp'] = pd.to_datetime(dat.timeStamp)
            if hitRatio_Filter == True:
                dat = dat[(dat.hitRatio_A > 0.10)]# | (dat.hitRatio_M > 0.10)]
            if con_rec_filter == True:
                dat = dat[dat.conRecLength_A >= 2]
            if pre_release_Filter == True:
                dat = dat[(dat.timeStamp >= dat.RelDate)]
            recapdata = recapdata.append(dat)
            del dat
    c.close()

    recapdata.drop_duplicates(keep = 'first', inplace = True)
    return recapdata
