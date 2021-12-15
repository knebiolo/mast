# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:41:15 2021

@author: KNebiolo

Script Intent: play around with an adjacency matrix filter on tblRecaptures - can 
I even load it into memory?
"""

# import modules
import networkx as nx
import pandas as pd
import sqlite3 
import os 
import numpy as np

# identify workspaces
inputWS = r"D:\Manuscript\CT_River_2015"

dbDir = os.path.join(inputWS,'Data', 'ctr_2015_v2.db')


# connect to the project database
conn = sqlite3.connect(dbDir)
c = conn.cursor()

# first step identify fish and the unique movements they make
dat = pd.read_sql('select FreqCode,Epoch,recID from tblRecaptures', con = conn)
dat = dat.groupby(['FreqCode','Epoch'])['recID','Epoch'].toframe()

dat['prev_recID'] = dat['recID'].shift(1)
dat['prev_Epoch'] = dat['Epoch'].shift(1)
dat['moves'] = tuple(zip(dat.prev_recID.values.tolist(),dat.recID.values.tolist()))
dat['dt'] = dat.Epoch - dat.prev_Epoch
fish = dat.FreqCode.unique()
moves = dat.moves.unique()

print (dat.head())

#del dat

# then, create a graph
G = nx.Graph()

# first get nodes
nodes = set(pd.read_sql('select Node from tblNodes', con = conn)['Node'].values.tolist())

# add nodes to graph
G.add_nodes_from(nodes)

# build edges 
G.add_edges_from([('T01','T02'),
                   ('T02','T01'),('T02','T03'),
                   ('T03','T02'),('T03','T05')])




