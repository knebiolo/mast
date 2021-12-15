# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:30:52 2019

@author: Alex Malvezzi
"""

# import modules
import biotas.biotas as biotas
import os
import warnings
warnings.filterwarnings('ignore')
import pickle

# set up script parameters
proj_dir = r'E:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                         # whad did you call the database?
                                            
projectDB = os.path.join(proj_dir,'Data',dbName)

# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output')


# what is the Node to State relationship - use Python dictionary 
# node_to_state = {'T23':1,'T24':1,'T25':2,'S4':3}
# recList = ['T23','T24','T25','T26','T27']
# o_fileName = "nfm_intake.csv"

node_to_state = {'T22':1,'T20':1,'T05':2,'T06':2,'T07':2}
recList = ['T22','T20','T05','T06','T07',]
o_fileName = "ds_canal_tte.csv"

# node_to_state = {'T03':1,'T19':2,'T20':2}
# recList = ['T03','T19','T20']
# o_fileName = "montague_spillway_2015.csv"

# node_to_state = {'T05':1,'T06':1,'T19':2,'T20':2}
# recList = ['T05','T06','T19','T20']
# o_fileName = "cabot_spillway_2015.csv"

# node_to_state = {'T15':1,'T19':2,'T20':2}
# recList = ['T15','T19','T20']
# o_fileName = "conte_spillway_2015.csv"

# node_to_state = {'T16':1,'T19':2,'T20':2}
# recList = ['T16','T19','T20']
# o_fileName = "sta1_spillway_2015.csv"

# node_to_state = {'T12E':1,'T12W':1,'T19':2,'T20':2}
# recList = ['T12E','T12W','T19','T20']
# o_fileName = "rawson_spillway_2015.csv"

# node_to_state = {'T18':1,'T21':2}
# recList = ['T18','T21']

# node_to_state = {'T23':1,'T24':1,'T21':2,'T18':2,'T13':2,'T14':2,'T08':2,'T09':2,'T19':3,'T20':3,'T05':4,'T06':4,'T03':4}
# recList = ['T23','T24','T21','T18','T13','T14','T08','T09','T19','T20','T05','T06','T03']
# o_fileName = "ds_canal_2015.csv"

# node_to_state = {'T08':1,'T05':2,'T06':2,'T03':2}
# recList = ['T08','T05','T06','T03']
# o_fileName = "canal_2015.csv"



# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
# tte = biotas.time_to_event(recList,
#                             node_to_state,
#                             projectDB,
#                             last_presence_time0 = True,
#                             rel_loc = 'Holyoke',
#                             cap_loc = 'Holyoke')


tte = biotas.time_to_event(recList,
                            node_to_state,
                            projectDB,
                            initial_state_release = True,
                            species = 'Shad')#,
                            # rel_loc = 'Holyoke', 
                            # cap_loc = 'Holyoke')

print ("Step 1 Complete, Data Class Finished")
# Step 2, format data - with covariates
# tte.data_prep(os.path.join(outputWS,o_fileName_cov), 
#               time_dependent_covariates = True,
#               adjacency_filter = [(3, 1),(3, 2),(2, 3),(4, 1),(4, 2)])
print ("Time to Event data formatted for time dependent covariates")

# Step 3, format data - without covariates
tte.data_prep(os.path.join(outputWS,o_fileName))#,adjacency_filter = [(2, 1),(3, 1),(3, 2),(2, 3),(4, 1),(4, 2)])


print ("Time to Event data formated without time dependent covariates")
# Step 4, generate a summary
tte.summary()
print ("Data formatting complete, proceed to R for Time to Event Modeling")