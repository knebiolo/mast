# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:30:52 2019

@author: Alex Malvezzi
"""

# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
import pickle

# set up script parameters
proj_dir = r'C:\Users\knebiolo\Desktop\vr2_test'                                      # what is the project directory?
dbName = 'vr2_test.db'                                                         # whad did you call the database?

projectDB = os.path.join(proj_dir,'Data',dbName)

# what is the output directory?
outputWS = os.path.join(proj_dir,'Output')


# what is the Node to State relationship - use Python dictionary
node_to_state = {'USE':1,'DS':2}
recList = ['VR2Tx-487374','VR2Tx-488390']
o_fileName = "vr2test.csv"


# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
tte = biotas.time_to_event(recList,
                            node_to_state,
                            projectDB)

print ("Step 1 Complete, Data Class Finished")
# Step 2, format data - with covariates
# tte.data_prep(os.path.join(outputWS,o_fileName_cov),
#               time_dependent_covariates = True,
#               adjacency_filter = [(3, 1),(3, 2),(2, 3),(4, 1),(4, 2)])
print ("Time to Event data formatted for time dependent covariates")

# Step 3, format data - without covariates
tte.data_prep(os.path.join(outputWS,o_fileName))#,adjacency_filter = [(3, 1),(3, 2),(3, 4),(5, 1),(5, 2),(5, 4)])


print ("Time to Event data formated without time dependent covariates")
# Step 4, generate a summary
tte.summary()
print ("Data formatting complete, proceed to R for Time to Event Modeling")