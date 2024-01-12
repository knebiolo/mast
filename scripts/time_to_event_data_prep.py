# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:30:52 2019

@author: Alex Malvezzi
"""

# import modules
import sys
sys.path.append(r'C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas')
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
import pickle

# set up script parameters
proj_dir = r'J:\2819\005\Calcs\Working_Nuyakuk2023Telem'                                      # what is the project directory?
dbName = '2023Nuyakuk.db'                                                         # whad did you call the database?

projectDB = os.path.join(proj_dir,'Data',dbName)

# what is the output directory?
outputWS = os.path.join(proj_dir,'Output')


# what is the Node to State relationship - use Python dictionary
node_to_state = {'R01':2,'R02':3,'R03':30,'R04':40,
                 'R10':10,'R11':11,'R12':12,'R13':13,
                 'R14':14,'R15':15,'R16':16}
recList = ['R01','R02','R03','R04',
           'R10','R11','R12','R13',
           'R14','R15','R16']
o_fileName = "vr2test.csv"


# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
tte = biotas.time_to_event(recList,
                            node_to_state,
                            projectDB,
                            initial_state_release = True)

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