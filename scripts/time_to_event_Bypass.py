# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:30:52 2019

@author: Alex Malvezzi
"""

# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'\\EGRET\Condor\Jobs\1210\005\Calcs\Studies\3_3_19\2019'                            
dbName = 'ultrasound_2019_ajm.db'                                                   
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output','TTE_Prep')
o_fileName_cov = "Bypass_cov.csv"
o_fileName = "Bypass.csv"
# what is the Node to State relationship - use Python dictionary 
node_to_state = {'S13':3,'S14':3,'S19':3,'S18':3, 'S20':3, 'S21':3, 'S22':3, 'S23':3, 'S12':1,'S01':2,'S02':2}
recList = ['T07','T05','T04','T03W','T03E','T02','T01','T21','T22','T08','T16','T17']
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
tte = abtas.time_to_event(recList,(node_to_state),projectDB, initial_state_release = False, cap_loc = 'Holyoke', rel_loc = 'Holyoke')
print ("Step 1 Complete, Data Class Finished")
# Step 2, format data - with covariates
tte.data_prep(os.path.join(outputWS,o_fileName_cov), time_dependent_covariates = True)
print ("Time to Event data formatted for time dependent covariates")
# Step 3, format data - without covariates
tte.data_prep(os.path.join(outputWS,o_fileName))
print ("Time to Event data formated without time dependent covariates")
# Step 4, generate a summary
tte.summary()
print ("Data formatting complete, proceed to R for Time to Event Modeling")