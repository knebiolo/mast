# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:21:08 2020

@author: tcastrosantos

The intent of this script is to produce a new database to be used for classifying new data using the classify_2.py routine.  
It uses the biotas.create_training_data() function to produce a new table and writes this to a training database where this is the only table.

"""
import sys
sys.path.append(r"C:\a\Projects\Winooski\2020\Data\BIOTAS\biotas")
# import modules required for function dependencies
import time
import os
import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import biotas

#set script parameters
site = '101'                                                                   # what is the site/receiver ID?
recType = 'orion'                                                              # what is the receiver type? (OMIT??)
proj_dir = r'C:\a\Projects\Winooski\2019\Data'                    # what is the project directory?
dbName = 'Winooski_2019_101.db'    
dbNew='Winooski_101_Trainer.db'    

class_iter='4'    #What was the last iteration during the classification process


biotas.createTrainDB(proj_dir, dbNew)  #This is where we'll stick the table.  For now this retains the full database structure.  I don't think that's necessary

#out_name='tblTrain'

projectDB = os.path.join(proj_dir,'Data',dbName)  #This is where the data will come from
outputDB = os.path.join(proj_dir,'Data',dbNew)    #And this is where it's going



train = biotas.create_training_data(site,projectDB)  #This is where we call the function. This function outputs a new training table based on only known noise from the initial (raw) data input and the observations that were classified as good during the final iteration

conn = sqlite3.connect(outputDB)
c = conn.cursor()
train.to_sql('tblTrain',con=conn,index = False, if_exists = 'append', chunksize = 1000)  #Copying this based on what I think it does...please check!
c.close()      
