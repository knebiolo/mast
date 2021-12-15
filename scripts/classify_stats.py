# -*- coding: utf-8 -*-
"""
classification stats
"""

# import modules required for function dependencies
import time
import os
import biotas.biotas as biotas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()
#set script parameters
recs = ['T12W','T12E']
recType = 'orion'                                                          # what is the receiver type?
proj_dir = r'D:\Manuscript\CT_River_2015'                        # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                    # what is the name of the project database?
# directory creations
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')

class_stats = biotas.classification_results(recType,projectDB,figureWS = figure_ws,rec_list = recs)#,site)#,reclass_iter = class_iter)
class_stats.classify_stats()    

