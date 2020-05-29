# -*- coding: utf-8 -*-
"""
classification stats
"""

# import modules required for function dependencies
import time
import os
import biotas
import warnings
warnings.filterwarnings('ignore')
tS = time.time()
#set script parameters
site = 'T21'                                                               # what is the site/receiver ID?
class_iter= 2 #Enter the iteration number here--start at 2
recType = 'orion'                                                          # what is the receiver type?
proj_dir = r'C:\Users\Kevin Nebiolo\Desktop\Articles for Submission\Ted'                        # what is the project directory?
dbName = 'manuscript.db'                                                    # what is the name of the project database?
# directory creations
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')
class_stats = biotas.classification_results(recType,projectDB,figure_ws)#,site)#,reclass_iter = class_iter)
class_stats.classify_stats()    

