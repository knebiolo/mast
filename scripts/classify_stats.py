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
# identify receivers you want to summarize
recs = ['T03','T06','T24']
# what is the receiver type?
recType = 'lotek'                                                          
# what is the project directory?
proj_dir = r'E:\Manuscript\CT_River_2015'                        
# what is the name of the project database?
dbName = 'ctr_2015_v2.db'                                                    
# directory creations
projectDB = os.path.join(proj_dir,'Data',dbName)
figure_ws = os.path.join(proj_dir,'Output','Figures')

class_stats = biotas.classification_results(recType,
                                            projectDB,
                                            figureWS = figure_ws,
                                            rec_list = recs)
                                            #,site,)
                                            #,reclass_iter = class_iter)
class_stats.classify_stats()    

