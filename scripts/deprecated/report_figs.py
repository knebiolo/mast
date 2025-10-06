import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas")

import time
import os
import biotas
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
# What receiver type are you assessing accuracy for?
recType = 'srx600'                                                              # what is the receiver type?
proj_dir = r'J:\2819\005\Calcs\MDN_Telemetry\Nuyakuk2023Telem'                                      # what is the project directory?
dbName = 'NuyakukFish.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
rec_list = ['1']

train_stats = biotas.training_results(recType,projectDB,figure_ws)#,'33')
train_stats.train_stats() 
class_stats = biotas.classification_results(recType,projectDB,figure_ws,rec_list = rec_list)#,reclass_iter = class_iter)
class_stats.classify_stats()  