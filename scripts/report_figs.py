import sys
sys.path.append(r"C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas")

import time
import os
import biotas
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
# What receiver type are you assessing accuracy for?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'J:\1871\196\Calcs\BIOTAS'                                      # what is the project directory?
dbName = 'Pepperell_v2.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
rec_list = ['T1']

train_stats = biotas.training_results(recType,projectDB,figure_ws)#,'33')
train_stats.train_stats() 
class_stats = biotas.classification_results(recType,projectDB,figure_ws,rec_list = rec_list)#,reclass_iter = class_iter)
class_stats.classify_stats()  