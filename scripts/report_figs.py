import time
import os
import biotas.biotas as biotas
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
# What receiver type are you assessing accuracy for?
recType = 'lotek'                                                              # what is the receiver type?
proj_dir = r'D:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
scratch_dir = os.path.join(proj_dir,'Output','Scratch')
figure_ws = os.path.join(proj_dir,'Output','Figures')
rec_list = ['T14']

train_stats = biotas.training_results(recType,projectDB,figure_ws,'T14')
train_stats.train_stats() 
class_stats = biotas.classification_results(recType,projectDB,figure_ws,rec_list = rec_list)#,reclass_iter = class_iter)
class_stats.classify_stats()  