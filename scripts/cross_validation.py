import time
import os
import numpy as np
import sys
sys.path.append(r"U:\Software\biotas")
import biotas.biotas as biotas
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()

# What receiver type are you assessing accuracy for?
recType = 'orion'                                                              # what is the receiver type?
proj_dir = r'D:\Manuscript\CT_River_2015'             # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                       # whad did you call the database?
figureWS = os.path.join(proj_dir,'Output','Figures')    
projectDB = os.path.join(proj_dir,'Data',dbName)
k = 10

# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['power']

# create cross validated data object
cross = biotas.cross_validated(k,recType,fields,projectDB, train_on = 'Study',rec_list = ['T03','T06','T24'])
print ("Created a cross validated data object")

# perform the cross validation method
for i in np.arange(0,k,1):
    cross.fold(i)
    print ("Classified fold %s"%i)
# print the summary
cross.summary()
print ("process took %s to compile"%(round(time.time() - t0,3)))