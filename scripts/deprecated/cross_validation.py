import time
import os
import numpy as np
import biotas

#%% Part 1: Set up Cross Validation Script Parameters

# what is the receiver type?
recType = 'orion'
# what is the project directory?
proj_dir = r'E:\Manuscript\CT_River_2015'
# what did you call the database?
dbName = 'ctr_2015_v2.db'
# use OS tools to create directories
figureWS = os.path.join(proj_dir,'Output','Figures')
projectDB = os.path.join(proj_dir,'Data',dbName)
# How many folds in your cross validation?
k = 10

#%% Part 2: Cross Validate and Assess Model
t0 = time.time()

# A-la-carte likelihood, construct a model from the following parameters:
# ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
fields = ['conRecLength','lagDiff']

# create cross validated data object
cross = biotas.cross_validated(k,
                               recType,
                               fields,
                               projectDB,
                               train_on = 'Study',
                               rec_list = ['T03'])
print ("Created a cross validated data object")

# perform the cross validation method
for i in np.arange(0,k,1):
    cross.fold(i)
    print ("Classified fold %s"%i)
# print the summary
cross.summary()
print ("process took %s to compile"%(round(time.time() - t0,3)))