import time
import os
import numpy as np
import abtas
import warnings
warnings.filterwarnings('ignore')
'''Script implements a k-fold cross validation procedure on a specific receiver 
type.  This is meant to test the validity of the training data collected at a 
specific receiver or receiver type.'''
if __name__ == "__main__":
    t0 = time.time()
    # What receiver type are you assessing accuracy for?
    recType = 'orion'                                                          # what is the receiver type?
    proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                             # what is the raw data directory
    dbName = 'ultrasound_2018_test.db'                                                    # what is the name of the project database
    figureWS = os.path.join(proj_dir,'Output','Figures')    
    projectDB = os.path.join(proj_dir,'Data',dbName)
    k = 10
    # ['conRecLength','consDet','hitRatio','noiseRatio','seriesHit','power','lagDiff']
    fields = ['conRecLength','hitRatio','power','lagBDiff']
    # create cross validated data object
    cross = abtas.cross_validated(k,recType,fields,projectDB,figureWS)
    print ("Created a cross validated data object")
    # perform the cross validation method
    for i in np.arange(0,k,1):
        cross.fold(i)
    # print the summary
    cross.summary()
    # If the data has already been classified, run the classification stats
    print ("process took %s to compile"%(round(time.time() - t0,3)))