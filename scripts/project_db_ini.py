import os
import biotas
import pandas as pd
proj_dir = r'C:\Users\knebiolo\Desktop\vr2_test'
dbName = 'vr2_test.db'
data_dir = os.path.join(proj_dir,'Data')
db_dir = os.path.join(proj_dir,'Data',dbName)
# number of detections (+/-) in the PDH
det = 5
# duration used in noise ratio calculation
duration = 1
# import data to Python
tblMasterTag = pd.read_csv(os.path.join(data_dir,'tblMasterTag.csv'))
tblMasterReceiver = pd.read_csv(os.path.join(data_dir,'tblMasterReceiver.csv'))
tblNodes = pd.read_csv(os.path.join(data_dir,'tblNodes.csv'))
# write data to SQLite
biotas.studyDataImport(tblMasterTag,db_dir,'tblMasterTag')
print ('tblMasterTag imported')
biotas.studyDataImport(tblMasterReceiver,db_dir,'tblMasterReceiver')
print ('tblMasterReceiver imported')
biotas.studyDataImport(tblNodes,db_dir,'tblNodes')
print ('tblNodes imported')
biotas.setAlgorithmParameters(det,duration,db_dir)
print ('tblAlgParams data entry complete, begin importing data and training')