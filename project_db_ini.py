import os
import abtas
import pandas as pd
proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                       # what is the project directory?
dbName = 'ultrasound_2018_test.db'                                             # whad did you call the database?
data_dir = os.path.join(proj_dir,'Data')                                       
db_dir = os.path.join(proj_dir,'Data',dbName)                                  
det = 5                                                                        # number of detections we will look forwards and backwards for in detection history
duration = 1                                                                   # duration used in noise ratio calculation
# import data to Python
tblMasterTag = pd.read_csv(os.path.join(data_dir,'tblMasterTag.csv'))
tblMasterReceiver = pd.read_csv(os.path.join(data_dir,'tblMasterReceiver.csv'))
tblNodes = pd.read_csv(os.path.join(data_dir,'tblNodes.csv'))                  # no nodes?  then comment this line
# write data to SQLite
abtas.studyDataImport(tblMasterTag,db_dir,'tblMasterTag')
print ('tblMasterTag imported')
abtas.studyDataImport(tblMasterReceiver,db_dir,'tblMasterReceiver')
print ('tblMasterReceiver imported')
abtas.studyDataImport(tblNodes,db_dir,'tblNodes')                              # no nodes? then comment out this line
print ('tblNodes imported')                                                    # no nodes? then comment this line
abtas.setAlgorithmParameters(det,duration,db_dir)
print ('tblAlgParams data entry complete, begin importing data and training')