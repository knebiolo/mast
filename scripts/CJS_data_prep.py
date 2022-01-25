# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'E:\Manuscript\CT_River_2015'             # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                       # whad did you call the database?
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?
outputWS = os.path.join(proj_dir,'Output')
modelName = "ds_canal"
# what is the Node to State relationship - use Python dictionary
receiver_to_recap = {'T23':'R00','T24':'R00',
                     'T22':'R01','T21':'R01','T18':'R01',
                     'T13':'R02',
                     'T14':'R03',
                     'T08':'R04','T09':'R04',
                     'T05':'R05','T06':'R05','T07':'R05',
                     'T15':'R06','T33':'R06','T03':'R06','T02':'R06'}

# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = biotas.cjs_data_prep(receiver_to_recap, projectDB, initial_recap_release = True)
print ("Step 1 Completed, Data Class Finished")
# Step 2, Create input file for MARK
cjs.input_file(modelName,outputWS)
print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

import biotas
proj_dir = r'D:\Manuscript\CT_River_2015'
dbName = 'ctr_2015_v3.db'
biotas.createTrainDB(proj_dir, dbName)