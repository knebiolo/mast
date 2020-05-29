# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'C:\Users\Kevin Nebiolo\Desktop\Articles for Submission\Ted'             # what is the project directory?
dbName = 'manuscript.db'                                                       # whad did you call the database?
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output')
modelName = "con_rec_ge6"
# what is the Node to State relationship - use Python dictionary 
receiver_to_recap = {'T10':'R00','T04':'R01','T05':'R01','T02':'R01','T03':'R01','T06':'R02','T07':'R03'}

# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = biotas.cjs_data_prep(receiver_to_recap,projectDB)
print ("Step 1 Completed, Data Class Finished")
# Step 2, Create input file for MARK
cjs.input_file(modelName,outputWS)
print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")