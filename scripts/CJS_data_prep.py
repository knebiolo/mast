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
                     'T13':'R02','T14':'R02',
                     'T08':'R03','T09':'R03',
                     'T05':'R04','T06':'R04','T07':'R04',
                     'T33':'R05','T03':'R05','T02':'R05'}

# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = biotas.cjs_data_prep(receiver_to_recap, projectDB, species = 'Shad', initial_recap_release = True)
print ("Step 1 Completed, Data Class Finished")

# Step 2, Create input file for MARK
cjs.input_file(modelName,outputWS)
cjs.inp.to_csv(os.path.join(outputWS,'ds_canal_2.csv'), index = False)

print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")

