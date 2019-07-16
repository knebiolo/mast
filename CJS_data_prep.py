# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                             # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Ouptut')
modelName = "MontagueToSpillway"
# what is the Node to State relationship - use Python dictionary 
receiver_to_recap = {'t01':'R0','t02':'R0','t03O':'R1','t03L':'R1','t04':'R2',
                     't05':'R2','t06':'R2','t07':'R2','t08':'R2','t09':'R3',
                     't10':'R4','t11':'R4','t13':'R5','t12':'R6'}
recList = ['t01','t02','t03O','t03L','t04','t05','t06','t07','t08','t09','t10',
           't11','t12','t13']
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
cjs = abtas.cjs_data_prep(recList,receiver_to_recap,projectDB)
print ("Step 1 Completed, Data Class Finished")
# Step 2, Create input file for MARK
cjs.input_file(modelName,outputWS)
print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture modeling (CJS)")