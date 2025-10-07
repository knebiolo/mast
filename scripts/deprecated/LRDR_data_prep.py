# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'\\EGRET\Condor\Jobs\1210\005\Calcs\Studies\3_3_19\2019'                             # what is the raw data directory
dbName = 'ultrasound_2019_ajm.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output','LRDR_Prep')
modelName = "CabotRelease_Holyoke"
# what is the Node to State relationship - use Python dictionary 
receiver_to_recap = {'T18':'R00','T19':'R00','T03E':'R02','T03W':'R02','T04':'R02','T05':'R02','T06':'R02','T07':'R02','T01':'R04','T21':'R06'}  # stationary receiver to recapture occasion - EVEN NUMBERS ONLY
mobile_to_recap = {'M1':'R01','M2':'R03','M3':'R05','M4':'R07'}                # mobile reach to recapture occasion - ODD NUMBERS ONLY  
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
lrdr = abtas.lrdr_data_prep(receiver_to_recap,mobile_to_recap, projectDB, rel_loc = "Holyoke", cap_loc = "Holyoke", initial_recap_release = False, time_limit = 48.0)
print ("Step 1 Completed, Data Class Finished")
# Step 2, Create input file for MARK
lrdr.input_file(modelName,outputWS)
print ("Step 2 Completed, MARK Input file created")
print ("Data formatting complete, proceed to MARK for live recapture dead recovery modeling (LRDR)")