# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                            
dbName = 'ultrasound_2018_test.db'                                                   
projectDB = os.path.join(proj_dir,'Data',dbName)
# what is the output directory?                         
outputWS = os.path.join(proj_dir,'Output')
o_fileName_cov = "bypass_detail_cov.csv"
o_fileName = "bypass_detail.csv"
# what is the Node to State relationship - use Python dictionary 
node_to_state = {'S01':1,'S02':1,'S03':1,'S04':1,'S05':1,'S06':1,'S07':1,'S08':1,'S09':2,'S10':3,'S11':3,'S12':4}
recList = ['t01','t02','t03O','t03L','t05','t06','t07','t08','t09','t10','t11','t13']
# Step 1, create time to event data class - we only need to feed it the directory and file name of input data
tte = abtas.time_to_event(recList,(node_to_state),projectDB)
print ("Step 1 Complete, Data Class Finished")
# Step 2, format data - with covariates
tte.data_prep(os.path.join(outputWS,o_fileName_cov), time_dependent_covariates = True)
print ("Time to Event data formatted for time dependent covariates")
# Step 3, format data - without covariates
tte.data_prep(os.path.join(outputWS,o_fileName))
print ("Time to Event data formated without time dependent covariates")
# Step 4, generate a summary
tte.summary()
print ("Data formatting complete, proceed to R for Time to Event Modeling")