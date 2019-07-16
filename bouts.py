'''Script Intent: the intent of this script is to find the bout length at each 
node within our telemetry network.  

Currently our bout procedure uses a broken stick model, however there is literature 
that suggests the data can be assessed with a maximum likelihood procedure rather than 
our brute force method in current use'''
# import modules
import os
import warnings
import abtas
warnings.filterwarnings('ignore')
# set up script parameters
project_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                         # what is the name of the project database
inputWS = os.path.join(project_dir,'Data')                             
scratchWS = os.path.join(project_dir,'Output','Scratch')
figureWS = os.path.join(project_dir,'Output','Figures')
projectDB = os.path.join(inputWS,dbName)
# which node do you care about?
nodes = ['S00','S01','S02','S03','S04','S05','S06','S09','S10','S11','S12','S13']
#nodes = ['S13']
bout_len_dict = dict()
for i in nodes:   
    # Step 1, initialize the bout class object
    bout = abtas.bout(i,projectDB,lag_window = 15, time_limit = 3600)
    # Step 2: get fishes - we only care about test = 1 and hitRatio > 0.3
    fishes = bout.fishes    
    print ("Got a list of tags at this node")
    print ("Start Calculating Bout Length for node %s"%(i))
    # Step 3, find the knot if by site - 
    '''more appropriate to use bouts by site because the small lags we see within a 
    fish are are more than likely due to pulse randomization than a fish leaving 
    and coming back.  Perhaps when we get more precise clocks on the orion receivers
    we can we can calculate bouts by fish, but for now, use bouts per site'''
    boutLength_s = bout.broken_stick_3behavior(figureWS) 
    bout_len_dict[i] = boutLength_s
    # step 4 enumerate presence at this receiver for each fish
    for j in fishes:
        # Step 5i, enumerate presence at this receiver
        #boutLength_s = bout.broken_stick_fish(j)
        print ("Fish %s had a bout length of %s seconds at node %s"%(j,boutLength_s,i))
        bout.presence(j,boutLength_s,projectDB,scratchWS)      
    print ("Completed bout length calculation and enumerating presences for node %s"%(i))    
    # clean up your mess
    del bout
# manage all of that data, there's a lot of it!
abtas.manage_node_presence_data(scratchWS,projectDB)