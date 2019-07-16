'''Script Intent: The Intent of this script is to identify overlapping records 
to remove redundant detections and to positively place a fish.'''
# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
import time
if __name__ == "__main__":
    tS = time.time()
    # set up script parameters
    project_dir = r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'
    dbName = 'ultrasound_2018_test.db'          
    outputWS = os.path.join(project_dir,'Output','Scratch')
    figureWS = os.path.join(project_dir,'Output','Figures')
    projectDB = os.path.join(project_dir,'Data',dbName)
    # which node do you care about?
    nodes = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
    edges = [('S01','S05'),('S01','S06'),('S01','S07'),('S01','S08'),('S01','S09'),
             ('S02','S05'),('S02','S06'),('S02','S07'),('S02','S08'),('S02','S09'),
             ('S03','S05'),('S03','S06'),('S03','S07'),('S03','S08'),('S03','S09'),
             ('S04','S05'),('S04','S06'),('S04','S07'),('S04','S08'),('S04','S09'),
             ('S09','S05'),('S09','S06'),('S09','S07'),('S09','S08'),
             ('S05','S08'),
             ('S06','S08'),
             ('S07','S08')]
    # Step 1, create an overlap object
    print ("Start creating overlap data objects - 1 per node")
    iters = []
    for i in nodes:
        iters.append(abtas.overlap_reduction(i,nodes,edges,projectDB,outputWS,figureWS))
        print ("Completed overlap data object for node %s"%(i))
    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")    
    pool = mp.Pool(processes = 8)                                              # the number of processes equals the number of processors you have
    pool.map(abtas.russian_doll, iters)                                         # map the russian doll function over each training data object 
    print ("Overlap analysis complete proceed to data management")
    # Step 3, Manage Data   
    abtas.manage_node_overlap_data(outputWS,projectDB)
    print ("Overlap Removal Process complete, took %s seconds to compile"%(round(time.time() - tS,4)))
    del iters