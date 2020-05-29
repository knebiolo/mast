'''Script Intent: The Intent of this script is to identify overlapping records 
to remove redundant detections and to positively place a fish.'''
# import modules
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
import time
if __name__ == "__main__":
    tS = time.time()
    # set up script parameters
    proj_dir = r'C:\Users\Kevin Nebiolo\Desktop\Articles for Submission\Ted'             # what is the project directory?
    dbName = 'manuscript.db'                                                       # whad did you call the database?
       
    outputWS = os.path.join(proj_dir,'Output','Scratch')
    figureWS = os.path.join(proj_dir,'Output','Figures')
    projectDB = os.path.join(proj_dir,'Data',dbName)
    # which node do you care about?
    nodes = ['S02','S03','S04','S05','S06','S07','S08']
    edges = [('S02','S03'),('S02','S04'),('S02','S05'),('S02','S06'),('S02','S07'),
             ('S06','S07'),
             ('S07','S05'),('S07','S06'),('S07','S04'),('S07','S02'),
             ('S08','S07'),('S08','S06'),('S08','S05'),('S08','S04'),
             ('S05','S07'),]
    # Step 1, create an overlap object
    print ("Start creating overlap data objects - 1 per node")
    iters = []
    for i in nodes:
        iters.append(abtas.overlap_reduction(i,nodes,edges,projectDB,outputWS,figureWS))
        print ("Completed overlap data object for node %s"%(i))
    print ("Start Multiprocessing")
    print ("This will take a while")
    print ("Grab a coffee, call your mother.")    
    for i in iters:
        abtas.russian_doll(i)

    print ("Overlap analysis complete proceed to data management")
    # Step 3, Manage Data   
    abtas.manage_node_overlap_data(outputWS,projectDB)
    print ("Overlap Removal Process complete, took %s seconds to compile"%(round(time.time() - tS,4)))
    del iters