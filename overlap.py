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
    project_dir = r'C:\Users\Alex Malvezzi\Desktop'
    dbName = 'ultrasound_2019.db'          
    outputWS = os.path.join(project_dir,'Output','Scratch')
    figureWS = os.path.join(project_dir,'Output','Figures')
    projectDB = os.path.join(project_dir,'Data',dbName)
    # which node do you care about?
    nodes = ['S01','S02','S12','S13','S14','S15','S16','S17','S18','S19','S21','S22','S23','S24']
    edges = [('S02','S01'),
             ('S02','S01'),
             ('S13','S14'),('S13','S15'),('S13','S12'),('S13','S16'),('S13','S17'),
             ('S14','S15'),('S14','S16'),('S14','S17'),
             ('S16','S13'),('S16','S15'),('S16','S22'),('S16','S18'),('S16','S19'),('S16','S17'),('S16','S14'),('S16','S23'),('S16','S24'),('S16','S21'),
             ('S17','S18'),('S17','S19'),('S17','S21'),('S17','S13'),('S17','S24'),('S17','S14'),('S17','S22'),('S17','S23'),
             ('S18','S16'),('S18','S17'),
             ('S21','S16'),('S21','S17'),
             ('S22','S16'),
             ('S19','S18'),
             ('S22','S21'),('S22','S16'),('S22','S17'),
             ('S23','S16'),('S23','S17'),
             ('S24','S16'),('S24','S17')]
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