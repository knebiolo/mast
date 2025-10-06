'''Script Intent: The Intent of this script is to identify overlapping records 
to remove redundant detections and to positively place a fish.'''
# import modules
import biotas.biotas as biotas
import os
import warnings
warnings.filterwarnings('ignore')
import time

tS = time.time()
# set up script parameters
proj_dir = r'D:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                         # whad did you call the database?
  
outputWS = os.path.join(proj_dir,'Output','Scratch')
figureWS = os.path.join(proj_dir,'Output','Figures')
projectDB = os.path.join(proj_dir,'Data',dbName)
# which node do you care about?
nodes = ['T08','T09','T05','T06','S2','T19','T24','T23','T25','T20','S3','T17','T15','T16']
edges = [('T08','T09'),('T08','T05'),('T08','T06'),('T08','S2'),
         ('T05','S2'),
         ('T06','T05'),('T06','S2'),
         ('T17','T15'),('T17','T16'),
         ('T19','T23'),('T19','S3'),
         ('T20','T23'),('T20','S3'),
         ('T23','T25'),('T23','S3'),
         ('T24','T25')]
# Step 1, create an overlap object
print ("Start creating overlap data objects - 1 per node")

for i in nodes:
    dat = biotas.overlap_reduction(i,nodes,edges,projectDB,outputWS,figureWS)
    print ("Completed overlap data object for node %s"%(i))
    biotas.russian_doll(dat)
    print ("Completed russian doll for node %s"%(i))
    del dat

print ("Overlap analysis complete proceed to data management")
# Step 3, Manage Data   
biotas.manage_node_overlap_data(outputWS,projectDB)
print ("Overlap Removal Process complete, took %s seconds to compile"%(round(time.time() - tS,4)))
