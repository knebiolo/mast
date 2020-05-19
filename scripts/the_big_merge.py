# import modules
import biotas
import os

# declare workspaces
proj_dir = r'C:\Users\Kevin Nebiolo\Desktop\Articles for Submission\Ted'             # what is the project directory?
dbName = 'manuscript.db'                                                       # whad did you call the database?
outputWS = os.path.join(proj_dir,'Data')
projectDB = os.path.join(proj_dir,'Data',dbName)

# the big merge
biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True, con_rec_filter = 6)