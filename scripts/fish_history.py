'''Module creates a 3d plot to view fish history''' 
# import modules
import biotas.biotas as biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'D:\Manuscript\CT_River_2015'       # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                       # whad did you call the database?
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['T05','T06','T07','T08']
# Step 1, create fish history object
fishHistory = biotas.fish_history(projectDB,filtered = True, overlapping = False, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '149.800 168'
fishHistory.fish_plot(fish)