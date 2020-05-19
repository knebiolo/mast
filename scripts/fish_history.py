'''Module creates a 3d plot to view fish history''' 
# import modules
import biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'C:\Users\Kevin Nebiolo\Desktop\Articles for Submission\Ted'       # what is the project directory?
dbName = 'manuscript.db'                                                       # whad did you call the database?
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['T02','T04','T05','T06','T07','T10']
# Step 1, create fish history object
fishHistory = biotas.fish_history(projectDB,filtered = True, overlapping = False, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '149.420 47'
fishHistory.fish_plot(fish)