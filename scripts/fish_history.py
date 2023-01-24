'''Module creates a 3d plot to view fish history''' 
# import modules
import biotas as biotas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir = r'J:\1871\196\Calcs\BIOTAS\kpn_test'       # what is the project directory?
dbName = 'pepperell_am_eel.db'                                                       # whad did you call the database?
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['T1','T2','T3','T4','T5']
# Step 1, create fish history object
fishHistory = biotas.fish_history(projectDB,filtered = True, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '149.440 136'
fishHistory.fish_plot(fish)