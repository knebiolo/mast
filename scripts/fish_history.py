'''Module creates a 3d plot to view fish history''' 
# import modules
import sys
sys.path.append(r"J:\Jobs\1210\005\Calcs\Studies\3_3_19\2018\Program")
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir =  r'J:\1503\212\Calcs\Scotland_Fall2019'                             # what is the raw data directory
dbName = 'Scotland_Eel_2019.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['T01','T02','T03', 'T04', 'T05', 'T06', 'T07', 'T10']
# Step 1, create fish history object
fishHistory = abtas.fish_history(projectDB,filtered = True, overlapping = True, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '149.420 36'
fishHistory.fish_plot(fish)