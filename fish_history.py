'''Module creates a 3d plot to view fish history''' 
# import modules
import sys
sys.path.append(r"J:\Jobs\1210\005\Calcs\Studies\3_3_19\2018\Program")
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir =  r'J:\1210\005\Calcs\Studies\3_3_19\2019'                             # what is the raw data directory
dbName = 'ultrasound_2019.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['T01','T02','T03E', 'T03W', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T12', 'T13', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T23', 'T24']
# Step 1, create fish history object
fishHistory = abtas.fish_history(projectDB,filtered = True, overlapping = False, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '148.340 34'
fishHistory.fish_plot(fish)