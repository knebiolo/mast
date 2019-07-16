'''Module creates a 3d plot to view fish history''' 
# import modules
import sys
sys.path.append(r"J:\Jobs\1210\005\Calcs\Studies\3_3_19\2018\Program")
import abtas
import os
import warnings
warnings.filterwarnings('ignore')
# set up script parameters
proj_dir =  r'J:\1210\005\Calcs\Studies\3_3_19\2018\Test'                             # what is the raw data directory
dbName = 'ultrasound_2018_test.db'                                                    # what is the name of the project database
projectDB = os.path.join(proj_dir,'Data',dbName)
rec_list = ['t01','t02','t03O','t03L','t04','t05','t06','t07','t08','t09','t10','t11','t13']
# Step 1, create fish history object
fishHistory = abtas.fish_history(projectDB,filtered = False, overlapping = False, rec_list = rec_list)
print ("Fish History class object created, plotting now")
# Step 2, pick a fish, make a plot
fish = '150.500 160'
fishHistory.fish_plot(fish)