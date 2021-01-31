# import modules
import biotas
import os
import sqlite3

# declare workspaces
proj_dir = r'C:\a\Projects\BCHydro\Data'                                      # what is the project directory?
dbName = 'SiteC_2020.db'                                                    # what is the name of the project database
outputWS = os.path.join(proj_dir,'Output')
projectDB = os.path.join(proj_dir,'Data',dbName)

rec_list = ['F33a','F33b','F34','F35','F36a','F36b','F37','F38','F39','F40']

# the big merge
#recaptures = biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True, rec_list = rec_list)
recaptures = biotas.the_little_merge(outputWS,projectDB, pre_release_Filter = True, rec_list = rec_list)


recaptures.to_sql('tblRecaptures',sqlite3.connect(projectDB),if_exists ='append')

recaptures.to_csv(os.path.join(outputWS,'recaptures.csv'))

rec_by_fish = recaptures.groupby(['FreqCode'])['recID'].unique()

rec_by_fish.to_csv(os.path.join(outputWS,'rec_by_fish.csv'))

